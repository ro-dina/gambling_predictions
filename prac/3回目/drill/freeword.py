import re
import numpy as np
import pandas as pd

from datetime import datetime, timezone, timedelta
from dateutil.relativedelta import relativedelta
from atproto import Client
from atproto_client.models.app.bsky.feed.search_posts import Params

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import os
from dotenv import load_dotenv

load_dotenv()  # .env を読み込む

ACCOUNT = os.getenv("BSKY_EMAIL")
PASSWORD = os.getenv("BSKY_APP_PASSWORD")

if not ACCOUNT or not PASSWORD:
    raise RuntimeError("環境変数 BSKY_EMAIL / BSKY_APP_PASSWORD が読み込めていません")

KWD='EU'
def search_posts(
        query: str,
        since_date: str, until_date: str ,
        email: str, app_password: str,
        per_page: int = 100, max_pages: int = 50,
):

    client = Client()
    client.login(email, app_password)

    cursor = None
    results = []
    for _ in range(max_pages):
        params = Params(q=query, limit=per_page, cursor=cursor,
                        since=since_date, until=until_date)
        res = client.app.bsky.feed.search_posts(params=params)
        posts = res.posts or []
        results.extend(posts)
        cursor = getattr(res, "cursor", None)
        if not cursor or not posts:
            break
    return results

def posts_to_df(posts):
    rows = []
    for p in posts:
        record = getattr(p, "record", {}) or {}
        text = getattr(record, "text", None) or record.get("text", "")
        created_at = getattr(record, "created_at", None) or record.get("createdAt")

        if created_at:
            # createdAt は ISO 8601 文字列
            dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            ts = int(dt.timestamp())  # Unix time (秒)
        else:
            ts = None

        rows.append({
            "unixtime": ts,
            "uri": getattr(p, "uri", None),
            "handle": getattr(getattr(p, "author", None), "handle", None),
            "text": text,
        })

    df = pd.DataFrame(rows).dropna(subset=["unixtime"])
    df = df.set_index("unixtime").sort_index()
    return df


# --- 1) クリーニング & トークナイズ ----------------------------
def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r'https?://\S+', ' ', s)       # URL除去
    s = re.sub(r'@[A-Za-z0-9_\.]+', ' ', s)   # @mention簡易除去
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def _tokenize_with_fugashi(text: str):
    try:
        from fugashi import Tagger
        tagger = Tagger()
        return [w.surface for w in tagger(text)]
    except Exception:
        return None  # 失敗時はフォールバックへ

def _tokenize_fallback(text: str):
    # 日本語（空白が少ない）なら文字2-gram、英数混在なら空白split
    if ' ' not in text:
        chars = [c for c in re.sub(r'\s+', '', text)]
        return [a+b for a, b in zip(chars, chars[1:])] or chars
    else:
        return re.findall(r"[A-Za-z0-9_#@]+|[一-龥ぁ-んァ-ヶー]+", text)

def tokenize(text: str):
    text = clean_text(text)
    toks = _tokenize_with_fugashi(text)
    return toks if toks else _tokenize_fallback(text)

# --- 2) 週（JST）にまとめて TaggedDocument 化 -------------------
def make_weekly_documents(df: pd.DataFrame,
                          text_col: str = "text",
                          jst: bool = True,
                          rule: str = "W-MON"):
    """
    df: index=unixtime(秒) の DataFrame を想定。textカラム必須。
    jst=TrueならJST(UTC+9)に変換して週集計（rule=W-MON: 月曜始まり）
    """
    assert text_col in df.columns, f"'{text_col}' column not found"
    s = df[text_col].fillna("").astype(str)

    # index(unixtime秒) → datetime（UTC）
    dt_utc = pd.to_datetime(df.index, unit="s", utc=True)
    if jst:
        tz = timezone(timedelta(hours=9))
        dt = dt_utc.tz_convert(tz)
    else:
        dt = dt_utc

    tmp = pd.DataFrame({"text": s, "dt": dt})
    # 週ごとにテキスト結合
    weekly = tmp.resample(rule, on="dt")["text"].apply(lambda x: "\n".join(x)).dropna()
    # 週の代表日時（期首）を index に
    weekly.index.name = "week_start"

    # TaggedDocument 化
    docs = []
    tags = []
    for ws, text in weekly.items():
        tokens = tokenize(text)
        tag = f"week_{ws.strftime('%Y-%m-%d')}"
        docs.append(TaggedDocument(words=tokens, tags=[tag]))
        tags.append((ws, tag))
    return docs, tags  # tags: [(Timestamp, "week_YYYY-MM-DD"), ...]

# --- 3) Doc2Vec を訓練 → 週ベクトルの DataFrame -----------------
def train_doc2vec_and_vectorize(docs,
                                vector_size: int = 128,
                                window: int = 10,
                                min_count: int = 2,
                                epochs: int = 40,
                                dm: int = 1,
                                seed: int = 42):
    model = Doc2Vec(vector_size=vector_size,
                    window=window,
                    min_count=min_count,
                    workers=4,
                    epochs=epochs,
                    dm=dm,
                    seed=seed)
    model.build_vocab(docs)
    model.train(docs, total_examples=len(docs), epochs=model.epochs)
    return model

def weekly_vectors_df(model: Doc2Vec, tags):
    # tags: [(week_start_ts, "week_YYYY-MM-DD"), ...]
    rows = []
    index = []
    for ws, tag in tags:
        vec = model.dv[tag]  # 学習済み → 直接取り出し
        rows.append(vec)
        index.append(ws)
    mat = np.vstack(rows) if rows else np.zeros((0, model.vector_size))
    dfv = pd.DataFrame(mat, index=pd.DatetimeIndex(index, name="week_start"))
    return dfv.sort_index()

# --- 4) 使い方（例） -------------------------------------------

def main():
    since_date = '2024-07-31T15:00:00Z'
    until_date = '2025-08-31T15:00:00Z'

    posts = search_posts(
        KWD,
        since_date,
        until_date,
        ACCOUNT,
        PASSWORD,
        100, 50,
    )

    posts_df = posts_to_df(posts)
    print(posts_df)

    docs, tags = make_weekly_documents(posts_df, text_col="text", jst=True, rule="W-MON")
    model = train_doc2vec_and_vectorize(docs, vector_size=128, window=10, min_count=2, epochs=40, dm=1)
    vecdf = weekly_vectors_df(model, tags)
    print(vecdf.shape)   # (週数, 128)
    print(vecdf.head())  # 週ごとの特徴ベクトル（時系列）


if __name__ == "__main__":
    main()