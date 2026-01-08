import re
import numpy as np
import pandas as pd

from datetime import datetime, timezone, timedelta
from atproto import Client
from atproto_client.models.app.bsky.feed.search_posts import Params

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import os
from dotenv import load_dotenv

from typing import Any, Iterable, Optional, Tuple, List
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

load_dotenv()  # .env を読み込む

ACCOUNT = os.getenv("BSKY_EMAIL")
PASSWORD = os.getenv("BSKY_APP_PASSWORD")

if not ACCOUNT or not PASSWORD:
    raise RuntimeError("環境変数 BSKY_EMAIL / BSKY_APP_PASSWORD が読み込めていません")

KWD='百鬼あやめ'

def search_posts(
        query: str,
        since_date: str, until_date: str,
        email: str, app_password: str,
        per_page: int = 100, max_pages: int = 50,
) -> List[Any]:

    client = Client()
    client.login(email, app_password)

    cursor = None
    results: List[Any] = []
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

from pytrends.request import TrendReq

def fetch_google_trends(keyword: str, start: str, end: str, geo: str = "JP") -> pd.DataFrame:
    """
    start/end: 'YYYY-MM-DD'（JST想定でOK）
    戻り値: index=日付（datetime64[ns]） / columns=['trend']
    """
    pytrends = TrendReq(hl="ja-JP", tz=540)  # tz=540: JST
    timeframe = f"{start} {end}"
    pytrends.build_payload([keyword], timeframe=timeframe, geo=geo)

    df = pytrends.interest_over_time()
    if df.empty:
        raise RuntimeError("Google Trends が空です（キーワード/期間/geoを確認）")

    # keyword列を trend に統一
    out = df[[keyword]].rename(columns={keyword: "trend"}).copy()
    out.index = pd.to_datetime(out.index).tz_localize(None)  # timezone無しに
    # isPartial があれば落とす
    if "isPartial" in df.columns:
        out = out.loc[~df["isPartial"]].copy() if df["isPartial"].dtype == bool else out
    return out


def _get_record_field(record: Any, attr_name: str, dict_key: str) -> Optional[Any]:
    """Support both Pydantic model objects and dict-like records."""
    if record is None:
        return None
    # Pydantic / object
    if hasattr(record, attr_name):
        return getattr(record, attr_name)
    # dict
    if isinstance(record, dict):
        return record.get(dict_key)
    return None


def _get_post_text_and_created_at(record: Any) -> Tuple[str, Optional[str]]:
    text = _get_record_field(record, "text", "text")
    created_at = (
        _get_record_field(record, "created_at", "createdAt")
        or _get_record_field(record, "createdAt", "createdAt")
    )
    return (str(text) if text is not None else ""), (str(created_at) if created_at is not None else None)

def posts_to_df(posts: Iterable[Any]) -> pd.DataFrame:
    rows: List[dict[str, Any]] = []
    for p in posts:
        record = getattr(p, "record", None)
        text, created_at = _get_post_text_and_created_at(record)

        if created_at:
            # createdAt は ISO 8601 文字列
            try:
                dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                ts = int(dt.timestamp())  # Unix time (秒)
            except Exception:
                ts = None
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

# ここがUnix time とJSTの関係を調べてみよう。このプログラムでは、どこでその処理を行っているか確認だろう
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



def make_sequences(X: np.ndarray, y: np.ndarray, lookback: int = 14):
    xs, ys = [], []
    for i in range(lookback, len(X)):
        xs.append(X[i-lookback:i])   # 過去lookback日
        ys.append(y[i])              # 当日（または i+1 にして翌日予測でもOK）
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).unsqueeze(-1)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.lstm(x)          # (B, T, H)
        last = out[:, -1, :]           # 最終時刻
        return self.fc(last)           # (B, 1)

def train_lstm_on_merged(data: pd.DataFrame, lookback: int = 14, epochs: int = 20, batch_size: int = 32, return_preds: bool = False):
    # X: Doc2Vec列, y: trend
    y = data["trend"].values.astype(np.float32)
    X = data.drop(columns=["trend"]).values.astype(np.float32)

    # 標準化（学習データでfit → testに適用）
    # 時系列なので後ろをtestにする
    n = len(X)
    split = int(n * 0.8)
    if n < (lookback + 2):
        raise ValueError(
            f"データが少なすぎます: n={n}, lookback={lookback}. "
            f"対処: (1) 期間を伸ばす/keyword変更でnを増やす, (2) lookbackを下げる(週次なら1〜3), (3) join後のdata行数を確認"
        )
    if split <= lookback or (n - split) <= lookback:
        raise ValueError(
            f"train/testが小さすぎるか lookback が大きすぎます: n={n}, split={split}, lookback={lookback}. "
            f"対処: lookbackを下げる or split比率を変える（例: train_ratio=0.7） or 期間を伸ばす"
        )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[:split])
    X_test  = scaler.transform(X[split:])

    y_train = y[:split]
    y_test  = y[split:]

    y_scaler = StandardScaler()
    y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_s  = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

    Xtr_seq, ytr_seq = make_sequences(X_train, y_train_s, lookback)
    Xte_seq, yte_seq = make_sequences(X_test,  y_test_s,  lookback)

    train_loader = DataLoader(SeqDataset(Xtr_seq, ytr_seq), batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(SeqDataset(Xte_seq, yte_seq), batch_size=batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMRegressor(n_features=X.shape[1], hidden=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for ep in range(1, epochs+1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(xb)

        model.eval()
        te_loss = 0.0
        preds = []
        trues = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                te_loss += loss_fn(pred, yb).item() * len(xb)
                preds.append(pred.cpu().numpy().ravel())
                trues.append(yb.cpu().numpy().ravel())

        preds_epoch = np.concatenate(preds) if preds else np.array([])
        trues_epoch = np.concatenate(trues) if trues else np.array([])
        rmse = float(np.sqrt(np.mean((preds_epoch - trues_epoch) ** 2))) if len(preds_epoch) else float("nan")
        print(f"epoch {ep:02d} train_mse={tr_loss/len(Xtr_seq):.4f} test_mse={te_loss/max(1,len(Xte_seq)):.4f} test_rmse={rmse:.3f}")

    # --- Final prediction on test (for plotting) ---
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = model(xb)
            preds.append(pred.cpu().numpy().ravel())
            trues.append(yb.cpu().numpy().ravel())

    preds = np.concatenate(preds) if preds else np.array([], dtype=np.float32)
    trues = np.concatenate(trues) if trues else np.array([], dtype=np.float32)

    # テスト系列の対応日付（週次なら週の代表日）
    # X_test は data.index[split:] に対応し、make_sequences は最初の lookback 個を捨てる
    dates_test = pd.to_datetime(data.index[split + lookback:]).tz_localize(None)
    preds_orig = y_scaler.inverse_transform(preds.reshape(-1, 1)).ravel()
    trues_orig = y_scaler.inverse_transform(trues.reshape(-1, 1)).ravel()

    if return_preds:
        return model, dates_test, preds_orig, trues_orig

    return model

def plot_actual_vs_pred(dates: pd.DatetimeIndex, actual: np.ndarray, pred: np.ndarray, title: str = "Google Trends: actual vs predicted"):
    if len(actual) == 0 or len(pred) == 0:
        print("No predictions to plot (empty arrays).")
        return

    # 長さがズレた場合に備えて揃える
    m = min(len(actual), len(pred), len(dates))
    dates = dates[:m]
    actual = actual[:m]
    pred = pred[:m]

    plt.figure(figsize=(10, 4))
    plt.plot(dates, actual, label="actual")
    plt.plot(dates, pred, label="pred")
    plt.title(title)
    plt.xlabel("date")
    plt.ylabel("trend")
    plt.legend()
    plt.tight_layout()
    plt.show()

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

    docs, tags = make_weekly_documents(posts_df, text_col="text", jst=True, rule="W-MON") #<- 週
    #docs, tags = make_weekly_documents(posts_df, text_col="text", jst=True, rule="D") #<- 日
    d2v_model = train_doc2vec_and_vectorize(docs, vector_size=128, window=10, min_count=2, epochs=40, dm=1)
    vecdf = weekly_vectors_df(d2v_model, tags)

    # === Google Trends 取得（同じキーワードで） ===
    # ※期間は trend の都合で短めの方が日次が取りやすいことがあります
    trend_start = "2024-08-01"
    trend_end   = "2025-08-31"
    trends_df = fetch_google_trends(KWD, trend_start, trend_end, geo="JP")  # 'trend' 列
    trends_df.index = pd.to_datetime(trends_df.index).normalize()

    # === vecdf を「日付」で揃える（JSTにしてから date に落とす） ===
    # vecdf.index は tz-aware の可能性があるので date に統一
    x = vecdf.copy()
    x.index = pd.to_datetime(x.index).tz_localize(None)  # 念のため tz を外す
    x.index = x.index.normalize()  # 00:00 に揃える

    # === 結合（同じ日付に揃える） ===
    # Trends を月曜始まりの週に揃える（平均でも最後の値でもOK）
    trends_w = trends_df.resample("W-MON").mean()
    trends_w.index = trends_w.index.normalize()
    data = x.join(trends_w, how="inner").dropna()
    print("merged:", data.shape, "columns:", data.columns[-5:])  # 最後に trend があるはず
    if len(data) > 0:
        print("merged date range:", data.index.min(), "->", data.index.max())
        print("merged weeks:", len(data))

    print(vecdf.shape)   # (週数, 128)
    print(vecdf.head())  # 週ごとの特徴ベクトル（時系列）
    # === LSTM 学習 ===
    # 週次データが少ないと lookback=4 でも学習サンプルが作れないことがあるので自動調整
    n = len(data)
    split = int(n * 0.8)
    max_lb_train = max(1, split - 1)
    max_lb_test = max(1, (n - split) - 1)
    lookback = min(4, max_lb_train, max_lb_test)
    print(f"auto lookback={lookback} (n={n}, split={split})")

    lstm_model, dates_test, preds, trues = train_lstm_on_merged(data, lookback=lookback, epochs=20, batch_size=32, return_preds=True)
    plot_actual_vs_pred(dates_test, trues, preds, title=f"Google Trends '{KWD}' (test)")


if __name__ == "__main__":
    main()