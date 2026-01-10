import os
from datetime import datetime
from typing import List, Dict

from dotenv import load_dotenv
import pandas as pd
from atproto import Client

load_dotenv()

HANDLE = os.environ["BSKY_EMAIL"]
APP_PASSWORD = os.environ["BSKY_APP_PASSWORD"]

# 収集したいアカウント（display nameではなく handle or did）
TARGET_HANDLES = [             
      "ec.europa.eu",
      "politico.eu",
      "aoc.bsky.social"
]


def login_client() -> Client:
    client = Client()
    client.login(HANDLE, APP_PASSWORD)
    return client


def fetch_all_posts_of_user(client: Client, handle: str, limit: int = 5000) -> List[Dict]:
    """
    特定ユーザ(handle)の投稿を全部（最大limit件）取得して辞書リストで返す。
    """
    profile = client.get_profile(handle)
    did = profile.did

    posts: List[Dict] = []
    cursor = None

    while True:
        resp = client.get_author_feed(actor=did, cursor=cursor, limit=100)
        feed = resp.feed

        if not feed:
            break

        for item in feed:
            post = item.post
            record = post.record

            text = getattr(record, "text", "")
            created_at = getattr(record, "created_at", None) or getattr(record, "createdAt", None)

            posts.append(
                {
                    "handle": handle,
                    "uri": post.uri,
                    "cid": post.cid,
                    "created_at": created_at,
                    "text": text,
                }
            )

        cursor = resp.cursor
        if not cursor:
            break
        if len(posts) >= limit:
            break

    return posts


def normalize_posts_to_df(posts: List[Dict]) -> pd.DataFrame:
    if not posts:
        return pd.DataFrame(columns=["handle", "uri", "cid", "created_at", "date", "text"])

    df = pd.DataFrame(posts)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["created_at"]).reset_index(drop=True)

    # 日付(UTC)を日次のキーにする
    df["date"] = df["created_at"].dt.date

    # 文字列型に揃える
    df["handle"] = df["handle"].astype("string")
    df["uri"] = df["uri"].astype("string")
    df["cid"] = df["cid"].astype("string")
    df["text"] = df["text"].astype("string")

    return df


def main():
    out_dir = "data/bluesky_accounts"
    os.makedirs(out_dir, exist_ok=True)

    client = login_client()

    all_rows = []

    for h in TARGET_HANDLES:
        print(f"fetching posts for: {h}")
        posts = fetch_all_posts_of_user(client, h, limit=5000)
        print(f"  -> got {len(posts)} posts")
        all_rows.extend(posts)

    df = normalize_posts_to_df(all_rows)
    print("total rows:", len(df))

    # CSV / Parquetで保存
    today = datetime.utcnow().strftime("%Y%m%d")
    csv_path = os.path.join(out_dir, f"bluesky_accounts_{today}.csv")
    parquet_path = os.path.join(out_dir, f"bluesky_accounts_{today}.parquet")

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_parquet(parquet_path)

    print("saved:", csv_path)
    print("saved:", parquet_path)


if __name__ == "__main__":
    main()