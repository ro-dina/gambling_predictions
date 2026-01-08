# ingest.py
import os
import pandas as pd
from dotenv import load_dotenv
import argparse
from pathlib import Path

from atproto import Client
from atproto_client.models.app.bsky.feed.search_posts import Params

from datetime import datetime

load_dotenv()
ACCOUNT = os.getenv("BSKY_EMAIL")
PASSWORD = os.getenv("BSKY_APP_PASSWORD")
if not ACCOUNT or not PASSWORD:
    raise RuntimeError("環境変数 BSKY_EMAIL / BSKY_APP_PASSWORD が読み込めていません")

DEFAULT_QUERY = "円安 OR 円高 OR ドル円 OR 為替 OR USDJPY OR EURJPY"

def normalize_iso_z(s: str) -> str:
    """Accept 'YYYY-MM-DD' or ISO strings and return an ISO string ending with 'Z'."""
    s = s.strip()
    if len(s) == 10 and s[4] == '-' and s[7] == '-':
        # Date only -> treat as 00:00:00Z
        return s + "T00:00:00Z"
    # If it already ends with Z, keep it
    if s.endswith("Z"):
        return s
    # If it has timezone offset, keep it as-is
    if "+" in s or s.endswith("+00:00"):
        return s
    # Otherwise assume it's UTC without Z
    return s + "Z"

def search_posts(query, since_date, until_date, email, app_password, per_page=100, max_pages=20):
    client = Client()
    client.login(email, app_password)

    cursor = None
    results = []
    for _ in range(max_pages):
        params = Params(q=query, limit=per_page, cursor=cursor, since=since_date, until=until_date)
        res = client.app.bsky.feed.search_posts(params=params)
        posts = res.posts or []
        results.extend(posts)
        cursor = getattr(res, "cursor", None)
        if not cursor or not posts:
            break
        # Be polite to the API
        if cursor:
            import time
            time.sleep(0.2)
    return results

def fetch_author_posts(handle: str, email: str, app_password: str,
                        per_page: int = 100, max_pages: int = 50):
    client = Client()
    client.login(email, app_password)

    # handle -> DID 解決（必要になることが多い）
    resolved = client.app.bsky.actor.get_profile({"actor": handle})
    actor = resolved.did  # ここがDID

    cursor = None
    results = []
    for _ in range(max_pages):
        res = client.app.bsky.feed.get_author_feed({
            "actor": actor,
            "limit": per_page,
            "cursor": cursor,
        })
        feed = res.feed or []
        for item in feed:
            post = getattr(item, "post", None)
            if post is not None:
                results.append(post)
        cursor = getattr(res, "cursor", None)
        if not cursor or not feed:
            break

    return results

def posts_to_df(posts) -> pd.DataFrame:
    """Convert Bluesky post objects into a typed DataFrame.

    Returns an empty typed DataFrame when no rows are available.
    Index: unixtime (int64)
    Columns:
      - createdAt (datetime64[ns, UTC])
      - date (datetime64[ns])  # UTC date (normalized to 00:00)
      - uri (string)
      - handle (string)
      - text (string)
    """
    rows = []

    for p in posts or []:
        record = getattr(p, "record", None) or {}

        # text
        text = getattr(record, "text", None)
        if text is None and isinstance(record, dict):
            text = record.get("text", "")
        if text is None:
            text = ""

        # createdAt (ISO)
        created_at = (
            getattr(record, "created_at", None)
            or getattr(record, "createdAt", None)
            or (record.get("createdAt") if isinstance(record, dict) else None)
        )
        if not created_at:
            continue

        # parse createdAt safely
        try:
            dt = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
        except Exception:
            continue

        # ensure timezone-aware (UTC)
        if dt.tzinfo is None:
            # treat as UTC if missing tz
            dt = dt.replace(tzinfo=datetime.now().astimezone().tzinfo)

        ts = int(dt.timestamp())

        rows.append({
            "unixtime": ts,
            "createdAt": dt,
            "uri": getattr(p, "uri", None),
            "handle": getattr(getattr(p, "author", None), "handle", None),
            "text": text,
        })

    # Empty -> return empty typed DF
    if not rows:
        empty = pd.DataFrame({
            "unixtime": pd.Series(dtype="int64"),
            "createdAt": pd.Series(dtype="datetime64[ns, UTC]"),
            "uri": pd.Series(dtype="string"),
            "handle": pd.Series(dtype="string"),
            "text": pd.Series(dtype="string"),
        }).set_index("unixtime")
        empty["date"] = pd.Series(dtype="datetime64[ns]")
        return empty

    df = pd.DataFrame(rows)

    # types
    df["unixtime"] = pd.to_numeric(df["unixtime"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["unixtime"]).copy()
    df["unixtime"] = df["unixtime"].astype("int64")

    df["createdAt"] = pd.to_datetime(df["createdAt"], utc=True, errors="coerce")
    df = df.dropna(subset=["createdAt"]).copy()

    df["uri"] = df["uri"].astype("string")
    df["handle"] = df["handle"].astype("string")
    df["text"] = df["text"].astype("string")

    # UTC day bucket
    df["date"] = df["createdAt"].dt.floor("D").dt.tz_localize(None)

    df = df.set_index("unixtime").sort_index()
    return df


# Daily aggregate function
def daily_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate post-level data into daily features.

    Output index: date (naive datetime, UTC day)
    Columns:
      - post_count (int64)
      - unique_authors (int64)
      - avg_text_len (float64)
    """
    if df is None or df.empty:
        out = pd.DataFrame({
            "date": pd.Series(dtype="datetime64[ns]"),
            "post_count": pd.Series(dtype="int64"),
            "unique_authors": pd.Series(dtype="int64"),
            "avg_text_len": pd.Series(dtype="float64"),
        }).set_index("date")
        return out

    tmp = df.reset_index(drop=False).copy()
    # text length (safe for pandas string dtype)
    tmp["text_len"] = tmp["text"].fillna("").astype("string").str.len().astype("int64")

    g = tmp.groupby("date", dropna=True)
    out = pd.DataFrame({
        "post_count": g.size().astype("int64"),
        "unique_authors": g["handle"].nunique(dropna=True).astype("int64"),
        "avg_text_len": g["text_len"].mean().astype("float64"),
    })

    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    return out

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ingest Bluesky posts for FX/travel research")
    p.add_argument("--query", default=DEFAULT_QUERY, help="Bluesky search query")
    p.add_argument("--since", default="2024-07-31", help="Start date/time (YYYY-MM-DD or ISO)")
    p.add_argument("--until", default="2025-08-31", help="End date/time (YYYY-MM-DD or ISO)")
    p.add_argument("--out", default="data", help="Output directory")
    p.add_argument("--per-page", type=int, default=100, help="Posts per page (max 100)")
    p.add_argument("--max-pages", type=int, default=20, help="Max pages to fetch")
    p.add_argument("--csv", action="store_true", help="Also save CSV")
    p.add_argument("--daily", action="store_true", help="Also save daily aggregated features")
    return p

def main():
    args = build_arg_parser().parse_args()

    since_date = normalize_iso_z(args.since)
    until_date = normalize_iso_z(args.until)

    posts = search_posts(args.query, since_date, until_date, ACCOUNT, PASSWORD,
                         per_page=args.per_page, max_pages=args.max_pages)
    df = posts_to_df(posts)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = out_dir / "raw_posts.parquet"
    df.to_parquet(parquet_path)

    if args.csv:
        csv_path = out_dir / "raw_posts.csv"
        df.reset_index().to_csv(csv_path, index=False, encoding="utf-8-sig")

    if args.daily:
        daily_df = daily_aggregate(df)
        daily_parquet_path = out_dir / "daily_posts.parquet"
        daily_df.to_parquet(daily_parquet_path)
        if args.csv:
            daily_csv_path = out_dir / "daily_posts.csv"
            daily_df.reset_index().to_csv(daily_csv_path, index=False, encoding="utf-8-sig")
        print("saved daily:", daily_df.shape, "->", str(daily_parquet_path))

    if df.empty:
        print("saved:", df.shape, "(empty)", "->", str(parquet_path))
    else:
        print("saved:", df.shape, "date range:", df["date"].min(), "..", df["date"].max(), "->", str(parquet_path))

if __name__ == "__main__":
    main()
   