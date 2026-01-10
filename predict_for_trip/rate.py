# make_bs_features.py
import pandas as pd
import numpy as np
import os
import re

BLUESKY_PATH = "data/bluesky_accounts/bluesky_accounts_with_sent.parquet"

RATE_HIKE_KEYWORDS = [
    "利上げ", "金利引き上げ", "rate hike", "raise rates", "raised rates",
    "tightening", "hawkish",
]
RATE_CUT_KEYWORDS = [
    "利下げ", "金利引き下げ", "rate cut", "cut rates", "lowered rates",
    "easing", "dovish",
]

def pattern(words):
    # 正規表現の OR パターンを作る（大文字小文字無視）
    escaped = [re.escape(w) for w in words]
    return "|".join(escaped)

def main():
    if BLUESKY_PATH.endswith(".parquet"):
        df = pd.read_parquet(BLUESKY_PATH)
    else:
        df = pd.read_csv(BLUESKY_PATH)

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["text"] = df["text"].fillna("").astype(str)

    # キーワードフラグ
    hike_pat = pattern(RATE_HIKE_KEYWORDS)
    cut_pat  = pattern(RATE_CUT_KEYWORDS)

    df["is_hike"] = df["text"].str.contains(hike_pat, case=False, na=False)
    df["is_cut"]  = df["text"].str.contains(cut_pat,  case=False, na=False)

    # 日次集計
    daily = (
        df.groupby("date")
          .agg(
              post_count=("text", "size"),
              sent_mean=("sent_score", "mean"),
              sent_std=("sent_score", "std"),
              hike_count=("is_hike", "sum"),
              cut_count=("is_cut", "sum"),
          )
          .reset_index()
    )

    daily["sent_std"] = daily["sent_std"].fillna(0.0)

    out_dir = "data/bluesky_daily"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "bluesky_daily_features.csv")
    daily.to_csv(out_path, index=False)
    print("saved:", out_path)

if __name__ == "__main__":
    main()