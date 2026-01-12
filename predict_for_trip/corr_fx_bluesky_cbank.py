# corr_fx_bluesky_cbank.py
import os
import pandas as pd
import numpy as np

PAIRS = ["GBPJPY", "EURJPY", "USDJPY"]
BASE_DIR = "data/fx_bluesky"

def analyze_pair(pair: str):
    path = os.path.join(BASE_DIR, f"{pair}_with_bluesky_cbank.csv")
    if not os.path.exists(path):
        print(f"[SKIP] {path} not found")
        return

    print(f"\n=== ANALYSIS: {pair} ===")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # log return
    df["log_ret"] = np.log(df["value"]) - np.log(df["value"].shift(1))
    df["log_ret_next"] = df["log_ret"].shift(-1)

    # 主要特徴候補
    cols = ["log_ret", "post_count", "sent_mean",
            "US_POLICY", "EU_POLICY", "UK_POLICY", "JP_POLICY"]

    # スプレッド（外 - 日本）
    if pair == "USDJPY":
        df["spread"] = df["US_POLICY"] - df["JP_POLICY"]
    elif pair == "EURJPY":
        df["spread"] = df["EU_POLICY"] - df["JP_POLICY"]
    elif pair == "GBPJPY":
        df["spread"] = df["UK_POLICY"] - df["JP_POLICY"]
    cols.append("spread")

    # 変化率(差分)
    df["spread_chg"] = df["spread"] - df["spread"].shift(1)
    cols.append("spread_chg")

    df = df.dropna(subset=["log_ret", "log_ret_next"])

    # 使える列だけ抽出
    use_cols = [c for c in cols if c in df.columns]
    corr = df[use_cols + ["log_ret_next"]].corr()["log_ret_next"].sort_values(ascending=False)

    print(corr)


def main():
    for p in PAIRS:
        analyze_pair(p)


if __name__ == "__main__":
    main()