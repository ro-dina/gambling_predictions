# analyze_payems_corr.py
import pandas as pd
import os

BASE_DIR = "data/fx_bluesky"
CB_PATH = "data/central_bank/payems_daily.csv"
PAIRS = ["GBPJPY", "EURJPY", "USDJPY"]

def main():
    cb = pd.read_csv(CB_PATH)
    cb["date"] = pd.to_datetime(cb["date"])

    for pair in PAIRS:
        fx_path = os.path.join(BASE_DIR, f"{pair}_with_bluesky.csv")
        if not os.path.exists(fx_path):
            print(f"[WARN] {fx_path} がないのでスキップ")
            continue

        print(f"\n=== ANALYSIS: {pair} ===")
        df = pd.read_csv(fx_path)
        df["date"] = pd.to_datetime(df["date"])

        merged = pd.merge(df, cb, on="date", how="left").sort_values("date")
        merged["PAYEMS"] = merged["PAYEMS"].ffill()

        # log return & next return を作成
        merged["log_ret"] = (merged["value"].apply(float)).apply(lambda x: pd.Series([pd.np.log(x)]))
        merged["log_ret"] = merged["log_ret"].diff()
        merged["log_ret_next"] = merged["log_ret"].shift(-1)

        merged["payems_chg"] = merged["PAYEMS"].pct_change()

        # 相関
        cols = ["log_ret_next", "log_ret", "PAYEMS", "payems_chg"]
        corr = merged[cols].corr()
        print(corr["log_ret_next"])

if __name__ == "__main__":
    main()