# add_nfp_flag.py
import pandas as pd
import os

PAIRS = ["GBPJPY", "EURJPY", "USDJPY"]
FX_DIR = "data/fx_bluesky"
CB_PATH = "data/central_bank/payems_daily.csv"
OUT_DIR = "data/fx_bluesky_nfp"

os.makedirs(OUT_DIR, exist_ok=True)

def main():
    cb = pd.read_csv(CB_PATH)
    cb["date"] = pd.to_datetime(cb["date"])

    # PAYEMSの変化日を抽出
    cb["payems_chg"] = cb["PAYEMS"].diff()
    cb["NFP_FLAG"] = (cb["payems_chg"].abs() > 0).astype(int)

    for pair in PAIRS:
        fx_path = os.path.join(FX_DIR, f"{pair}_with_bluesky.csv")
        if not os.path.exists(fx_path):
            print(f"[WARN] {fx_path} が見つからないのでスキップ")
            continue

        print(f"=== merging NFP flag into {pair} ===")
        df = pd.read_csv(fx_path)
        df["date"] = pd.to_datetime(df["date"])

        merged = pd.merge(df, cb[["date","NFP_FLAG"]], on="date", how="left").sort_values("date")

        # 欠損は0埋め（PAYEMS未更新日）
        merged["NFP_FLAG"] = merged["NFP_FLAG"].fillna(0).astype(int)

        out_path = os.path.join(OUT_DIR, f"{pair}_with_nfp.csv")
        merged.to_csv(out_path, index=False)
        print(f" -> saved: {out_path}")

if __name__ == "__main__":
    main()