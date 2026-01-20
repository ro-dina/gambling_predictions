# add_nfp_flag_emp.py
import os
import pandas as pd
import numpy as np

FX_DIR = "data/fx_bluesky"
PAIRS = ["GBPJPY", "EURJPY", "USDJPY"]

OUT_SUFFIX = "_with_bluesky_cbank_emp_nfp.csv"


def add_flag_for_pair(pair: str):
    src = os.path.join(FX_DIR, f"{pair}_with_bluesky_cbank_emp.csv")
    if not os.path.exists(src):
        print(f"[SKIP] {pair}: {src} not found")
        return

    df = pd.read_csv(src)
    if "date" not in df.columns or "PAYEMS" not in df.columns:
        print(f"[WARN] {pair}: 必要な列(date, PAYEMS)がありません")
        return

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # PAYEMS差分とダミー
    df["PAYEMS_chg"] = df["PAYEMS"].diff().fillna(0.0)
    df["NFP_FLAG"] = (df["PAYEMS_chg"].abs() > 0).astype(int)

    out_path = os.path.join(FX_DIR, f"{pair}{OUT_SUFFIX}")
    df.to_csv(out_path, index=False)
    print(f"=== {pair} ===")
    print(df[["date", "PAYEMS", "PAYEMS_chg", "NFP_FLAG"]].head(10))
    print(f" -> saved: {out_path}")


def main():
    for p in PAIRS:
        add_flag_for_pair(p)


if __name__ == "__main__":
    main()