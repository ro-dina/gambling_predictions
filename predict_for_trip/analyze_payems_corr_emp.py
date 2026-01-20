# analyze_payems_corr_emp.py
import os
import numpy as np
import pandas as pd

FX_DIR = "data/fx_bluesky"
PAIRS = ["GBPJPY", "EURJPY", "USDJPY"]

# 必要なら期間を合わせる
DATE_START = "2021-01-01"
DATE_END = "2025-12-01"


def analyze_pair(pair: str):
    path = os.path.join(FX_DIR, f"{pair}_with_bluesky_cbank_emp.csv")
    if not os.path.exists(path):
        print(f"[SKIP] {pair}: {path} not found")
        return

    df = pd.read_csv(path)
    if "date" not in df.columns or "value" not in df.columns or "PAYEMS" not in df.columns:
        print(f"[WARN] {pair}: 必要な列(date,value,PAYEMS)がありません")
        return

    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= DATE_START) & (df["date"] <= DATE_END)].sort_values("date")

    # logリターンと翌日リターン
    df["log_ret"] = np.log(df["value"]) - np.log(df["value"].shift(1))
    df["log_ret_next"] = df["log_ret"].shift(-1)

    # PAYEMSの変化量（レベル差 or ％変化どちらでもOK）
    df["PAYEMS_chg"] = df["PAYEMS"].diff()         # 差分
    # df["PAYEMS_chg_pct"] = df["PAYEMS"].pct_change()  # %変化にしたければこっち

    # 相関を計算（NaNは落とす）
    cols = ["log_ret_next", "log_ret", "PAYEMS", "PAYEMS_chg"]
    sub = df[cols].dropna()

    if len(sub) == 0:
        print(f"[WARN] {pair}: 有効なデータがありません")
        return

    corr = sub.corr()
    print(f"\n=== ANALYSIS: {pair} ===")
    print(corr["log_ret_next"])


def main():
    for p in PAIRS:
        analyze_pair(p)


if __name__ == "__main__":
    main()