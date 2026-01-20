import os
import pandas as pd

PAIRS = ["GBPJPY", "EURJPY", "USDJPY"]
FX_DIR = "data/fx_bluesky"

NFP_DAILY_PATH = "data/fred/nfp_daily.csv"

IN_SUFFIX  = "_with_bluesky_cbank_emp.csv"
OUT_SUFFIX = "_with_bluesky_cbank_emp_nfp.csv"

def main():
    if not os.path.exists(NFP_DAILY_PATH):
        raise RuntimeError(f"{NFP_DAILY_PATH} が見つかりません。先に make_nfp_daily.py を実行してください。")

    nfp = pd.read_csv(NFP_DAILY_PATH)
    nfp["date"] = pd.to_datetime(nfp["date"])

    # 使いたい列だけ（必要なら増やしてOK）
    keep_cols = ["date", "NFP_FLAG", "NFP_SHOCK_ACTIVE", "PAYEMS_chg_ACTIVE"]
    nfp = nfp[keep_cols].copy()

    for pair in PAIRS:
        src = os.path.join(FX_DIR, f"{pair}{IN_SUFFIX}")
        if not os.path.exists(src):
            print(f"[SKIP] {pair}: {src} not found")
            continue

        df = pd.read_csv(src)
        if "date" not in df.columns:
            raise RuntimeError(f"{pair}: date列がありません: {src}")

        df["date"] = pd.to_datetime(df["date"])
        out = df.merge(nfp, on="date", how="left")

        # 非イベント日は0
        for c in ["NFP_FLAG", "NFP_SHOCK_ACTIVE", "PAYEMS_chg_ACTIVE"]:
            if c in out.columns:
                out[c] = out[c].fillna(0.0)

        dst = os.path.join(FX_DIR, f"{pair}{OUT_SUFFIX}")
        out.to_csv(dst, index=False)
        print("saved:", dst, "rows=", len(out))

if __name__ == "__main__":
    main()