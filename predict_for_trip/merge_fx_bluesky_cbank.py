# merge_fx_bluesky_cbank.py
import os
import pandas as pd

CB_PATH = "data/central_bank/central_bank_daily.csv"
FX_DIR = "data/fx_bluesky"

PAIRS = ["GBPJPY", "EURJPY", "USDJPY"]


def main():
    cb = pd.read_csv(CB_PATH)
    cb["date"] = pd.to_datetime(cb["date"])

    for pair in PAIRS:
        fx_path = os.path.join(FX_DIR, f"{pair}_with_bluesky.csv")
        out_path = os.path.join(FX_DIR, f"{pair}_with_bluesky_cbank.csv")
        if not os.path.exists(fx_path):
            print(f"[WARN] {fx_path} が見つかりません。スキップします。")
            continue

        print(f"=== merging {pair} ===")
        df = pd.read_csv(fx_path)
        # 既に datetime になっていなければ変換
        if "date" not in df.columns:
            raise RuntimeError(f"{fx_path} に 'date' 列がありません。")

        df["date"] = pd.to_datetime(df["date"])

        merged = pd.merge(df, cb, on="date", how="left").sort_values("date")

        # 政策金利列の欠損を前方埋め / 後方埋めし、それでも残る NaN は 0 で埋める
        for col in ["US_POLICY", "EU_POLICY", "UK_POLICY", "JP_POLICY"]:
            if col in merged.columns:
                merged[col] = (
                    merged[col]
                    .astype(float)
                    .ffill()
                    .bfill()
                    .fillna(0.0)
                )

        # 元ファイルはそのままにし、中央銀行列を足したものを別ファイルとして出力
        merged.to_csv(out_path, index=False)
        print(f" -> saved (merged with central bank): {out_path}")


if __name__ == "__main__":
    main()