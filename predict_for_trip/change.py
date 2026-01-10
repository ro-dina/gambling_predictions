import os
import pandas as pd

# ========= 設定 =========
BLUESKY_PATH = "data/bluesky_accounts/bluesky_accounts_with_sent.csv"
# FXのCSVを保存したパス（例）
# 例: data/fx/GBPJPY/fx_GBPJPY_20200101_20251231.csv
FX_BASE_DIR = "data/fx"

# 結合したい通貨ペア（保存したCSVに合わせて調整）
PAIR_LIST = [
    ("GBP", "JPY"),
    ("EUR", "JPY"),
    ("USD", "JPY"),
]

# ========================


def load_bluesky_daily(path: str) -> pd.DataFrame:
    """Bluesky投稿(with sent_score)を日次に集約"""
    print("Loading Bluesky data from:", path)
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    # date列がある前提（ingest→normalize→negaposiの流れ）
    df["date"] = pd.to_datetime(df["date"]).dt.date

    # 日次集計：投稿数・平均/分散など
    daily = (
        df.groupby("date")
          .agg(
              post_count=("text", "size"),
              sent_mean=("sent_score", "mean"),
              sent_std=("sent_score", "std"),
          )
          .reset_index()
    )

    print("Bluesky daily rows:", len(daily))
    return daily


def load_fx_csv(base: str, target: str) -> pd.DataFrame:
    """FX CSV を読み出す。ファイル名は自分のものに合わせて調整してOK。"""
    pair = f"{base}{target}"
    # ここはあなたが保存したファイル名に合わせてパスを変えてください
    # 例: data/fx/GBPJPY/fx_GBPJPY_20200101_20251231.csv
    fx_dir = os.path.join(FX_BASE_DIR, pair)
    # ディレクトリ内の最初のcsvを使う例（1個だけならこれでOK）
    csv_files = [f for f in os.listdir(fx_dir) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"{fx_dir} に CSV が見つかりません")

    csv_path = os.path.join(fx_dir, csv_files[0])
    print(f"Loading FX data for {pair} from:", csv_path)

    fx_df = pd.read_csv(csv_path)
    fx_df["date"] = pd.to_datetime(fx_df["date"]).dt.date

    # 例: 列が ["date", "value"] になっている想定
    return fx_df


def merge_fx_bluesky_for_pair(base: str, target: str, daily_bs: pd.DataFrame) -> pd.DataFrame:
    """ある通貨ペアについて FX と Bluesky を日付で結合して返す"""
    pair = f"{base}{target}"
    fx_df = load_fx_csv(base, target)

    # left join: 為替の日付を基準に Bluesky をくっつける
    merged = fx_df.merge(daily_bs, on="date", how="left")

    # Blueskyがない日は NaN → 0 埋め
    merged["post_count"] = merged["post_count"].fillna(0).astype(int)
    merged["sent_mean"] = merged["sent_mean"].fillna(0.0)
    merged["sent_std"] = merged["sent_std"].fillna(0.0)

    print(f"{pair}: merged rows = {len(merged)}")

    # 保存
    out_dir = os.path.join("data", "fx_bluesky")
    os.makedirs(out_dir, exist_ok=True)
    out_path_csv = os.path.join(out_dir, f"{pair}_with_bluesky.csv")
    out_path_parquet = os.path.join(out_dir, f"{pair}_with_bluesky.parquet")

    merged.to_csv(out_path_csv, index=False)
    merged.to_parquet(out_path_parquet)

    print("saved:", out_path_csv)
    print("saved:", out_path_parquet)

    return merged


def main():
    daily_bs = load_bluesky_daily(BLUESKY_PATH)

    for base, target in PAIR_LIST:
        merge_fx_bluesky_for_pair(base, target, daily_bs)


if __name__ == "__main__":
    main()