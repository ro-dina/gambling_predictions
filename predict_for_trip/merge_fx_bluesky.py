import os
import pandas as pd

# ===== パスの設定 =====

# rate.py が出力した Bluesky 日次集計
BLUESKY_DAILY_PATH = "data/bluesky_daily/bluesky_daily_features.csv"

# 為替CSVの場所（※ここを自分のファイル名に合わせて直してください）
FX_CSV_PATHS = {
    # 例: "ペア": "CSVのパス"
    "GBPJPY": "data/fx/EURJPY/EURJPY_20200101_20251201.csv",
    "EURJPY": "data/fx/GBPJPY/GBPJPY_20200101_20251201.csv",
    "USDJPY": "data/fx/USDJPY/USDJPY_20200101_20251201.csv",
}

# 出力先ディレクトリ（withBlueSky.py が読む場所）
OUT_DIR = "data/fx_bluesky"


def load_bluesky_daily(path: str) -> pd.DataFrame:
    """Bluesky日次特徴量を読み込む."""
    bs = pd.read_csv(path)
    bs["date"] = pd.to_datetime(bs["date"]).dt.date

    # 必須列が足りなければ0で埋める（保険）
    for col, fill in [
        ("post_count", 0),
        ("sent_mean", 0.0),
        ("sent_std", 0.0),
        ("hike_count", 0),
        ("cut_count", 0),
    ]:
        if col not in bs.columns:
            bs[col] = fill

    return bs


def merge_one_pair(pair: str, fx_path: str, bs_daily: pd.DataFrame):
    """1つの通貨ペアについて、FX CSV と Bluesky 日次を date でマージして保存."""
    print(f"=== merging {pair} ===")
    fx = pd.read_csv(fx_path)

    # 為替側の date 列を datetime.date に
    fx["date"] = pd.to_datetime(fx["date"]).dt.date

    # 必須列名チェック（value 列が為替レート列になっている前提）
    if "value" not in fx.columns:
        raise RuntimeError(f"{fx_path} に 'value' 列がありません。為替レート列の名前を 'value' に揃えてください。")

    # left join: 為替の日付をベースに Bluesky 情報をくっつける
    merged = fx.merge(bs_daily, on="date", how="left")

    # Blueskyのない日は NaN → 0 埋め
    int_cols = ["post_count", "hike_count", "cut_count"]
    float_cols = ["sent_mean", "sent_std"]

    for col in int_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0).astype(int)
        else:
            merged[col] = 0

    for col in float_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.0)
        else:
            merged[col] = 0.0

    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DIR, f"{pair}_with_bluesky.csv")
    out_parquet = os.path.join(OUT_DIR, f"{pair}_with_bluesky.parquet")

    merged.to_csv(out_csv, index=False)
    try:
        merged.to_parquet(out_parquet)
    except Exception as e:
        print(f"(parquet保存はスキップ: {e})")

    print(" -> saved:", out_csv)


def main():
    bs_daily = load_bluesky_daily(BLUESKY_DAILY_PATH)

    for pair, fx_path in FX_CSV_PATHS.items():
        if not os.path.exists(fx_path):
            print(f"WARNING: {fx_path} が見つかりません。スキップします。")
            continue
        merge_one_pair(pair, fx_path, bs_daily)


if __name__ == "__main__":
    main()