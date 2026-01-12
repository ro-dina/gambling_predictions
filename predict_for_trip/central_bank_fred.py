# central_bank_fred.py
import os
from datetime import datetime

import requests
import pandas as pd
from dotenv import load_dotenv

# .env を読み込む（FRED_API_KEY を取り込む）
load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")

if not FRED_API_KEY:
    raise RuntimeError("FRED_API_KEY が .env に設定されていません。 .env に FRED_API_KEY=... を追加してください。")

# ================= 設定 =================

# 期間は FX モデルと揃えておく
DATE_START = "2022-01-01"
DATE_END   = "2025-12-01"

# 取ってくるシリーズID（必要に応じて差し替えOK）
SERIES_IDS = {
    "US_POLICY": "FEDFUNDS",     # Federal Funds Effective Rate (US)
    "EU_POLICY": "ECBMRRFR",     # ECB Main Refinancing Operations Rate
    "UK_POLICY": "BOERUKM",      # Bank of England Policy Rate
    "JP_POLICY": "INTDSRJPM193N" # Discount Rate for Japan
}

OUT_DIR = "data/central_bank"


# ============== FRED から 1シリーズ取る関数 ==============

def fetch_fred_series(series_id: str,
                      start: str = DATE_START,
                      end: str = DATE_END) -> pd.DataFrame:
    """
    FREDの series_id を指定して観測データを取得し、
    DataFrame[date, value] を返す。
    """
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end,
    }

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    obs = data.get("observations", [])
    rows = []
    for o in obs:
        date_str = o["date"]
        value_str = o["value"]
        # 欠損は "." で来ることがある
        try:
            val = float(value_str)
        except ValueError:
            val = None
        rows.append({"date": date_str, "value": val})

    df = pd.DataFrame(rows)
    # 観測値が1件もなかった場合の対処
    if df.empty:
        print(f"Warning: no observations returned for series {series_id} between {start} and {end}")
        # 空の DataFrame だが、後段で参照する 'date'/'value' 列は必ず持たせておく
        return pd.DataFrame({"date": pd.to_datetime([]), "value": []})
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["value"]).reset_index(drop=True)
    return df


# ============== 全てまとめて日次テーブルにする関数 ==============

def build_central_bank_daily(start: str = DATE_START,
                             end: str = DATE_END) -> pd.DataFrame:
    """
    SERIES_IDS で指定した各シリーズを取得して、
    日次インデックスに揃え、forward-fill したテーブルを作る。

    戻り値: DataFrame[date, US_POLICY, EU_POLICY, UK_POLICY, JP_POLICY]
    """
    # 日次の骨組み
    idx = pd.date_range(start=start, end=end, freq="D")
    base = pd.DataFrame(index=idx)
    base.index.name = "date"

    for label, sid in SERIES_IDS.items():
        print(f"Fetching {label} ({sid}) ...")
        s_df = fetch_fred_series(sid, start, end)
        if s_df.empty:
            # このシリーズにはデータがないので、列全体を NaN で埋めてスキップ
            print(f"Warning: {label} ({sid}) has no data in the specified period; filling with NaN.")
            base[label] = pd.NA
            continue
        s_df = s_df.set_index("date").sort_index()

        # 元系列の頻度が月次・日次どちらでも、とりあえず日次に reindex して前日値で埋める
        s_daily = s_df.reindex(idx).ffill()
        base[label] = s_daily["value"]

    base = base.reset_index()
    return base


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = build_central_bank_daily(DATE_START, DATE_END)

    out_csv = os.path.join(OUT_DIR, "central_bank_daily.csv")
    out_parquet = os.path.join(OUT_DIR, "central_bank_daily.parquet")

    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)

    # parquet は環境によっては pyarrow が必要なので try/except にしておく
    try:
        df.to_parquet(out_parquet)
        print("Saved:", out_parquet)
    except Exception as e:
        print("Parquet 保存はスキップしました:", e)


if __name__ == "__main__":
    main()
