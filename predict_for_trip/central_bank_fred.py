# central_bank_fred.py
import os
import requests
import pandas as pd
from datetime import datetime

# ==== 設定 ====

# FREDのAPIキー（環境変数に入れておくのがおすすめ）
FRED_API_KEY = os.environ.get("FRED_API_KEY", "<ここに一時的に直書きもOK>")

# 取得したい期間
DATE_START = "2022-01-01"
DATE_END   = "2025-12-01"

# ここはあとで実際のFREDシリーズIDに差し替える
# （FREDのサイトで検索して確認してください）
SERIES_IDS = {
    "US_POLICY": "FEDFUNDS",     # 米：FF金利（例）
    "EU_POLICY": "ECBMAIN",      # 欧：ECB main refinancing rate など（要確認）
    "UK_POLICY": "BOERATE",      # 英：Bank Rate（要確認）
    "JP_POLICY": "INTDSRJPM193N" # 日：政策金利系（要確認）
}

OUT_DIR = "data/central_bank"
os.makedirs(OUT_DIR, exist_ok=True)


# ==== FREDからシリーズを1本取ってくる関数 ====

def fetch_fred_series(series_id: str,
                      start: str = DATE_START,
                      end: str = DATE_END) -> pd.DataFrame:
    """
    FREDのシリーズIDを指定して日次データを取得する。
    返り値: DataFrame[date, value]
    """
    if FRED_API_KEY in (None, "", "<ここに一時的に直書きもOK>"):
        raise RuntimeError("FRED_API_KEY が設定されていません。環境変数かスクリプト内で指定してください。")

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
        # "."（欠損）が来ることがあるので処理
        try:
            val = float(value_str)
        except ValueError:
            val = None
        rows.append({"date": date_str, "value": val})

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["value"]).reset_index(drop=True)
    return df


def build_central_bank_daily(start: str = DATE_START, end: str = DATE_END) -> pd.DataFrame:
    """
    SERIES_IDS で指定した各政策金利をまとめて日次テーブルにする。
    日付でインデックスを作り、各列を forward-fill した形に整形。
    """
    # 日次の骨組み
    idx = pd.date_range(start=start, end=end, freq="D")
    base = pd.DataFrame(index=idx)
    base.index.name = "date"

    for label, sid in SERIES_IDS.items():
        print(f"Fetching {label} ({sid}) ...")
        s_df = fetch_fred_series(sid, start, end)
        s_df = s_df.set_index("date").sort_index()
        # 日次にreindexして前日値で埋める
        s_daily = s_df.reindex(idx).ffill()
        base[label] = s_daily["value"]

    base = base.reset_index()
    return base


def main():
    df = build_central_bank_daily()
    out_csv = os.path.join(OUT_DIR, "central_bank_daily.csv")
    out_parquet = os.path.join(OUT_DIR, "central_bank_daily.parquet")

    df.to_csv(out_csv, index=False)
    try:
        df.to_parquet(out_parquet)
    except Exception as e:
        print(f"(parquet保存はスキップ: {e})")

    print("Saved central bank daily data to:", out_csv)


if __name__ == "__main__":
    main()