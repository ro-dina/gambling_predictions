import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

FRED_API_KEY = os.environ.get("FRED_API_KEY")
if not FRED_API_KEY:
    raise RuntimeError("FRED_API_KEY が設定されていません（.env / export を確認）")

# Employment Situation の release_id（FRED側）
NFP_RELEASE_ID = 50

OUT_PATH = "data/fred/nfp_release_dates.csv"

def fetch_release_dates(start="2010-01-01", end="2026-12-31"):
    url = "https://api.stlouisfed.org/fred/release/dates"
    params = {
        "release_id": NFP_RELEASE_ID,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "include_release_dates_with_no_data": "false",
        "realtime_start": start,
        "realtime_end": end,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()
    dates = j.get("release_dates", [])
    df = pd.DataFrame(dates)
    if df.empty:
        raise RuntimeError("release_dates が空でした。APIキー/制限/endpoint を確認してください。")
    # df["date"] は YYYY-MM-DD
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df[["date"]]

def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df = fetch_release_dates()
    df.to_csv(OUT_PATH, index=False)
    print("saved:", OUT_PATH, "rows=", len(df))

if __name__ == "__main__":
    main()