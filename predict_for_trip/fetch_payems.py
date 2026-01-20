# fetch_payems.py
import os
import requests
import pandas as pd

OUT_DIR = "data/fred"
SERIES_ID = "PAYEMS"  # Nonfarm Payrolls: All Employees, Total Nonfarm, Monthly, Level
API_URL = "https://api.stlouisfed.org/fred/series/observations"


def get_fred_api_key() -> str:
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("環境変数 FRED_API_KEY が設定されていません。")
    return api_key


def fetch_payems(start_date: str = "2000-01-01", end_date: str = "2025-12-31") -> pd.DataFrame:
    api_key = get_fred_api_key()
    params = {
        "series_id": SERIES_ID,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
        "observation_end": end_date,
    }
    resp = requests.get(API_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    obs = data.get("observations", [])
    rows = []
    for o in obs:
        date = o["date"]          # "YYYY-MM-DD"
        value_str = o["value"]    # "." or number as string
        if value_str == ".":
            val = None
        else:
            try:
                val = float(value_str)
            except ValueError:
                val = None
        rows.append({"date": date, "PAYEMS": val})

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").dropna(subset=["PAYEMS"]).reset_index(drop=True)
    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = fetch_payems(start_date="2000-01-01", end_date="2025-12-31")
    out_path = os.path.join(OUT_DIR, "PAYEMS.csv")
    df.to_csv(out_path, index=False)
    print(f"saved PAYEMS to: {out_path}")
    print(df.head())
    print(df.tail())


if __name__ == "__main__":
    main()