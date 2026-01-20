import os
import requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

FRED_API_KEY = os.environ.get("FRED_API_KEY")
if not FRED_API_KEY:
    raise RuntimeError("FRED_API_KEY が設定されていません")

RELEASE_DATES_PATH = "data/fred/nfp_release_dates.csv"
OUT_DAILY_PATH = "data/fred/nfp_daily.csv"
OUT_EVENTS_PATH = "data/fred/nfp_events.csv"

SERIES_ID = "PAYEMS"

DATE_START = "2010-01-01"
DATE_END   = "2026-12-31"

def fetch_fred_series(series_id: str, start: str, end: str) -> pd.DataFrame:
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
    j = r.json()
    obs = j.get("observations", [])
    df = pd.DataFrame(obs)
    if df.empty:
        raise RuntimeError(f"{series_id}: observations が空でした")
    df["date"] = pd.to_datetime(df["date"])
    # valueは "." が来ることがあるので数値化
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).sort_values("date").reset_index(drop=True)
    return df[["date", "value"]].rename(columns={"value": series_id})

def main():
    if not os.path.exists(RELEASE_DATES_PATH):
        raise RuntimeError(f"{RELEASE_DATES_PATH} が見つかりません。先に fred_nfp_release_dates.py を実行してください。")

    rel = pd.read_csv(RELEASE_DATES_PATH)
    rel["date"] = pd.to_datetime(rel["date"])
    rel = rel[(rel["date"] >= DATE_START) & (rel["date"] <= DATE_END)].sort_values("date").reset_index(drop=True)

    payems = fetch_fred_series(SERIES_ID, DATE_START, DATE_END)

    # PAYEMS_chg（月次差分）と「差分の変化（ショック）」を月次で作る
    payems["PAYEMS_chg"] = payems[SERIES_ID].diff()
    payems["PAYEMS_chg_prev"] = payems["PAYEMS_chg"].shift(1)
    payems["NFP_SHOCK"] = payems["PAYEMS_chg"] - payems["PAYEMS_chg_prev"]

    # release dateごとに「その時点で利用可能な最新PAYEMS」を割り当て
    # （雇用統計発表日に新しい月次値が出る想定）
    rel = rel.rename(columns={"date": "release_date"})
    # asofマージ：release_date <= PAYEMS.date の直前（PAYEMS.dateは月初）なので、
    # release_dateに対して PAYEMSの最新観測（月初）を拾う
    merged = pd.merge_asof(
        rel.sort_values("release_date"),
        payems.sort_values("date"),
        left_on="release_date",
        right_on="date",
        direction="backward",
    )

    # イベントテーブル（発表日ごとの特徴）
    events = pd.DataFrame({
        "date": merged["release_date"].dt.date,
        "PAYEMS": merged[SERIES_ID],
        "PAYEMS_chg": merged["PAYEMS_chg"],
        "NFP_SHOCK": merged["NFP_SHOCK"],
    })

    # 欠損は0扱い（初期はdiffがNaNになりやすい）
    events["PAYEMS_chg"] = events["PAYEMS_chg"].fillna(0.0)
    events["NFP_SHOCK"] = events["NFP_SHOCK"].fillna(0.0)

    # 日次カレンダーを作る
    daily = pd.DataFrame({"date": pd.date_range(DATE_START, DATE_END, freq="D")})
    daily["date"] = daily["date"].dt.date

    # NFP_FLAG（発表日=1）
    daily = daily.merge(events[["date"]].assign(NFP_FLAG=1), on="date", how="left")
    daily["NFP_FLAG"] = daily["NFP_FLAG"].fillna(0).astype(int)

    # イベント値を日次へ（発表日にだけ値が立つ）
    daily = daily.merge(events, on="date", how="left")

    # 非イベント日は0（学習に入れやすい）
    for c in ["PAYEMS", "PAYEMS_chg", "NFP_SHOCK"]:
        daily[c] = daily[c].fillna(0.0)

    # イベント日にだけ効かせる特徴
    daily["NFP_SHOCK_ACTIVE"] = daily["NFP_FLAG"] * daily["NFP_SHOCK"]
    daily["PAYEMS_chg_ACTIVE"] = daily["NFP_FLAG"] * daily["PAYEMS_chg"]

    os.makedirs(os.path.dirname(OUT_DAILY_PATH), exist_ok=True)
    daily.to_csv(OUT_DAILY_PATH, index=False)
    events.to_csv(OUT_EVENTS_PATH, index=False)

    print("saved:", OUT_DAILY_PATH, "rows=", len(daily))
    print("saved:", OUT_EVENTS_PATH, "rows=", len(events))
    print("events sample:")
    print(events.tail())

if __name__ == "__main__":
    main()