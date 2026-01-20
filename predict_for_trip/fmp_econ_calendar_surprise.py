# fmp_econ_calendar_surprise.py
import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()  

FMP_API_KEY = os.environ.get("FMP_API_KEY")
if FMP_API_KEY is None:
    raise RuntimeError("環境変数 FMP_API_KEY が設定されていません")

BASE_URL = "https://financialmodelingprep.com/api/v3/economic_calendar"

# NFP など対象にしたいイベント名のキーワード
TARGET_KEYWORDS = [
    "nonfarm",           # Nonfarm Payrolls
    "non-farm",
    "payroll",
    "unemployment rate",
    "employment change",
]


def fetch_econ_calendar(start: str, end: str) -> list[dict]:
    params = {
        "from": start,
        "to": end,
        "apikey": FMP_API_KEY,
    }
    r = requests.get(BASE_URL, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected response format: {data}")
    return data


def is_target_event(ev: dict) -> bool:
    name = str(ev.get("event", "")).lower()
    return any(kw in name for kw in TARGET_KEYWORDS)


def parse_to_df(events: list[dict]) -> pd.DataFrame:
    rows = []
    for ev in events:
        if not is_target_event(ev):
            continue
        if ev.get("country") != "US":
            # NFP など米国指標に絞る場合
            continue

        date_raw = ev.get("date")
        dt = pd.to_datetime(date_raw, utc=True, errors="coerce")
        if dt is pd.NaT:
            continue

        # 数値に変換
        def to_float(x):
            try:
                return float(x) if x is not None else None
            except (TypeError, ValueError):
                return None

        actual = to_float(ev.get("actual"))
        estimate = to_float(ev.get("estimate"))
        previous = to_float(ev.get("previous"))
        change = to_float(ev.get("change"))

        if actual is None:
            continue

        # ---- サプライズ計算ロジック ----
        base_for_pct = None

        if estimate is not None:
            surprise = actual - estimate
            base_for_pct = estimate
        elif previous is not None:
            surprise = actual - previous
            base_for_pct = previous
        elif change is not None:
            surprise = change
            base_for_pct = actual - change  # 一応の近似
        else:
            # 何もなければスキップ
            continue

        surprise_pct = None
        if base_for_pct not in (None, 0):
            surprise_pct = surprise / abs(base_for_pct)

        rows.append(
            {
                "date": dt.date(),
                "event": ev.get("event"),
                "country": ev.get("country"),
                "actual": actual,
                "estimate": estimate,
                "previous": previous,
                "change": change,
                "surprise": surprise,
                "surprise_pct": surprise_pct,
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def main():
    start = "2010-01-01"
    end = datetime.utcnow().strftime("%Y-%m-%d")

    events = fetch_econ_calendar(start, end)
    print(f"raw events: {len(events)}")

    df = parse_to_df(events)
    print(df.tail())

    os.makedirs("data/fmp", exist_ok=True)
    out_path = "data/fmp/us_nfp_surprise.csv"
    df.to_csv(out_path, index=False)
    print("saved:", out_path)


if __name__ == "__main__":
    main()