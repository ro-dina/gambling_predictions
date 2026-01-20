# merge_fx_bluesky_employment.py
import os
import pandas as pd

FX_DIR = "data/fx_bluesky"
PAYEMS_PATH = "data/fred/PAYEMS.csv"
PAIRS = ["GBPJPY", "EURJPY", "USDJPY"]


def main():
    if not os.path.exists(PAYEMS_PATH):
        raise RuntimeError(f"{PAYEMS_PATH} が見つかりません。先に fetch_payems.py を実行してください。")

    emp = pd.read_csv(PAYEMS_PATH)
    emp["date"] = pd.to_datetime(emp["date"])
    emp = emp.sort_values("date")

    # 月次PAYEMSを日次に展開して ffll する
    # 全期間の日付インデックスを作成
    full_idx = pd.date_range(emp["date"].min(), emp["date"].max(), freq="D")
    emp_daily = (
        emp.set_index("date")
        .reindex(full_idx)
        .rename_axis("date")
        .sort_index()
    )
    emp_daily["PAYEMS"] = emp_daily["PAYEMS"].ffill()
    emp_daily = emp_daily.reset_index()

    print("PAYEMS daily example:")
    print(emp_daily.head())

    for pair in PAIRS:
        src_path = os.path.join(FX_DIR, f"{pair}_with_bluesky_cbank.csv")
        if not os.path.exists(src_path):
            print(f"[WARN] {src_path} が見つかりません。スキップします。")
            continue

        print(f"=== merging {pair} with PAYEMS ===")
        df = pd.read_csv(src_path)
        if "date" not in df.columns:
            raise RuntimeError(f"{src_path} に 'date' 列がありません。")

        df["date"] = pd.to_datetime(df["date"])

        merged = pd.merge(
            df.sort_values("date"),
            emp_daily,
            on="date",
            how="left",
        ).sort_values("date")

        # 念のため欠損PAYEMSは前方埋め・後方埋めしておく
        merged["PAYEMS"] = (
            merged["PAYEMS"]
            .astype(float)
            .ffill()
            .bfill()
        )

        out_path = os.path.join(FX_DIR, f"{pair}_with_bluesky_cbank_emp.csv")
        merged.to_csv(out_path, index=False)
        print(f" -> saved: {out_path}")


if __name__ == "__main__":
    main()