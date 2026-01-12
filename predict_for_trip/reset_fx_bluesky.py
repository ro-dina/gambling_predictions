# predict_for_trip/reset_fx_bluesky.py

import os
import pandas as pd

BASE_DIR = "data/fx_bluesky"
PAIRS = ["GBPJPY", "EURJPY", "USDJPY"]

def main():
    os.makedirs(os.path.join(BASE_DIR, "backup"), exist_ok=True)

    for pair in PAIRS:
        src = os.path.join(BASE_DIR, f"{pair}_with_bluesky.csv")
        if not os.path.exists(src):
            print(f"[WARN] {src} が見つかりません。スキップします。")
            continue

        print(f"=== processing {pair} ===")
        df = pd.read_csv(src)

        # 1) 今のファイルをバックアップ（中央銀行付きのフル版）
        backup_path = os.path.join(BASE_DIR, "backup", f"{pair}_with_bluesky_full_backup.csv")
        df.to_csv(backup_path, index=False)
        print(f"  -> backup saved: {backup_path}")

        # 2) 「結合前（FX + BlueSkyだけ）」に相当する列だけ残す
        base_cols = [c for c in df.columns if c in [
            "date",
            "value",
            "post_count",
            "sent_mean",
            "sent_std",
            "hike_count",
            "cut_count",
        ]]
        base_df = df[base_cols]

        # 3) それを _with_bluesky.csv として上書き（＝元の状態に戻すイメージ）
        base_path = os.path.join(BASE_DIR, f"{pair}_with_bluesky.csv")
        base_df.to_csv(base_path, index=False)
        print(f"  -> base (FX+BlueSky) saved: {base_path}")

        # 4) 中央銀行付きのフル版も別名で残しておく
        cbank_path = os.path.join(BASE_DIR, f"{pair}_with_bluesky_cbank.csv")
        df.to_csv(cbank_path, index=False)
        print(f"  -> full (FX+BlueSky+CBank) saved: {cbank_path}")

if __name__ == "__main__":
    main()