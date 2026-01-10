import pandas as pd
import numpy as np

df = pd.read_csv("data/fx_bluesky/EURJPY_with_bluesky.csv")
df["date"] = pd.to_datetime(df["date"])

# log return（ターゲット）
df["log_ret"] = np.log(df["value"]).diff()

# 特徴量を1日ずらす（今日の特徴 → 明日のリターン）
df["log_ret_next"] = df["log_ret"].shift(-1)

cols = ["post_count", "sent_mean", "sent_std", "hike_count", "cut_count"]
corr = df[["log_ret_next"] + cols].corr()["log_ret_next"]
print(corr)