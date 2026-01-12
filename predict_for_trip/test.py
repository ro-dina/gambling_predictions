import pandas as pd

df = pd.read_csv("data/fx_bluesky/GBPJPY_with_bluesky.csv")
print("rows:", len(df))
print(df[["date", "value"]].head())
print(df[["date", "value"]].tail())
print("NaN in value:", df["value"].isna().sum())