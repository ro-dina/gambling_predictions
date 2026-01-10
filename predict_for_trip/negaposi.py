import pandas as pd
import numpy as np
import torch
from transformers import pipeline

# さっき作った Parquet / CSV を読む
df = pd.read_parquet("data/bluesky_accounts/bluesky_accounts_20260109.parquet")
# or
# df = pd.read_csv("...csv")

# 変な NaN を消しておく
df["text"] = df["text"].fillna("")

device = 0 if torch.cuda.is_available() else -1
sent_cls = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=device,
)

def compute_sent_scores(texts, batch_size=32):
    scores = []
    n = len(texts)
    for i in range(0, n, batch_size):
        batch = texts[i:i+batch_size]
        # None などを防ぐ
        batch = [t if isinstance(t, str) else "" for t in batch]
        outputs = sent_cls(batch)
        for out in outputs:
            # POSITIVE → +score, NEGATIVE → -score にマッピング
            s = out["score"] if out["label"] == "POSITIVE" else -out["score"]
            scores.append(s)
        print(f"{i+len(batch)}/{n} done", end="\r")
    return np.array(scores, dtype=np.float32)

df["sent_score"] = compute_sent_scores(df["text"].tolist())

df.to_parquet("data/bluesky_accounts/bluesky_accounts_with_sent.parquet")
df.to_csv("data/bluesky_accounts/bluesky_accounts_with_sent.csv", index=False)