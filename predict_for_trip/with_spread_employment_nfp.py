# with_spread_employment.py
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# 優先度: GBP > EUR > USD
PAIRS = ["GBPJPY", "EURJPY", "USDJPY"]
DATA_DIR = "data/fx_bluesky"

DATE_START = "2021-01-01"
DATE_END = "2025-12-01"

SEQ_LEN = 30
TRAIN_WINDOW = 150
EPOCHS = 80
BATCH_SIZE = 32
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ===== Dataset =====
class FxDataset(Dataset): 
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ===== Model =====
class LSTMModel(nn.Module):
    def __init__(self, input_dim: int = 5, hidden: int = 128, layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze(-1)


# ===== Helpers =====
def make_sequences(arrX: np.ndarray, arrY: np.ndarray):
    Xs, Ys = [], []
    for i in range(len(arrY) - SEQ_LEN):
        Xs.append(arrX[i : i + SEQ_LEN])
        Ys.append(arrY[i + SEQ_LEN])
    return np.array(Xs), np.array(Ys)


def train_one(model, loader, opt, loss_fn):
    model.train()
    total = 0.0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        total += loss.item() * len(y)
    return total / len(loader.dataset)


# ===== Walk-forward =====
def walk_forward(df: pd.DataFrame, pair: str):
    df = df[(df["date"] >= DATE_START) & (df["date"] <= DATE_END)].reset_index(drop=True)

    # 念のため：特徴量に使う列が無い場合は 0 で補完
    for c in ["sent_mean", "post_count", "NFP_SHOCK_ACTIVE"]:
        if c not in df.columns:
            df[c] = 0.0
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # 1. log return と次期リターン
    df["log_ret"] = np.log(df["value"]) - np.log(df["value"].shift(1))
    df["log_ret_next"] = df["log_ret"].shift(-1)

    # 2. 金利スプレッド（外国 - 日本）
    if pair == "USDJPY":
        if "US_POLICY" in df.columns and "JP_POLICY" in df.columns:
            df["spread"] = df["US_POLICY"] - df["JP_POLICY"]
        else:
            df["spread"] = 0.0
    elif pair == "EURJPY":
        if "EU_POLICY" in df.columns and "JP_POLICY" in df.columns:
            df["spread"] = df["EU_POLICY"] - df["JP_POLICY"]
        else:
            df["spread"] = 0.0
    elif pair == "GBPJPY":
        if "UK_POLICY" in df.columns and "JP_POLICY" in df.columns:
            df["spread"] = df["UK_POLICY"] - df["JP_POLICY"]
        else:
            df["spread"] = 0.0
    else:
        df["spread"] = 0.0

    df["spread_chg"] = df["spread"].diff().fillna(0.0)

    # 3. PAYEMS の変化量
    if "PAYEMS" in df.columns:
        df["PAYEMS_chg"] = df["PAYEMS"].diff().fillna(0.0)
    else:
        df["PAYEMS_chg"] = 0.0

    # 4. 欠損処理（log_ret 系だけは必須）
    df = df.dropna(subset=["log_ret", "log_ret_next"]).reset_index(drop=True)

    # 特徴量行列
    features = df[["log_ret", "sent_mean", "post_count", "spread_chg", "NFP_SHOCK_ACTIVE"]].to_numpy(dtype=np.float32)
    rets_next = df["log_ret_next"].to_numpy(dtype=np.float32)
    prices = df["value"].to_numpy(dtype=np.float32)

    n = len(df)
    if n <= TRAIN_WINDOW + SEQ_LEN + 1:
        print(f"{pair}: データが少なすぎます (len={n})。パラメータを調整してください。")
        return

    preds_price = []
    truths_price = []

    i = TRAIN_WINDOW
    while i + SEQ_LEN < n - 1:
        # 訓練部分
        train_feats = features[:i]
        train_y = rets_next[:i]

        # 特徴量標準化（各ウォークで再計算）
        mean = train_feats.mean(axis=0, keepdims=True)
        std = train_feats.std(axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)

        train_feats_norm = (train_feats - mean) / std
        X_seq, y_seq = make_sequences(train_feats_norm, train_y)

        train_ds = FxDataset(X_seq, y_seq)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

        model = LSTMModel(input_dim=5).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        loss_fn = nn.MSELoss()

        for ep in range(EPOCHS):
            train_one(model, train_loader, opt, loss_fn)

        # 直近SEQ_LENを使って1ステップ先予測
        X_last = features[i - SEQ_LEN : i]
        X_last_norm = (X_last - mean) / std
        X_last_t = torch.tensor(X_last_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred_ret = model(X_last_t).item()

        # リターン → 価格へ変換
        price_t = float(prices[i])
        pred_price = price_t * np.exp(pred_ret)
        true_price = float(prices[i + 1])

        preds_price.append(pred_price)
        truths_price.append(true_price)

        i += 1

    preds_price = np.array(preds_price)
    truths_price = np.array(truths_price)

    mae_price = np.mean(np.abs(preds_price - truths_price))
    print(f"{pair}: MAE(price) = {mae_price:.4f}")


def main():
    for pair in PAIRS:
        # NFP特徴量を作った後のファイルを優先（無ければ従来ファイルにフォールバック）
        path_nfp = os.path.join(DATA_DIR, f"{pair}_with_bluesky_cbank_emp_nfp.csv")
        path_base = os.path.join(DATA_DIR, f"{pair}_with_bluesky_cbank_emp.csv")
        if os.path.exists(path_nfp):
            path = path_nfp
        else:
            path = path_base

        if not os.path.exists(path):
            print(f"[SKIP] {pair}: {path} not found")
            continue

        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        print(f"=== {pair} ===")
        walk_forward(df, pair)


if __name__ == "__main__":
    main()