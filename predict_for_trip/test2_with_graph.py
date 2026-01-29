# light_with_spread.py
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt  # 追加

PAIRS = ["GBPJPY", "EURJPY", "USDJPY"]
DATA_DIR = "data/fx_bluesky"
DATE_START = "2020-01-01"
DATE_END = "2025-12-01"

SEQ_LEN = 30
TRAIN_WINDOW = 180
EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== Dataset =====
class FxDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ===== LSTM Model =====
class LSTMModel(nn.Module):
    def __init__(self, input_dim=4, hidden=128, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1])
        return out.squeeze(-1)

# ===== Walk-forward training =====
def make_sequences(arrX, arrY):
    Xs, Ys = [], []
    for i in range(len(arrY) - SEQ_LEN):
        Xs.append(arrX[i:i+SEQ_LEN])
        Ys.append(arrY[i+SEQ_LEN])
    return np.array(Xs), np.array(Ys)

def train_one(model, loader, opt, loss_fn):
    model.train()
    total = 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        total += loss.item() * len(y)
    return total / len(loader.dataset)

def walk_forward(df, pair, plot=True, save_dir="plots"):
    # 念のため日付順に並べ替え（CSVが保証していない場合に備える）
    df = df.sort_values("date").reset_index(drop=True)

    df = df[(df["date"] >= DATE_START) & (df["date"] <= DATE_END)].reset_index(drop=True)

    df["log_ret"] = np.log(df["value"]) - np.log(df["value"].shift(1))
    df["log_ret_next"] = df["log_ret"].shift(-1)

    # compute spread_chg
    if "US_POLICY" in df.columns:
        df["spread"] = df["US_POLICY"] - df["JP_POLICY"]
    elif "EU_POLICY" in df.columns:
        df["spread"] = df["EU_POLICY"] - df["JP_POLICY"]
    elif "UK_POLICY" in df.columns:
        df["spread"] = df["UK_POLICY"] - df["JP_POLICY"]
    else:
        df["spread"] = 0.0

    df["spread_chg"] = df["spread"] - df["spread"].shift(1)

    # keep only necessary columns（必要ならここに sent_mean/post_count/spread_chg の dropna も追加推奨）
    df = df.dropna(subset=["log_ret", "log_ret_next"]).reset_index(drop=True)

    X_all = df[["log_ret", "sent_mean", "post_count", "spread_chg"]].to_numpy(dtype=np.float32)
    y_all = df["log_ret_next"].to_numpy(dtype=np.float32)

    preds_price = []
    truths_price = []
    pred_dates = []  # 追加：予測対象日のdateを保存
    arr_prices = df["value"].to_numpy(dtype=np.float32)
    arr_dates = df["date"].to_numpy()

    i = TRAIN_WINDOW
    while i + SEQ_LEN < len(df):
        X_train = X_all[:i]
        y_train = y_all[:i]

        X_seq, y_seq = make_sequences(X_train, y_train)
        train_ds = FxDataset(X_seq, y_seq)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

        model = LSTMModel(input_dim=4).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        loss_fn = nn.MSELoss()

        for ep in range(EPOCHS):
            train_one(model, train_loader, opt, loss_fn)

        # one-step prediction
        X_last = X_all[i-SEQ_LEN:i]
        with torch.no_grad():
            pred_ret = model(torch.tensor(X_last, dtype=torch.float32).unsqueeze(0).to(DEVICE)).item()

        # convert return -> price
        pred_price = float(arr_prices[i]) * np.exp(pred_ret)
        true_price = float(arr_prices[i+1])  # 翌日価格が真値
        date_next = pd.to_datetime(arr_dates[i+1])  # 予測対象日は「翌日」

        preds_price.append(pred_price)
        truths_price.append(true_price)
        pred_dates.append(date_next)

        i += 1

    preds_price = np.array(preds_price, dtype=np.float64)
    truths_price = np.array(truths_price, dtype=np.float64)
    pred_dates = pd.to_datetime(pd.Series(pred_dates))

    mae = np.mean(np.abs(preds_price - truths_price))
    print(f"{pair}: MAE(price) = {mae:.4f}")

    # ===== 追加：グラフ描画 =====
    if plot and len(preds_price) > 0:
        # 1) 価格推移（予測 vs 真値）
        plt.figure(figsize=(12, 5))
        plt.plot(pred_dates, truths_price, label="True price")
        plt.plot(pred_dates, preds_price, label="Pred price")
        plt.title(f"{pair} Walk-forward 1-step Price Prediction (MAE={mae:.4f})")
        plt.xlabel("date")
        plt.ylabel("rate")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # 2) 絶対誤差の推移（レポート用に便利）
        abs_err = np.abs(preds_price - truths_price)
        plt.figure(figsize=(12, 3.5))
        plt.plot(pred_dates, abs_err)
        plt.title(f"{pair} Absolute Error |Pred-True|")
        plt.xlabel("date")
        plt.ylabel("abs error")
        plt.tight_layout()
        plt.show()

        # 3) 保存（任意）
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

            # 保存用にもう一度同じ図を作って保存（show()後だと環境により空になることがあるので作り直し）
            plt.figure(figsize=(12, 5))
            plt.plot(pred_dates, truths_price, label="True price")
            plt.plot(pred_dates, preds_price, label="Pred price")
            plt.title(f"{pair} Walk-forward 1-step Price Prediction (MAE={mae:.4f})")
            plt.xlabel("date")
            plt.ylabel("rate")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{pair}_price_pred.png"), dpi=200)
            plt.close()

            plt.figure(figsize=(12, 3.5))
            plt.plot(pred_dates, abs_err)
            plt.title(f"{pair} Absolute Error |Pred-True|")
            plt.xlabel("date")
            plt.ylabel("abs error")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{pair}_abs_error.png"), dpi=200)
            plt.close()

def main():
    for pair in PAIRS:
        path = os.path.join(DATA_DIR, f"{pair}_with_bluesky_cbank.csv")
        if not os.path.exists(path):
            print(f"[SKIP] {pair}")
            continue
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        print(f"=== {pair} ===")
        walk_forward(df, pair, plot=True, save_dir="plots")

if __name__ == "__main__":
    main()