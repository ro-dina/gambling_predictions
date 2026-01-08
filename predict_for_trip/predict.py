import requests
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

# レートをとってくる（Frankfurter）
def get_exchange_rate(base: str, target: str, date_start: str, date_end: str) -> pd.DataFrame:
    url = f"https://api.frankfurter.app/{date_start}..{date_end}?from={base}&to={target}"
    response = requests.get(url, timeout=30)

    if response.status_code != 200:
        raise Exception(f"APIエラー: {response.status_code} {response.text}")

    data = response.json()
    rates = data.get("rates", {})

    df = pd.DataFrame([
        {"date": date, "value": values.get(target)}
        for date, values in rates.items()
    ])

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=["value"]).reset_index(drop=True)
    return df

class MinMaxScaler1D:
    def fit(self, x: np.ndarray):
        self.min = x.min()
        self.max = x.max()
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.min) / (self.max - self.min)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * (self.max - self.min) + self.min

class FXWindowDataset(Dataset):
    def __init__(self, values: np.ndarray, seq_len: int):
        self.values = values
        self.seq_len = seq_len

    def __len__(self):
        return len(self.values) - self.seq_len

    def __getitem__(self, idx):
        x = self.values[idx:idx+self.seq_len]
        y = self.values[idx+self.seq_len]
        x = torch.from_numpy(x).unsqueeze(-1)                  # (seq_len, 1)
        y = torch.tensor([y], dtype=torch.float32)             # (1,)
        return x, y

class LSTMRegressor(nn.Module):
    def __init__(self, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (B, T, 1)
        out, _ = self.lstm(x)
        last = out[:, -1, :]   # (B, hidden)
        y = self.fc(last)      # (B, 1)
        return y


def train_one(model: nn.Module, loader: DataLoader, optimizer, criterion, device: str) -> float:
    model.train()
    losses = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


def predict_series(model: nn.Module, ds: Dataset, device: str) -> np.ndarray:
    """Predict y for each window in ds; returns shape (len(ds),)."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(len(ds)):
            x, _ = ds[i]
            x = x.unsqueeze(0).to(device)  # (1, T, 1)
            pred = model(x).cpu().numpy().reshape(-1)[0]
            preds.append(pred)
    return np.array(preds, dtype=np.float32)


def run_pair(base: str, target: str, date_start: str, date_end: str,
             seq_len: int = 30, epochs: int = 30, lr: float = 1e-3,
             hidden: int = 64, layers: int = 1):
    pair = f"{base}{target}"
    print(f"\n=== {pair} {date_start}..{date_end} ===")

    df = get_exchange_rate(base, target, date_start, date_end)
    if len(df) <= seq_len + 5:
        raise RuntimeError(f"データが少なすぎます: len(df)={len(df)}。期間を広げるか seq_len を下げてください")

    values = df["value"].to_numpy(dtype=np.float64)

    # 時系列 split（shuffle禁止）
    split = int(len(values) * 0.8)
    train_values = values[:split]
    test_values = values[split:]

    # scalerは train のみに fit（リーク防止）
    scaler = MinMaxScaler1D().fit(train_values)
    train_scaled = scaler.transform(train_values)
    test_scaled = scaler.transform(test_values)

    train_ds = FXWindowDataset(train_scaled, seq_len)
    test_ds = FXWindowDataset(test_scaled, seq_len)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMRegressor(hidden_size=hidden, num_layers=layers).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(1, epochs + 1):
        loss = train_one(model, train_loader, optimizer, criterion, device)
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            print(f"Epoch {ep:3d}/{epochs} | train_loss={loss:.6f}")

    # 予測（test windows）
    preds_scaled = predict_series(model, test_ds, device)

    # testの真値（windowの次の1点）を作る
    # test_scaled: [t0, t1, ..., tN]
    # window seq_len -> label at index seq_len
    y_true_scaled = test_scaled[seq_len:]

    # 逆正規化（rateに戻す）
    preds = scaler.inverse_transform(preds_scaled)
    y_true = scaler.inverse_transform(y_true_scaled)

    # 予測に対応する日付（testの seq_len 以降）
    test_dates = df["date"].iloc[split:].reset_index(drop=True)
    pred_dates = test_dates.iloc[seq_len:].reset_index(drop=True)

    # 簡単な評価（MAE）
    mae = float(np.mean(np.abs(preds - y_true)))
    print(f"{pair} MAE: {mae:.6f}")

    # グラフ
    plt.figure(figsize=(12, 5))
    plt.plot(pred_dates, y_true, label="True")
    plt.plot(pred_dates, preds, label="Pred")
    plt.title(f"{pair} LSTM (seq_len={seq_len})")
    plt.xlabel("date")
    plt.ylabel("rate")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 期間はまず広めがおすすめ（短すぎるとLSTMが学習できません）
    DATE_START = "2024-01-01"
    DATE_END = "2025-12-01"

    # ハイパーパラメータ（ここだけ触ればOK）
    SEQ_LEN = 30
    EPOCHS = 30
    LR = 1e-3

    for base in ["GBP", "EUR", "USD"]:
        run_pair(base=base, target="JPY", date_start=DATE_START, date_end=DATE_END,
                 seq_len=SEQ_LEN, epochs=EPOCHS, lr=LR, hidden=64, layers=1)