# =========================
# 1) セットアップ
# =========================
# pip3 install yfinance torch matplotlib pandas scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import yfinance as yf

# -------------------------
# 取得する銘柄を2つ指定（ANA, JAL）
# ANA = 9202.T, JAL = 9201.T
# -------------------------
TICKERS = ["9202.T", "9201.T"]  # [ANA, JAL]
PERIOD = "5y"
INTERVAL = "1d"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 2) Yahoo Finance から複数銘柄の終値を取得
# =========================

def fetch_series_multi(tickers, period=PERIOD, interval=INTERVAL) -> pd.DataFrame:
    """複数銘柄の終値を DataFrame で取得（列=各ティッカー）。
    日次に揃え、市場休場日は前方補間で埋める。
    """
    data = {}
    for tk in tickers:
        hist = yf.Ticker(tk).history(period=period, interval=interval)
        if hist.empty or "Close" not in hist.columns:
            raise ValueError(f"'{tk}' の価格データが取得できませんでした。")
        s = hist["Close"].dropna().asfreq("D").ffill()
        data[tk] = s
    df = pd.DataFrame(data).ffill()
    return df

# =========================
# 3) Dataset と モデル
# =========================
SEQ_LEN = 24  # 直近24点 → 次の1点（ANA）
ANA_IDX = 0   # 入力列のうち ANA の列インデックス（TICKERS[0]）

class MultiSeqDataset(Dataset):
    def __init__(self, array_2d: np.ndarray, target_idx: int = ANA_IDX, seq_len: int = SEQ_LEN):
        self.data = array_2d
        self.target_idx = target_idx
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        # x: (T, F=2) / y: ANA の次の1点（スカラー）
        x = self.data[idx:idx + self.seq_len, :]
        y = self.data[idx + self.seq_len, self.target_idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class MultiLSTMModel(nn.Module):
    def __init__(self, input_size: int = 2, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (B, T, F)
        out, _ = self.lstm(x)      # (B, T, H)
        out = out[:, -1, :]        # (B, H) 最後の時刻
        out = self.fc(out)         # (B, 1)
        return out.squeeze(-1)     # (B,)

# =========================
# 4) メイン処理：ANA を JAL とともに使って予測
# =========================
if __name__ == "__main__":
    assert len(TICKERS) == 2, "TICKERS には [ANA, JAL] の2銘柄を指定してください。"

    # ---- データ取得（列順 = [ANA, JAL]）
    df_close = fetch_series_multi(TICKERS)
    print("\n[HEAD] 終値 DataFrame:\n", df_close.head())

    # ---- 学習/テスト分割（時間順）
    values = df_close.values  # (N, 2)
    n = len(df_close)
    train_size = int(n * 0.8)
    train_vals = values[:train_size]
    test_vals  = values[train_size:]

    # ---- 標準化（2列同時に：trainでfit→testをtransform）
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_vals)
    test_scaled  = scaler.transform(test_vals)

    # ---- Dataset/DataLoader
    train_ds = MultiSeqDataset(train_scaled, target_idx=ANA_IDX, seq_len=SEQ_LEN)
    test_ds  = MultiSeqDataset(test_scaled,  target_idx=ANA_IDX, seq_len=SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    # ---- モデル/学習
    model = MultiLSTMModel(input_size=2, hidden_size=64, num_layers=1).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    EPOCHS = 30
    model.train()
    for epoch in range(EPOCHS):
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total += loss.item() * x.size(0)
        print(f"Epoch {epoch+1}/{EPOCHS}  loss={total/len(train_ds):.4f}")

    # ---- 1ステップ先 予測（ティーチャーフォース：真のJALも供給）
    model.eval()
    preds_scaled = []
    with torch.no_grad():
        for i in range(len(test_ds)):
            x, _ = test_ds[i]                 # (T, 2)
            x = x.unsqueeze(0).to(DEVICE)     # (1, T, 2)
            yhat = model(x).cpu().item()      # ANA の標準化スケール
            preds_scaled.append(yhat)

    # ---- ANA の逆標準化（列ごとのスケール/平均を使う）
    ana_scale = scaler.scale_[ANA_IDX]
    ana_mean  = scaler.mean_[ANA_IDX]
    preds_ana = np.array(preds_scaled) * ana_scale + ana_mean

    # 真値（ANA）と時系列インデックス（テスト先頭からSEQ_LENぶんシフト）
    true_ana = df_close.iloc[train_size:,  ANA_IDX].values[SEQ_LEN:]
    idx_ana  = df_close.index[train_size:][SEQ_LEN:]

    plt.figure(figsize=(12, 6))
    plt.plot(idx_ana, true_ana, label="True ANA")
    plt.plot(idx_ana, preds_ana, label="Pred ANA")
    plt.title("ANA prediction using ANA+JAL (1-step)")
    plt.xlabel("Date"); plt.ylabel("Close Price"); plt.legend(); plt.tight_layout(); plt.show()

    # ---- 再帰予測（将来 K ステップ）
    # 直近SEQ_LENの標準化済みウィンドウを使う。未来のJALは未知なので「持ち続け（persistence）」を採用。
    K_STEPS = 12
    all_scaled = scaler.transform(values)
    window = all_scaled[-SEQ_LEN:, :].copy()   # (T, 2)
    jal_last = window[-1, 1]                   # 直近JAL（標準化スケール）

    future_scaled = []
    with torch.no_grad():
        for _ in range(K_STEPS):
            xin = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1, T, 2)
            yhat_ana = model(xin).cpu().item()  # ANA(標準化)
            future_scaled.append(yhat_ana)
            # JAL は persistence（同じ値を使い続ける）
            new_row = np.array([yhat_ana, jal_last], dtype=np.float32)
            window = np.vstack([window[1:], new_row])

    future_ana = np.array(future_scaled) * ana_scale + ana_mean
    future_idx = pd.date_range(df_close.index[-1], periods=K_STEPS+1, freq="D")[1:]

    plt.figure(figsize=(12, 4))
    plt.plot(df_close.index[-100:], df_close.iloc[-100:, ANA_IDX], label="ANA history (last 100)")
    plt.plot(future_idx, future_ana, label=f"ANA recursive (+{K_STEPS})")
    plt.title("ANA recursive forecast (JAL=persistence)")
    plt.xlabel("Date"); plt.ylabel("Close Price"); plt.legend(); plt.tight_layout(); plt.show()

    # ---- CSV 保存→再読込→一致確認
    out_csv = "ANA_JAL_close.csv"
    df_close.to_csv(out_csv, index=True)
    df2 = pd.read_csv(out_csv, index_col=0, parse_dates=True)
    print(f"\n同じ内容か？({out_csv}):", df_close.equals(df2))