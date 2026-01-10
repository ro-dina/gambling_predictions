import requests
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from onlypastdata import get_exchange_rate

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
    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, x: np.ndarray):
        self.min_ = float(np.min(x))
        self.max_ = float(np.max(x))
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        eps = 1e-12
        return (x - self.min_) / (self.max_ - self.min_ + eps)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        eps = 1e-12
        return x * (self.max_ - self.min_ + eps) + self.min_

class FXWindowDataset(Dataset):
    def __init__(self, series: np.ndarray, seq_len: int):
        """
        series: 1D array (scaled rate)
        """
        self.series = series.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.series) - self.seq_len

    def __getitem__(self, idx):
        x = self.series[idx : idx + self.seq_len]              # (seq_len,)
        y = self.series[idx + self.seq_len]                    # scalar
        x = torch.from_numpy(x).unsqueeze(-1)                  # (seq_len, 1)
        y = torch.tensor([y], dtype=torch.float32)             # (1,)
        return x, y

class MultiFeatureWindowDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray, seq_len: int):
        """features: 2D array (N, D), targets: 1D array (N,)
        学習用に (過去 seq_len ステップ → 翌日のターゲット) のペアを作る Dataset。
        """
        assert features.shape[0] == targets.shape[0], "features と targets の長さが一致していません"
        self.features = features.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_len, :]       # (seq_len, D)
        y = self.targets[idx + self.seq_len]                 # scalar
        x = torch.from_numpy(x)                              # (seq_len, D)
        y = torch.tensor([y], dtype=torch.float32)           # (1,)
        return x, y

class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
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

def load_fx_bluesky(base: str, target: str, date_start: str, date_end: str) -> pd.DataFrame:
    """FXとBlueskyを結合済みのCSVを読み込む。

    想定する列:
      - date: 日付
      - value: 為替レート
      - post_count: 投稿数
      - sent_mean: センチメント平均
      - sent_std: センチメント分散/標準偏差

    CSVパスは data/fx_bluesky/{base}{target}_with_bluesky.csv を想定。
    必要に応じてファイル名は調整してください。
    """
    pair = f"{base}{target}"
    csv_path = f"data/fx_bluesky/{pair}_with_bluesky.csv"

    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])

    start_ts = pd.to_datetime(date_start)
    end_ts = pd.to_datetime(date_end)
    df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].copy()
    df = df.sort_values("date").reset_index(drop=True)

    # 必要な列がなければ0で埋める
    if "post_count" not in df.columns:
        df["post_count"] = 0
    if "sent_mean" not in df.columns:
        df["sent_mean"] = 0.0
    if "sent_std" not in df.columns:
        df["sent_std"] = 0.0
    if "hike_count" not in df.columns:
        df["hike_count"] = 0
    if "cut_count" not in df.columns:
        df["cut_count"] = 0

    return df

# ===== 旧バージョン: 単純 hold-out 検証用 LSTM（参考用にコメントアウトで残しておく） =====
# def run_pair(base: str, target: str, date_start: str, date_end: str,
#              seq_len: int = 30, epochs: int = 30, lr: float = 1e-3,
#              hidden: int = 64, layers: int = 1):
#     """旧バージョン: 過去データの 80% を学習、20% をテストに使うシンプルなLSTM。
#     現在は run_pair_walk_forward を使っているので、この関数は呼び出していない。
#     """
#     pair = f"{base}{target}"
#     print(f"\n=== {pair} {date_start}..{date_end} (hold-out) ===")
#
#     df = get_exchange_rate(base, target, date_start, date_end)
#     if len(df) <= seq_len + 5:
#         raise RuntimeError(f"データが少なすぎます: len(df)={len(df)}。期間を広げるか seq_len を下げてください")
#
#     # return 学習
#     df["ret"] = df["value"].pct_change()
#     df = df.dropna().reset_index(drop=True)
#
#     rets = df["ret"].to_numpy(dtype=np.float64)
#     prices = df["value"].to_numpy(dtype=np.float64)
#
#     split = int(len(rets) * 0.8)
#     train_rets = rets[:split]
#     test_rets = rets[split:]
#     test_prices = prices[split:]
#
#     scaler = MinMaxScaler1D().fit(train_rets)
#     train_scaled = scaler.transform(train_rets)
#     test_scaled = scaler.transform(test_rets)
#
#     train_ds = FXWindowDataset(train_scaled, seq_len)
#     test_ds = FXWindowDataset(test_scaled, seq_len)
#     if len(train_ds) <= 0 or len(test_ds) <= 0:
#         raise RuntimeError(f"{pair}: データ不足 (train_len={len(train_rets)}, test_len={len(test_rets)}, seq_len={seq_len})")
#
#     train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
#
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = LSTMRegressor(hidden_size=hidden, num_layers=layers).to(device)
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#
#     for ep in range(1, epochs + 1):
#         loss = train_one(model, train_loader, optimizer, criterion, device)
#         if ep == 1 or ep % 5 == 0 or ep == epochs:
#             print(f"Epoch {ep:3d}/{epochs} | train_loss={loss:.6f}")
#
#     # テスト部分での予測
#     preds_scaled = predict_series(model, test_ds, device)
#     y_true_scaled = test_scaled[seq_len:]
#
#     preds_ret = scaler.inverse_transform(preds_scaled)
#     y_true_ret = scaler.inverse_transform(y_true_scaled)
#
#     base_price = float(test_prices[seq_len - 1])
#     true_prices = base_price * np.cumprod(1.0 + y_true_ret)
#     pred_prices = base_price * np.cumprod(1.0 + preds_ret)
#
#     test_dates = df["date"].iloc[split:].reset_index(drop=True)
#     pred_dates = test_dates.iloc[seq_len:].reset_index(drop=True)
#
#     mae = float(np.mean(np.abs(pred_prices - true_prices)))
#     print(f"{pair} hold-out MAE (price from return): {mae:.6f}")
#
#     plt.figure(figsize=(12, 5))
#     plt.plot(pred_dates, true_prices, label="True (hold-out)")
#     plt.plot(pred_dates, pred_prices, label="Pred (hold-out)")
#     plt.title(f"{pair} LSTM hold-out (return-learning, seq_len={seq_len})")
#     plt.xlabel("date")
#     plt.ylabel("rate")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()


def run_pair_walk_forward(base: str, target: str, date_start: str, date_end: str,
                          seq_len: int = 30, epochs_per_step: int = 5, lr: float = 1e-3,
                          hidden: int = 64, layers: int = 1, min_train_days: int = 200):
    """Walk-forward validation: 毎ステップ「過去だけ」で学習し直して翌日のreturnを予測する。

    - 学習対象: 前日比リターン (pct_change)
    - 評価: return MAE と、そこから再構成した価格パスの MAE
    """
    pair = f"{base}{target}"
    print(f"\n=== WALK-FORWARD {pair} {date_start}..{date_end} ===")

    df = load_fx_bluesky(base, target, date_start, date_end)
    if len(df) <= min_train_days + 5:
        raise RuntimeError(
            f"{pair}: データが少なすぎます (len(df)={len(df)}, min_train_days={min_train_days})。期間を広げるか min_train_days/seq_len を調整してください"
        )

    # return系列を作成
    #log returnに変える
    #df["ret"] = df["value"].pct_change()
    df["ret"] = np.log(df["value"]).diff()
    df = df.dropna().reset_index(drop=True)

    rets = df["ret"].to_numpy(dtype=np.float64)
    prices = df["value"].to_numpy(dtype=np.float64)
    dates = df["date"].to_numpy()

    # BlueSky由来の特徴は、まずは sent_mean のみを利用する
    sent_mean = df["sent_mean"].to_numpy(dtype=np.float64)

    # 特徴量行列: [log_return, sent_mean]
    # いったん次元数を減らし、為替の自己相関 + センチメント平均だけを見る
    features_raw = np.stack(
        [rets, sent_mean],
        axis=1,
    )  # (n, 2)

    n_features = features_raw.shape[1]

    n = len(rets)

    # 実際のデータ長に応じて有効な min_train_days を調整する
    # ・まずは「seq_len より十分大きいか」をチェック
    if n <= seq_len + 10:
        raise RuntimeError(
            f"{pair}: return系列が短すぎます (len(rets)={n}, seq_len={seq_len})。期間を広げるか seq_len を下げてください"
        )

    # 希望の min_train_days と、データ長から逆算した上限の小さい方を有効値とする
    # n - seq_len - 5 を上限として、少なくとも数ステップ分の検証区間を確保する
    effective_min_train_days = min(min_train_days, n - seq_len - 5)

    # それでもあまりに短い場合はエラー
    if effective_min_train_days < 30:
        raise RuntimeError(
            f"{pair}: 有効な学習期間が確保できません (len(rets)={n}, min_train_days={min_train_days}, seq_len={seq_len})。"
            "期間を広げるか、min_train_days/seq_len を下げてください"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    preds_ret_list = []
    true_ret_list = []
    date_list = []

    # effective_min_train_days から最後の手前まで1ステップずつ進める
    for end in range(effective_min_train_days, n):
        # rolling windowで直近N日のみの学習（直近約1年分を使用）
        train_window = 365
        start = max(0, end - train_window)
        train_rets = rets[start:end]
        train_features = features_raw[start:end]  # (L, D)

        # ターゲット(return)のスケーリング
        scaler = MinMaxScaler1D().fit(train_rets)
        train_ret_scaled = scaler.transform(train_rets)

        if len(train_ret_scaled) <= seq_len:
            continue

        # 特徴量の標準化（各特徴量ごとに平均0, 分散1程度にする）
        feat_mean = train_features.mean(axis=0, keepdims=True)
        feat_std = train_features.std(axis=0, keepdims=True)
        eps = 1e-8
        train_features_scaled = (train_features - feat_mean) / (feat_std + eps)

        train_ds = MultiFeatureWindowDataset(train_features_scaled, train_ret_scaled, seq_len)
        if len(train_ds) <= 0:
            continue

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)

        model = LSTMRegressor(input_size=n_features, hidden_size=hidden, num_layers=layers).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # 各ステップで少ないepochだけ学習（PC負荷を抑える）
        for ep in range(epochs_per_step):
            _ = train_one(model, train_loader, optimizer, criterion, device)

        # 直近seq_lenステップを使って「翌日」のreturnを1ステップ予測
        last_feat = features_raw[end - seq_len : end]
        last_feat_scaled = (last_feat - feat_mean) / (feat_std + eps)
        x = torch.from_numpy(last_feat_scaled.astype(np.float32)).unsqueeze(0).to(device)  # (1, T, D)
        with torch.no_grad():
            pred_scaled = model(x).cpu().numpy().reshape(-1)[0]

        pred_ret = float(scaler.inverse_transform(np.array([pred_scaled]))[0])
        true_ret = float(rets[end])

        preds_ret_list.append(pred_ret)
        true_ret_list.append(true_ret)
        date_list.append(dates[end])

    preds_ret = np.array(preds_ret_list, dtype=np.float64)
    true_ret = np.array(true_ret_list, dtype=np.float64)
    date_arr = pd.to_datetime(pd.Series(date_list))

    if len(preds_ret) == 0:
        raise RuntimeError(f"{pair}: walk-forwardで有効なステップが得られませんでした")

    # return空間でのMAE
    mae_ret = float(np.mean(np.abs(preds_ret - true_ret)))
    print(f"{pair} walk-forward MAE (return): {mae_ret:.6e}")

    # 価格パス再構成
    base_price = float(prices[effective_min_train_days - 1])
    #log return
    #true_prices = base_price * np.cumprod(1.0 + true_ret)
    #pred_prices = base_price * np.cumprod(1.0 + preds_ret)

    true_prices = base_price * np.exp(np.cumsum(true_ret))
    pred_prices = base_price * np.exp(np.cumsum(preds_ret))

    mae_price = float(np.mean(np.abs(pred_prices - true_prices)))
    print(f"{pair} walk-forward MAE (price):  {mae_price:.6f}")

    # グラフ描画
    plt.figure(figsize=(12, 5))
    plt.plot(date_arr, true_prices, label="True (WF)")
    plt.plot(date_arr, pred_prices, label="Pred (WF)")
    plt.title(f"{pair} LSTM walk-forward (return-learning, seq_len={seq_len})")
    plt.xlabel("date")
    plt.ylabel("rate")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # データ量を少し増やすため、2022年以降に拡張
    DATE_START = "2022-01-01"
    DATE_END = "2025-12-01"


    # ハイパーパラメータ（ここだけ触ればOK）
    SEQ_LEN = 30
    EPOCHS = 200
    LR = 1e-3

    for base in ["GBP", "EUR", "USD"]:
        run_pair_walk_forward(
            base=base,
            target="JPY",
            date_start=DATE_START,
            date_end=DATE_END,
            seq_len=SEQ_LEN,
            epochs_per_step=5,   # ステップごとのepoch。重いようなら減らす
            lr=LR,
            hidden=128,
            layers=2,
            # データ量とのバランスを考え、少なくとも約180営業日分を見てから予測を開始
            min_train_days=180,
        )