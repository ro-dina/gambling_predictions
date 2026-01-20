# with_spread_employment_nfp.py
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader

PAIRS = ["GBPJPY", "EURJPY", "USDJPY"]
DATA_DIR = "data/fx_bluesky"

DATE_START = "2021-01-01"
DATE_END = "2025-12-01"

SEQ_LEN = 30
TRAIN_WINDOW = 150

# デフォルト学習設定（USDなど）
EPOCHS = 80
HIDDEN = 128
LAYERS = 2

BATCH_SIZE = 32
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== Hyperparam search (GBP/EURのみ厚めに探索) =====
DO_HP_SEARCH = True
HP_PAIRS = {"GBPJPY", "EURJPY"}

# まずは軽めのグリッド（必要なら増やせる）
HP_GRID = {
    "hidden": [64, 128, 256],
    "layers": [1, 2],
    "epochs": [60, 100, 140],
}

# 探索時は全期間のWFを回すと重いので、評価ステップ数を制限（例: 120日分）
HP_EVAL_STEPS = 120

# どちらも出力したいので保存先
PLOT_DIR = "plots"


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
    def __init__(self, input_dim: int, hidden: int, layers: int):
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
def walk_forward(
    df: pd.DataFrame,
    pair: str,
    hidden: int = HIDDEN,
    layers: int = LAYERS,
    epochs: int = EPOCHS,
    max_steps: int | None = None,
    do_plot: bool = True,
):
    df = df[(df["date"] >= DATE_START) & (df["date"] <= DATE_END)].reset_index(drop=True)

    # 念のため：SNS列が無い/欠損でも動くように 0 補完
    for c in ["sent_mean", "post_count"]:
        if c not in df.columns:
            df[c] = 0.0
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # 念のため：NFP_FLAG がなければ 0 列を追加
    if "NFP_FLAG" not in df.columns:
        df["NFP_FLAG"] = 0
    else:
        df["NFP_FLAG"] = pd.to_numeric(df["NFP_FLAG"], errors="coerce").fillna(0).astype(int)

    # 1. log return と次期リターン
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).reset_index(drop=True)
    df["log_ret"] = np.log(df["value"]) - np.log(df["value"].shift(1))
    df["log_ret_next"] = df["log_ret"].shift(-1)

    # 2. 金利スプレッド（外国 - 日本）
    if pair == "USDJPY":
        if "US_POLICY" in df.columns and "JP_POLICY" in df.columns:
            df["spread"] = pd.to_numeric(df["US_POLICY"], errors="coerce") - pd.to_numeric(df["JP_POLICY"], errors="coerce")
        else:
            df["spread"] = 0.0
    elif pair == "EURJPY":
        if "EU_POLICY" in df.columns and "JP_POLICY" in df.columns:
            df["spread"] = pd.to_numeric(df["EU_POLICY"], errors="coerce") - pd.to_numeric(df["JP_POLICY"], errors="coerce")
        else:
            df["spread"] = 0.0
    elif pair == "GBPJPY":
        if "UK_POLICY" in df.columns and "JP_POLICY" in df.columns:
            df["spread"] = pd.to_numeric(df["UK_POLICY"], errors="coerce") - pd.to_numeric(df["JP_POLICY"], errors="coerce")
        else:
            df["spread"] = 0.0
    else:
        df["spread"] = 0.0

    df["spread"] = pd.to_numeric(df["spread"], errors="coerce").fillna(0.0)
    df["spread_chg"] = df["spread"].diff().fillna(0.0)

    # 3. PAYEMS の変化量（将来ON/OFF用：今は作るだけ）
    if "PAYEMS" in df.columns:
        df["PAYEMS"] = pd.to_numeric(df["PAYEMS"], errors="coerce")
        df["PAYEMS_chg"] = df["PAYEMS"].diff().fillna(0.0)
    else:
        df["PAYEMS_chg"] = 0.0

    # 4. 欠損処理（log_ret系のみ必須）
    df = df.dropna(subset=["log_ret", "log_ret_next"]).reset_index(drop=True)

    # 特徴量行列
    feat_cols = ["log_ret", "sent_mean", "post_count", "spread_chg", "NFP_FLAG"]
    features = df[feat_cols].to_numpy(dtype=np.float32)
    rets_next = df["log_ret_next"].to_numpy(dtype=np.float32)
    prices = df["value"].to_numpy(dtype=np.float32)

    input_dim = features.shape[1]
    n = len(df)
    if n <= TRAIN_WINDOW + SEQ_LEN + 1:
        print(f"{pair}: データが少なすぎます (len={n})。パラメータを調整してください。")
        return np.nan, [], [], []

    preds_price: list[float] = []
    truths_price: list[float] = []
    dates_pred: list[pd.Timestamp] = []

    i = TRAIN_WINDOW
    steps = 0
    while i + SEQ_LEN < n - 1:
        if max_steps is not None and steps >= max_steps:
            break

        # 訓練部分
        train_feats = features[:i]
        train_y = rets_next[:i]

        # 特徴量標準化（各ウォークで再計算）
        mean = train_feats.mean(axis=0, keepdims=True)
        std = train_feats.std(axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)

        train_feats_norm = (train_feats - mean) / std
        X_seq, y_seq = make_sequences(train_feats_norm, train_y)

        if len(y_seq) == 0:
            # あり得る：TRAIN_WINDOW/SEQ_LENの関係で空になる
            break

        train_ds = FxDataset(X_seq, y_seq)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

        model = LSTMModel(input_dim=input_dim, hidden=hidden, layers=layers).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        loss_fn = nn.MSELoss()

        for _ in range(epochs):
            train_one(model, train_loader, opt, loss_fn)

        # 直近SEQ_LENを使って1ステップ先予測
        X_last = features[i - SEQ_LEN : i]
        X_last_norm = (X_last - mean) / std
        X_last_t = torch.tensor(X_last_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred_ret = float(model(X_last_t).item())

        # リターン → 価格へ変換
        price_t = float(prices[i])
        pred_price = price_t * np.exp(pred_ret)
        true_price = float(prices[i + 1])

        preds_price.append(pred_price)
        truths_price.append(true_price)
        dates_pred.append(df["date"].iloc[i + 1])

        i += 1
        steps += 1

    if len(preds_price) == 0:
        print(f"{pair}: 予測点が0件です（期間/窓/欠損を確認してください）")
        return np.nan, [], [], []

    preds_price_np = np.array(preds_price, dtype=float)
    truths_price_np = np.array(truths_price, dtype=float)
    mae_price = float(np.mean(np.abs(preds_price_np - truths_price_np)))

    # === グラフ描画・保存 ===
    if do_plot:
        os.makedirs(PLOT_DIR, exist_ok=True)
        plt.figure(figsize=(10, 4))
        plt.plot(dates_pred, truths_price_np, label="True price")
        plt.plot(dates_pred, preds_price_np, label="Predicted price")
        plt.title(f"{pair} – True vs Predicted price (hidden={hidden}, layers={layers}, epochs={epochs})")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        out_png = os.path.join(PLOT_DIR, f"{pair}_pred_hidden{hidden}_L{layers}_E{epochs}.png")
        plt.savefig(out_png, dpi=150)
        plt.show()

    return mae_price, dates_pred, truths_price, preds_price


def hyperparam_search(df: pd.DataFrame, pair: str):
    best = {"mae": float("inf"), "hidden": HIDDEN, "layers": LAYERS, "epochs": EPOCHS}

    for h in HP_GRID["hidden"]:
        for l in HP_GRID["layers"]:
            for e in HP_GRID["epochs"]:
                mae, _, _, _ = walk_forward(
                    df,
                    pair,
                    hidden=h,
                    layers=l,
                    epochs=e,
                    max_steps=HP_EVAL_STEPS,
                    do_plot=False,
                )
                if np.isnan(mae):
                    continue
                print(f"[HP] {pair} hidden={h} layers={l} epochs={e} -> MAE={mae:.4f}")
                if mae < best["mae"]:
                    best = {"mae": mae, "hidden": h, "layers": l, "epochs": e}

    print(f"[HP BEST] {pair} -> hidden={best['hidden']} layers={best['layers']} epochs={best['epochs']} MAE={best['mae']:.4f}")
    return best


def main():
    for pair in PAIRS:
        # NFPマージ済みを優先（無ければ従来CSVにフォールバック）
        path_nfp = os.path.join(DATA_DIR, f"{pair}_with_bluesky_cbank_emp_nfp.csv")
        path_base = os.path.join(DATA_DIR, f"{pair}_with_bluesky_cbank_emp.csv")
        path = path_nfp if os.path.exists(path_nfp) else path_base

        if not os.path.exists(path):
            print(f"[SKIP] {pair}: {path} not found")
            continue

        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])

        print(f"=== {pair} ===")

        # GBP/EUR は探索を厚めに
        if DO_HP_SEARCH and pair in HP_PAIRS:
            best = hyperparam_search(df, pair)
            mae, _, _, _ = walk_forward(
                df,
                pair,
                hidden=best["hidden"],
                layers=best["layers"],
                epochs=best["epochs"],
                max_steps=None,
                do_plot=True,
            )
            print(f"{pair}: MAE(price) = {mae:.4f}")
        else:
            mae, _, _, _ = walk_forward(
                df,
                pair,
                hidden=HIDDEN,
                layers=LAYERS,
                epochs=EPOCHS,
                max_steps=None,
                do_plot=True,
            )
            print(f"{pair}: MAE(price) = {mae:.4f}")


if __name__ == "__main__":
    main()