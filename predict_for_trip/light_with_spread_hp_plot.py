# light_with_spread_hp_plot.py
import os
import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader

# ========= Config =========
PAIRS = ["GBPJPY", "EURJPY", "USDJPY"]          # 優先度: GBP > EUR > USD
DATA_DIR = "data/fx_bluesky"
DATE_START = "2020-01-01"
DATE_END = "2025-12-01"

# 予測（評価・グラフ）をこの期間に統一したい場合
# NOTE: 学習用の過去データは DATE_START..DATE_END で読み込む（ここは狭めない方が安全）
PLOT_START = "2024-01-01"
PLOT_END = "2025-05-01"

# 方向評価のため、ほぼ0のリターンは除外する閾値
DA_EPS = 1e-6

# 簡易バックテストの取引コスト（片道）。例: 0.0001 = 1bp
TX_COST = 0.0001

# ポジションを±1にするか、予測リターンをそのまま使うか
PNL_MODE = "sign"  # "sign" or "raw"

SEQ_LEN = 30
TRAIN_WINDOW = 150
BATCH_SIZE = 32
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Hyperparam search ---        
HP_PAIRS = {"GBPJPY", "EURJPY", "USDJPY"}  # 探索対象（全部やるならこのまま）
HP_EVAL_STEPS = 120  # 探索は重いので、最初のNステップだけで評価（大きいほど正確だが重い）
HP_GRID = {
    "hidden": [64],
    "layers": [1],
    "epochs": [60],
}

# 乱数固定（実行のブレを減らす）
SEED = 42


# ========= Utils =========
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # no-op on cpu
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ========= Dataset =========
class FxDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# ========= Model =========
class LSTMModel(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128, layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out.squeeze(-1)


# ========= Sequence builder =========
def make_sequences(arrX: np.ndarray, arrY: np.ndarray, seq_len: int):
    Xs, Ys = [], []
    for i in range(len(arrY) - seq_len):
        Xs.append(arrX[i : i + seq_len])
        Ys.append(arrY[i + seq_len])
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
    return total / max(1, len(loader.dataset))


# ========= Feature engineering =========
def compute_spread(df: pd.DataFrame, pair: str) -> pd.Series:
    # pair別に「外国 - 日本」で統一
    if pair == "USDJPY":
        if "US_POLICY" in df.columns and "JP_POLICY" in df.columns:
            return pd.to_numeric(df["US_POLICY"], errors="coerce") - pd.to_numeric(df["JP_POLICY"], errors="coerce")
        return pd.Series(0.0, index=df.index)

    if pair == "EURJPY":
        if "EU_POLICY" in df.columns and "JP_POLICY" in df.columns:
            return pd.to_numeric(df["EU_POLICY"], errors="coerce") - pd.to_numeric(df["JP_POLICY"], errors="coerce")
        return pd.Series(0.0, index=df.index)

    if pair == "GBPJPY":
        if "UK_POLICY" in df.columns and "JP_POLICY" in df.columns:
            return pd.to_numeric(df["UK_POLICY"], errors="coerce") - pd.to_numeric(df["JP_POLICY"], errors="coerce")
        return pd.Series(0.0, index=df.index)

    return pd.Series(0.0, index=df.index)


def prepare_df(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= DATE_START) & (df["date"] <= DATE_END)].reset_index(drop=True)

    # 必須列の型を安全化
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # SNS列が無い/欠損でも動くように
    for c in ["sent_mean", "post_count"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # log return
    df["log_ret"] = np.log(df["value"]) - np.log(df["value"].shift(1))
    df["log_ret_next"] = df["log_ret"].shift(-1)

    # spread / spread_chg
    df["spread"] = compute_spread(df, pair).fillna(0.0)
    df["spread_chg"] = df["spread"].diff().fillna(0.0)

    # 必須欠損を落とす
    df = df.dropna(subset=["value", "log_ret", "log_ret_next"]).reset_index(drop=True)
    return df


# ========= Walk-forward core =========
def walk_forward_mae(
    df: pd.DataFrame,
    pair: str,
    hidden: int,
    layers: int,
    epochs: int,
    eval_steps: int | None = None,
    make_plot: bool = False,
    plot_path: str | None = None,
):
    """
    eval_steps: 探索用に、最初のNステップだけでMAEを評価（Noneなら全期間）
    make_plot: Trueなら True vs Pred を描画（全期間推奨）
    """
    feat_cols = ["log_ret", "sent_mean", "post_count", "spread_chg"]
    X_all = df[feat_cols].to_numpy(dtype=np.float32)
    y_all = df["log_ret_next"].to_numpy(dtype=np.float32)
    prices = df["value"].to_numpy(dtype=np.float32)
    dates = df["date"].to_numpy()

    n = len(df)
    if n <= TRAIN_WINDOW + SEQ_LEN + 2:
        return np.nan, None

    preds_price, truths_price, dates_pred = [], [], []
    preds_ret, truths_ret = [], []

    i = TRAIN_WINDOW
    steps_done = 0
    while i + SEQ_LEN < n - 1:
        # train slice
        X_train = X_all[:i]
        y_train = y_all[:i]

        # 標準化（walkごとにtrainで再計算）
        mu = X_train.mean(axis=0, keepdims=True)
        sd = X_train.std(axis=0, keepdims=True)
        sd = np.where(sd < 1e-8, 1.0, sd)

        X_train_n = (X_train - mu) / sd
        X_seq, y_seq = make_sequences(X_train_n, y_train, SEQ_LEN)

        if len(y_seq) <= 0:
            break

        train_ds = FxDataset(X_seq, y_seq)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

        model = LSTMModel(input_dim=X_all.shape[1], hidden=hidden, layers=layers).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        loss_fn = nn.MSELoss()

        for _ in range(epochs):
            train_one(model, train_loader, opt, loss_fn)

        # predict 1 step
        X_last = X_all[i - SEQ_LEN : i]
        X_last_n = (X_last - mu) / sd
        X_last_t = torch.tensor(X_last_n, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred_ret = model(X_last_t).item()

        # 真の次期リターン（log_ret_next）は df のi行に対応
        true_ret = float(y_all[i])
        preds_ret.append(float(pred_ret))
        truths_ret.append(true_ret)

        pred_price = float(prices[i]) * float(np.exp(pred_ret))
        true_price = float(prices[i + 1])

        preds_price.append(pred_price)
        truths_price.append(true_price)
        dates_pred.append(dates[i + 1])

        i += 1
        steps_done += 1
        if eval_steps is not None and steps_done >= eval_steps:
            break

    if len(preds_price) == 0:
        return np.nan, None

    preds_price = np.array(preds_price, dtype=np.float32)
    truths_price = np.array(truths_price, dtype=np.float32)
    preds_ret_np = np.array(preds_ret, dtype=np.float32)
    truths_ret_np = np.array(truths_ret, dtype=np.float32)

    mae = float(np.mean(np.abs(preds_price - truths_price)))

    plot_data = None
    if make_plot:
        # 予測・評価を指定期間に揃える
        d = pd.to_datetime(np.array(dates_pred))
        mask = (d >= pd.to_datetime(PLOT_START)) & (d <= pd.to_datetime(PLOT_END))

        d_plot = d[mask]
        t_plot = truths_price[mask]
        p_plot = preds_price[mask]
        pr_plot = preds_ret_np[mask]
        tr_plot = truths_ret_np[mask]

        if len(d_plot) == 0:
            print(
                f"[WARN] {pair}: plot window has 0 points. "
                f"Check PLOT_START/PLOT_END or DATE_START/DATE_END."
            )
            plot_data = (d, truths_price, preds_price)
        else:
            mae_window = float(np.mean(np.abs(p_plot - t_plot)))

            # ===== Direction metrics =====
            # 方向：pred_ret と true_ret の符号が一致した割合
            # ほぼ0のtrue_retは除外（ノイズ）
            valid = np.abs(tr_plot) > DA_EPS
            if valid.sum() == 0:
                da = float("nan")
                wda = float("nan")
            else:
                da = float(np.mean((np.sign(pr_plot[valid]) == np.sign(tr_plot[valid])).astype(np.float32)))
                # 重み付き：|true_ret| を重みにする
                w = np.abs(tr_plot[valid])
                wda = float(np.sum(w * (np.sign(pr_plot[valid]) == np.sign(tr_plot[valid])).astype(np.float32)) / np.sum(w))

            # ===== Simple PnL backtest =====
            # position: sign(pred) or raw(pred)
            if PNL_MODE == "raw":
                pos = pr_plot
            else:
                pos = np.sign(pr_plot)

            # 取引コスト：ポジション変化量に比例（片道コスト）
            pos_prev = np.concatenate([[0.0], pos[:-1]])
            turnover = np.abs(pos - pos_prev)
            cost = TX_COST * turnover

            pnl = pos * tr_plot - cost
            pnl_sum = float(np.sum(pnl))
            pnl_mean = float(np.mean(pnl))

            print(f"{pair}: MAE(price, window {PLOT_START}..{PLOT_END}) = {mae_window:.4f}")
            print(f"{pair}: DA(window)={da:.4f}  WDA(window)={wda:.4f}  PnL_sum={pnl_sum:.6f}  PnL_mean={pnl_mean:.6e}")

            plot_data = (d_plot, t_plot, p_plot)

            plt.figure(figsize=(10, 4))
            plt.plot(d_plot, t_plot, label="True price")
            plt.plot(d_plot, p_plot, label="Predicted price")
            plt.title(
                f"{pair} – True vs Predicted price ({PLOT_START}..{PLOT_END}) "
                f"(hidden={hidden}, layers={layers}, epochs={epochs})"
            )
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            if plot_path:
                plt.savefig(plot_path, dpi=150)
            plt.show()

    return mae, plot_data


def hyperparam_search(df: pd.DataFrame, pair: str):
    results = []
    best = {"mae": float("inf"), "hidden": None, "layers": None, "epochs": None}

    for hidden in HP_GRID["hidden"]:
        for layers in HP_GRID["layers"]:
            for epochs in HP_GRID["epochs"]:
                mae, _ = walk_forward_mae(
                    df, pair, hidden=hidden, layers=layers, epochs=epochs, eval_steps=HP_EVAL_STEPS
                )
                print(f"[HP] {pair} hidden={hidden} layers={layers} epochs={epochs} -> MAE={mae:.4f}")
                results.append({"pair": pair, "hidden": hidden, "layers": layers, "epochs": epochs, "mae": mae})

                if np.isfinite(mae) and mae < best["mae"]:
                    best = {"mae": mae, "hidden": hidden, "layers": layers, "epochs": epochs}

    print(f"[HP BEST] {pair} -> hidden={best['hidden']} layers={best['layers']} epochs={best['epochs']} MAE={best['mae']:.4f}")
    return best, pd.DataFrame(results)


def main():
    seed_everything(SEED)
    ensure_dir("plots")

    for pair in PAIRS:
        path = os.path.join(DATA_DIR, f"{pair}_with_bluesky_cbank.csv")
        if not os.path.exists(path):
            print(f"[SKIP] {pair}: {path} not found")
            continue

        df_raw = pd.read_csv(path)
        df = prepare_df(df_raw, pair)

        print(f"=== {pair} ===")

        # --- hyperparam search (optional) ---
        if pair in HP_PAIRS:
            best, df_res = hyperparam_search(df, pair)
            df_res.to_csv(f"plots/hp_results_{pair}.csv", index=False)

            # full run + plot
            plot_path = f"plots/{pair}_best.png"
            mae, _ = walk_forward_mae(
                df,
                pair,
                hidden=best["hidden"],
                layers=best["layers"],
                epochs=best["epochs"],
                eval_steps=None,
                make_plot=True,
                plot_path=plot_path,
            )
            print(f"{pair}: MAE(price) = {mae:.4f} (best)")
            print(f" -> saved plot: {plot_path}")

        else:
            # default run + plot
            plot_path = f"plots/{pair}_default.png"
            mae, _ = walk_forward_mae(
                df,
                pair,
                hidden=128,
                layers=2,
                epochs=100,
                eval_steps=None,
                make_plot=True,
                plot_path=plot_path,
            )
            print(f"{pair}: MAE(price) = {mae:.4f} (default)")
            print(f" -> saved plot: {plot_path}")


if __name__ == "__main__":
    main()