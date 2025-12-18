# =========================
# 1) 繝ｩ繧､繝悶Λ繝ｪ
# =========================
# pip3 install pytrends torch matplotlib pandas scikit-learn

from pytrends.request import TrendReq
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler



# =========================
# 2) Google繝医Ξ繝ｳ繝峨°繧峨ョ繝ｼ繧ｿ蜿門ｾ�
# =========================
#KEYWORD = "Bitcoin"          # 竊仙･ｽ縺阪↑蜊倩ｪ槭↓螟画峩�域律譛ｬ隱朧K��
KEYWORD = "ノルウェー"          # 竊仙･ｽ縺阪↑蜊倩ｪ槭↓螟画峩�域律譛ｬ隱朧K��

GEO = "JP"                   # 蝨ｰ蝓滂ｼ�"JP","US",""=蜈ｨ荳也阜 縺ｪ縺ｩ��
TIMEFRAME = "today 12-m"      # 譛滄俣�井ｾ�: "today 12-m", "today 5-y", "2019-01-01 2025-10-01"��

pytrends = TrendReq(hl='ja-JP', tz=540)  # 譌･譛ｬ譎る俣
#pytrends.build_payload([KEYWORD], timeframe=TIMEFRAME, geo=GEO)
#df_trend = pytrends.interest_over_time()

#df_trend.to_csv('trend.csv')

##### google繝医Ξ繝ｳ繝峨′蜿榊ｿ懊↑縺��ｴ蜷医�縺薙ｌ縺ｫ蛻�ｊ譖ｿ縺医ｋ
df_trend=pd.read_csv('trend.csv', index_col=0)
df_trend.index=pd.to_datetime(df_trend.index)

# 蜿門ｾ礼ｵ先棡縺ｮ謨ｴ蠖｢
if df_trend.empty:
    raise ValueError("Googleトレンドからデータが取得できませんでした。キーワード/期間/地域を見直してください。")

# isPartial蛻励ｒ蜑企勁縺励※縲∵ｬ�謳阪ｒ蜑肴婿蝓九ａ
#series = df_trend[KEYWORD].copy()
#series = series.asfreq(series.index.inferred_freq) if series.index.inferred_freq else series
#print(series)


##### google繝医Ξ繝ｳ繝峨〒縺ｯ縺ｪ縺乗�ｪ萓｡繧ょ酔讒倥↓蜃ｦ逅�〒縺阪ｋ
import yfinance as yf
ticked = yf.Ticker("^NDX")
hist = ticked.history(period="100d",interval="1d")
hist = ticked.history(period="max",interval="1d")
series=hist['Close']
print(series)


series = series.ffill()
series.name = "value"


# =========================
# 3) 蜑榊�逅�ｼ域ｨ呎ｺ門喧��ｭｦ鄙�/繝�せ繝亥�蜑ｲ��
# =========================
# 蟄ｦ鄙�80% / 繝�せ繝�20%
n = len(series)
train_size = int(n * 0.8)
train_idx_end = train_size

train_series = series.iloc[:train_idx_end]
test_series  = series.iloc[train_idx_end:]

#print(train_series.shape)
#print(test_series.shape)


# 標準化、適切な値に
scaler = StandardScaler()
train_vals = scaler.fit_transform(train_series.values.reshape(-1, 1)).flatten()
test_vals  = scaler.transform(test_series.values.reshape(-1, 1)).flatten()

# =========================
# 4) PyTorch Dataset
# =========================
SEQ_LEN = 24  # 1繧ｹ繝�ャ繝嶺ｺ域ｸｬ逕ｨ縺ｮ螻･豁ｴ髟ｷ�磯ｱ谺｡縺ｪ繧円蜊雁ｹｴ縲∵律谺｡縺ｪ繧臥ｴ�1繝ｶ譛茨ｼ�

#### 莉･蜑阪↓菴ｿ縺｣縺鬱imeSeriesDataset(Dataset)縺ｨ縺ｮ驕輔＞縺ｫ豕ｨ逶ｮ(繧�▲縺ｦ縺�ｋ縺薙→縺ｯ縺ｻ縺ｼ蜷後§縺�縺代←縲√ｂ縺怜ｾ後〒諡｡蠑ｵ縺励ｈ縺�→縺吶ｋ縺ｨ)
class SeqDataset(Dataset):
    def __init__(self, arr, seq_len=24):
        self.x = []
        self.y = []
        for i in range(len(arr) - seq_len):
            self.x.append(arr[i:i+seq_len])
            self.y.append(arr[i+seq_len])
        self.x = torch.tensor(np.array(self.x), dtype=torch.float32).unsqueeze(-1) # (N, seq_len, 1)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)               # (N,)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

train_ds = SeqDataset(train_vals, SEQ_LEN)
test_ds  = SeqDataset(test_vals,  SEQ_LEN)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

##縺薙％縺ｯ莉･蜑阪→蜷後§
class LSTMModel(nn.Module): 
    def __init__(self, input_size=1, hidden_size=100, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True ###縺薙％縺ｫ豕ｨ諢�
                            )
        self.fc = nn.Linear(hidden_size, 1)  ###縺薙％縺ｮ諢丞袖縺ｯ��
    def forward(self, x):
        out, _ = self.lstm(x)  # (B, T, H)
        out = out[:, -1, :]   # 譛蠕後�譎らせ縺�縺大�蜉帙☆繧�
        out = self.fc(out)   # (B, 1)
        return out.squeeze() # (B,)

    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(hidden_size=64, num_layers=1).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) ##Adam縺ｫ縺ｪ縺｣縺ｦ縺�ｋ縺ｮ縺ｫ豕ｨ諢�

# =========================
# 6) 蟄ｦ鄙�
# =========================
EPOCHS = 50
model.train()
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * x.size(0)

    print(f"Epoch {epoch+1}/{EPOCHS}  loss={epoch_loss/len(train_ds):.4f}")

# =========================
# 7) 繝�せ繝医ョ繝ｼ繧ｿ縺ｧ 1繧ｹ繝�ャ繝嶺ｺ域ｸｬ
# =========================
model.eval()
pred_scaled = []
with torch.no_grad():
    for i in range(len(test_ds)):
        x, _ = test_ds[i]
        x = x.unsqueeze(0).to(device)  # (1, T, 1)
        yhat = model(x).cpu().item()
        pred_scaled.append(yhat)

# 騾�ｨ呎ｺ門喧�医せ繧ｫ繝ｩ繝ｼ縺ｫ蟇ｾ縺励※ inverse_transform 繧剃ｽｿ縺医ｋ繧医≧縺ｫ2谺｡蜈�喧��
pred_scaled_arr = np.array(pred_scaled).reshape(-1, 1)
pred_inv = scaler.inverse_transform(pred_scaled_arr).flatten()


# 逵溷､�医ユ繧ｹ繝磯Κ蛻��縺�■縲ヾEQ_LEN 莉･髯阪′莠域ｸｬ蟇ｾ雎｡��
true_inv = test_series.values[SEQ_LEN:]

# 蟇ｾ蠢懊☆繧九う繝ｳ繝�ャ繧ｯ繧ｹ
plot_index = test_series.index[SEQ_LEN:]

# =========================
# 8) 繝励Ο繝�ヨ�育悄蛟､ vs 莠域ｸｬ��
# =========================
plt.figure(figsize=(12, 6))
plt.plot(plot_index, true_inv, label="True")
plt.plot(plot_index, pred_inv, label="Predicted")
plt.title(f"Google Trends: {KEYWORD}  - LSTM Prediction vs True")
plt.xlabel("Date")
plt.ylabel("Trend index (0-100)")
plt.legend()
plt.tight_layout()
plt.show()

# =========================
# 9)�医♀縺ｾ縺托ｼ画焚繧ｹ繝�ャ繝怜�縺ｮ蜀榊ｸｰ莠域ｸｬ�医ユ繧ｹ繝域忰蟆ｾ縺九ｉ蜈医∈��
# =========================
# 逶ｴ霑代�螳溘ョ繝ｼ繧ｿ�亥ｭｦ鄙�+繝�せ繝医�譛蠕後�SEQ_LEN轤ｹ�峨ｒ菴ｿ縺｣縺ｦ縲〔繧ｹ繝�ャ繝怜�縺ｾ縺ｧ蜀榊ｸｰ逧�↓莠域ｸｬ
K_STEPS = 12
hist_full = scaler.transform(series.values.reshape(-1,1)).flatten()
window = hist_full[-SEQ_LEN:].copy()

future_scaled = []
model.eval()
with torch.no_grad():
    for _ in range(K_STEPS):
        xin = torch.tensor(window, dtype=torch.float32).view(1, SEQ_LEN, 1).to(device)
        yhat = model(xin).cpu().item()
        future_scaled.append(yhat)
        # 遯薙ｒ1縺､騾ｲ繧√ｋ
        window = np.concatenate([window[1:], [yhat]])

future = scaler.inverse_transform(np.array(future_scaled).reshape(-1,1)).flatten()
future_index = pd.date_range(series.index[-1], periods=K_STEPS+1, freq=series.index.inferred_freq)[1:]

plt.figure(figsize=(12, 4))
plt.plot(series.index[-100:], series.values[-100:], label="History (last 100)")
plt.plot(future_index, future, label=f"Recursive forecast (+{K_STEPS})")
plt.title(f"Google Trends: {KEYWORD} - Recursive Forecast")
plt.xlabel("Date")
plt.ylabel("Trend index (0-100)")
plt.legend()
plt.tight_layout()
plt.show()

df_trend.to_csv("Ntrend.csv", index=True)
# === ③ 再度読み込み ===
# 2) 読み込み（index_col=0 で先頭列をindexに、parse_dates=Trueで日時型に戻す）
df2 = pd.read_csv("trend.csv", index_col=0, parse_dates=True)

# === ④ 同一か確認 ===
print("\n同じ内容か？:", df_trend.equals(df2))