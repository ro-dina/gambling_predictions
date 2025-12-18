import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ==== 1. ダミーデータ時系列 ====
np.random.seed(0)
time = np.arange(0, 200, 0.1)
data = np.sin(time) + 0.1 * np.random.randn(len(time))  # sin波+ノイズ

df = pd.DataFrame({"time": time, "value": data})
df.set_index("time", inplace=True)

# ==== 2. 学習テスト分割 ====
train_size = int(len(df) * 0.8)
train_series = df.iloc[:train_size]["value"]
test_series = df.iloc[train_size:]["value"]

SEQ_LEN = 60 ##縺薙�螟画焚縺ｮ諢丞袖��

class TimeSeriesDataset(Dataset):
    def __init__(self, series, seq_len=20):
        self.series = torch.tensor(series.values, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.series) - self.seq_len

    def __getitem__(self, idx):
        x = self.series[idx:idx+self.seq_len].unsqueeze(-1)  # (seq_len,1)
        y = self.series[idx+self.seq_len]
        return x, y

train_dataset = TimeSeriesDataset(train_series, SEQ_LEN)
test_dataset = TimeSeriesDataset(test_series, SEQ_LEN)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# ==== 3. LSTM繝｢繝�Ν螳夂ｾｩ ====
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True ###縺薙％縺ｫ豕ｨ諢�
                            )
        self.fc = nn.Linear(hidden_size, 1)  ###縺薙％縺ｮ諢丞袖縺ｯ��

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]   # 譛蠕後�譎らせ縺�縺大�蜉帙☆繧�
        out = self.fc(out)
        return out.squeeze()

model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) ##Adam縺ｫ縺ｪ縺｣縺ｦ縺�ｋ縺ｮ縺ｫ豕ｨ諢�

# ==== 4. 蟄ｦ鄙� ====
EPOCHS = 50
for epoch in range(EPOCHS):
    for x, y in train_loader:
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")

# ==== 5. 繝�せ繝医ョ繝ｼ繧ｿ縺ｧ莠域ｸｬ ====
model.eval() #縺薙％繧よｳｨ諢�
preds = []
with torch.no_grad(): #縺薙％繧よｳｨ諢�
    for i in range(len(test_dataset)):
        x, _ = test_dataset[i]
        x = x.unsqueeze(0)  # (1, seq_len, 1)
        pred = model(x).item()
        preds.append(pred)

# ==== 6. 繧ｰ繝ｩ繝戊｡ｨ遉ｺ ====
plt.figure(figsize=(12,6))
plt.plot(test_series.index[SEQ_LEN:], test_series.values[SEQ_LEN:], label="True", color="blue")
plt.plot(test_series.index[SEQ_LEN:], preds, label="Predicted", color="red")
plt.title("LSTM Prediction vs True values")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.show()