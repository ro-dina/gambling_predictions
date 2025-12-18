import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

import matplotlib
#matplotlib.use("TkAgg")
matplotlib.use("macosx")


import yfinance as yf

ticked = yf.Ticker("^NDX")
hist = ticked.history(period="100d",interval="1d")

#print(hist) #ここをコメントアウトするとデータの詳細が見れる

closes=hist['Close'].values #詳細データから必要なデータをlistで取り出した

import numpy as np

# 3日分の終値から翌日の終値を予測するためのデータセットを作成
data = closes.astype("float32")
window_size = 3
X_list = []
y_list = []
for i in range(len(data) - window_size):
    X_list.append(data[i:i + window_size])
    y_list.append(data[i + window_size])

X_tensor = torch.tensor(X_list, dtype=torch.float32)
y_tensor = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)
from torch.utils.data import TensorDataset

dataset = TensorDataset(X_tensor, y_tensor)


batch_size = 16  # 小さいデータなのでバッチサイズも小さめに設定

# データローダーの作成（株価用）
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


"""
for x, y in test_dataloader:
    print("Shape of x [N, C, H, W]: ", x.shape)
    print("Shape of y: ", y.shape, y.dtype)

    break
"""

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1)  # 最終出力は1つ（終値の予測）
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x) 
        return logits
    
model = NeuralNetwork().to(device)
#model = NeuralNetwork()


#print(model) #どんなニューラルネットワークか確認

# 入力は「3日分の終値」なので形は [batch_size, 3]
print(model(torch.rand((1, 3)).to(device)))

loss_fn = nn.MSELoss()  # 終値予測なので回帰問題としてMSEを使用
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Adamの方が収束しやすい

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        #損失誤差の計算
        pred = model(X)
        loss = loss_fn(pred, y)

        #バックプロパゲーション
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

epochs = 50
for t in range(epochs):
    print(f"Epoch {t+1}\n------------------")
    train(train_dataloader, model, loss_fn, optimizer)
print("Training Done!")

# 学習後に直近3日分から翌日の終値を予測してみる
with torch.no_grad():
    last_window = torch.tensor(data[-3:], dtype=torch.float32).unsqueeze(0).to(device)
    pred_next = model(last_window)
    print(f"Last 3 closes: {data[-3:]}")
    print(f"Predicted next close: {pred_next.item():.2f}")