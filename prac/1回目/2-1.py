import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

import matplotlib
#matplotlib.use("TkAgg")
matplotlib.use("macosx")


# 訓練データをdatasetsからダウンロードa
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# テストデータをdatasetsからダウンロード
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)


#データ確認
"""
print(len(training_data)) #60,000枚の訓練画像
print(len(test_data)) #10,000枚のテスト画像
print(training_data[0][0].shape) #チャネル数1、サイズ28×28 のグレースケール画像

"""

training_data[0][0][0][14][14] #1枚目の画像の（14,14）ピクセル値（0〜1のTensor）微分の拡張変数
test_data[10] # (画像, ラベル) のタプル
test_data[0][1] # 0番画像のラベル値

"""
plt.imshow(training_data[30000][0][0],cmap="gray") #30,000番目の画像をグレースケールで表示
print(training_data[30000][1])
plt.show()
"""

batch_size = 64 #後の学習用に 1回の学習で64枚ずつ取り出す設定。


#データローダーの作成
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


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
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 256), #入力を移動したり何倍かして怪談関数を調整？
            nn.ReLU(),
            #nn.Linear(512, 512),
            #nn.ReLu(),
            nn.Linear(256, 10), #512個の線形変換を最終的にまとめて、10この出力にする。なぜならデータの服の種類が10だから
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x) 
        return logits
    
model = NeuralNetwork().to(device)
#model = NeuralNetwork()


#print(model) #どんなニューラルネットワークか確認

torch.rand((3, 4))
print(model(torch.rand((1,28,28)).to(device)))

loss_fn = nn.CrossEntropyLoss() #どんな誤差を小さくしたいか定義する
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3) #どんな勾配法を実行するか選択する

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        #損失誤差の計算
        pred = model(X)
        loss = loss_fn(pred, y)

        #バックプロぱゲーション
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test(dataloader, moel):
    size = len(dataloader.dataset)
    model.eval() #微分無視
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= size
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

#epochs = 1
epochs = 50
for t in range(epochs):
    print(f"Epoch {t+1}\n------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model)
print("Done!")

x = model(torch.tensor(training_data[30000][0]).to(device))
print(x)

plt.imshow(training_data[30000][0][0],cmap="gray")
plt.show()