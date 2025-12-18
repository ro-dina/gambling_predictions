import torch
import torch.nn as nn

# ---- 学習時と同じ LSTM モデル定義（必要に応じて hidden_size/num_layers を合わせる）----
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)      # [B, T, H]
        out = out[:, -1, :]        # 最後の時刻だけ
        out = self.fc(out)         # [B, 1]
        return out.squeeze(-1)     # [B]

# ---- 保存済みファイルを読み込み ----
ckpt_path = "prac/2回目/data/lstm_googletrend.pth"
checkpoint = torch.load(ckpt_path, map_location="cpu")

# ---- モデルを作って state_dict をロード（生dict/チェックポイントdictの両対応）----
model = LSTMModel()
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    # そのまま state_dict が保存されているケース
    model.load_state_dict(checkpoint)

# ---- 確認出力 ----
print("=== Model state_dict keys ===")
for k, v in model.state_dict().items():
    print(f"{k}: {tuple(v.shape)}")

print("\n=== Raw checkpoint type/keys ===")
print(type(checkpoint))
if isinstance(checkpoint, dict):
    print(list(checkpoint.keys()))