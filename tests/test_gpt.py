import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

layer = nn.TransformerEncoderLayer(
    d_model=32,
    nhead=4,
    dim_feedforward=32,
    batch_first=True,
)
gpt = nn.Sequential(
    nn.Linear(2, 32),
    nn.ReLU(),
    nn.TransformerEncoder(layer, num_layers=3),
    nn.Linear(32, 1),
)

# 生成数据集
num_samples = 10000
X = torch.rand(num_samples, 1)  # 输入特征X
Y = torch.rand(num_samples, 1)  # 输入特征Y
Z = 10 * (X + X**2 + X**3 + Y + Y**2 + Y**3)  # 目标值
inputs = torch.cat((X, Y), dim=1)  # 合并特征 (1000, 2)

criterion = nn.MSELoss()
optimizer = optim.AdamW(gpt.parameters(), lr=0.0001)

# 创建DataLoader
dataset = TensorDataset(inputs, Z)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# 训练循环
for epoch in range(500):
    for batch_inputs, batch_Z in dataloader:
        optimizer.zero_grad()
        preds = gpt(batch_inputs)
        loss = criterion(preds, batch_Z)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

torch.save(gpt.state_dict(), "tmp.pt")
