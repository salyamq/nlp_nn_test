import pandas as pd
import numpy as np
import torch
from loguru import logger
from sklearn.preprocessing import StandardScaler

from torch.utils.data import DataLoader, Dataset
from torch import optim
import torch.nn as nn



df = pd.read_csv(r"../learning_datasets/simple_dataset.csv")
X = df.drop(columns=['salary']).values.astype(np.float32)
y = df['salary'].values.astype(np.float32)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

tensor_x = torch.from_numpy(X_scaled).float()
tensor_y = torch.from_numpy(y).unsqueeze(1).float()

# tensor_x.shape = torch.Size([1000, 6])

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


class SalaryDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]

dataset = SalaryDataset(tensor_x, tensor_y)
dataloader = DataLoader(dataset = dataset, batch_size = 100,
                        shuffle = True, drop_last = False, pin_memory=True)

# Simple mlp architecture

class SimpleMLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        return self.linear4(x)

model = SimpleMLP(input_size = tensor_x.shape[1]).to(device)

lr = 3e-4

loss_func = nn.MSELoss()
optimizer = optim.Adam(params = model.parameters(), lr = lr, weight_decay = 1e-4)
epochs = 500

for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        y_pred = model(x_batch)
        loss = loss_func(y_pred, y_batch)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(dataloader)
    logger.success(f"Epoch [{epoch+1:02d}/{epochs}] | Loss: {avg_loss ** 0.5:.4f}")











