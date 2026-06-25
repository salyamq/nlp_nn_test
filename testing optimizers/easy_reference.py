import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from torch.utils.data import DataLoader, Dataset
import torch.nn as nn


df = pd.read_csv(r"../learning_datasets/simple_dataset.csv")
X = df.drop(columns=['salary']).values.astype(np.float32)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
tensor_x = torch.from_numpy(X_scaled)

# tensor_x.shape = torch.Size([1000, 6])


class SalaryDataset(Dataset):
    def __init__(self, x_data):
        self.x_data = x_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, item):
        return self.x_data[item]

dataset = SalaryDataset(tensor_x)
dataloader = DataLoader(dataset = dataset, batch_size = 100,
                        shuffle = True, drop_last = False)

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
        return torch.sigmoid(self.linear4(x))

model = SimpleMLP(input_size = tensor_x.shape[1])

# ---



