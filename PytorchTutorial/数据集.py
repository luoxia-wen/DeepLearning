import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np


class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        data = pd.read_csv("diabetes.csv", dtype=np.float32)
        self.x_data = torch.from_numpy(data.iloc[:-32, :-1].values)
        self.y_data = torch.from_numpy(data.iloc[:-32, -1].values)
        self.length = len(self.x_data)

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]

    def __len__(self):
        return self.length


class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(8, 6)
        self.layer2 = torch.nn.Linear(6, 4)
        self.layer3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.layer1(x))
        x = self.sigmoid(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x


model = MyModel()
dataset = DiabetesDataset("diabetes.csv")
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)  # num_workers:并行读取的线程数

criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

test_xdata = pd.read_csv('diabetes.csv', dtype=np.float32).iloc[-32:, :-1].values
test_xdata = torch.from_numpy(test_xdata)
test_ydata = pd.read_csv('diabetes.csv', dtype=np.float32).iloc[-32:, -1].values
test_ydata = torch.from_numpy(test_ydata)

if __name__ == '__main__':
    for epoch in range(20):
        for i, data in enumerate(train_loader, 0):
            # 1.data
            inputs, labels = data
            # 2.Forward
            y_pred = model(inputs)
            loss = criterion(y_pred, labels.reshape(-1, 1))
            # 3.Bakward
            optimizer.zero_grad()
            loss.backward()
            # 4.update
            optimizer.step()
        print(f"{epoch + 1}: {loss.item()}")
    for i in range(len(test_xdata)):
        test_ypred = model(test_xdata[i])
        print(f"preds: {test_ypred}, true: {test_ydata[i]}")