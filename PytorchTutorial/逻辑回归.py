import torch
import torch.nn as nn


class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.layer(x))
        return y_pred


model = LogisticRegressionModel()
criterion = nn.BCELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])

for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"w = {model.layer.weight.data}")
print(f"b = {model.layer.bias.data}")

x_test = torch.Tensor([[4.]])
y_test = model(x_test)
print(f"y_pred = {y_test}")