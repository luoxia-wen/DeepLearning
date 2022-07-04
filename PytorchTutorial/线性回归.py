import torch


class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


model = LinearModel()
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


x_data = torch.tensor([[1.], [2.], [3.]])
y_data = torch.tensor([[2.], [4.], [6.]])

for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"w = {model.linear.weight.data}")
print(f"b = {model.linear.bias.data}")

x_test = torch.tensor([[4.0]])
y_test = model(x_test)
print(f"y_pred = {y_test.data}")