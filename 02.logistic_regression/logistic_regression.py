import torch

x_data = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_data = torch.tensor([[0.0], [0.0], [1.0], [1.0]])

class Model(torch.nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.linear = torch.nn.Linear(1, 1)

  def forward(self, x):
    y_pred = torch.sigmoid(self.linear(x))
    return y_pred

# our model
model = Model()

# loss & optimizer
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# training
for epoch in range(500):

  # Forward Pass : using model 
  y_pred = model(x_data)

  loss = criterion(y_pred, y_data)
  print('epoch=', epoch, 'loss=', loss.data.item())

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

# Testing
test_var = torch.tensor([[1.0]])
print('test 1: ', test_var.data[0][0].item(), model.forward(test_var).data[0][0].item() > 0.5)
test_var = torch.tensor([[7.0]])
print('test 2: ', test_var.data[0][0].item(), model.forward(test_var).data[0][0].item() > 0.5)