import torch
from torch.autograd import Variable

#  data : 3 X 1 tensor
x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])

class Model(torch.nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.linear = torch.nn.Linear(1, 1)

  def forward(self, x):
    y_pred = self.linear(x)
    return y_pred

# our model
model = Model()

# loss & optimizer
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# training
for epoch in range(1000):

  # Forward Pass : using model 
  y_pred = model(x_data)

  loss = criterion(y_pred, y_data)
  print('epoch=', epoch, 'loss=', loss.data.item())

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

# Testing
test_var = torch.tensor([[4.0]])
print('test ', test_var.data[0][0].item(), model.forward(test_var).data[0][0].item())