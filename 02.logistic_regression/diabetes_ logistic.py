import numpy as np
import torch

xy = np.loadtxt('data/diabetes.csv', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:, 0:-1])
y_data = torch.from_numpy(xy[:, [-1]])

class Model(torch.nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.l1 = torch.nn.Linear(8, 6)
    self.l2 = torch.nn.Linear(6, 4)
    self.l3 = torch.nn.Linear(4, 1)

    self.sigmoid = torch.sigmoid

  def forward(self, x):
    y_pred = self.sigmoid(self.l3( self.sigmoid(self.l2( self.sigmoid(self.l1(x))))))
    return y_pred

# our model
model = Model()

# loss & optimizer
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# training
for epoch in range(1500):

  # Forward Pass : using model 
  y_pred = model(x_data)

  loss = criterion(y_pred, y_data)
  print('epoch=', epoch, 'loss=', loss.data.item())

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()