import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class DiabetesDataset(Dataset):
  # Initialize data, download, read data etc.
  def __init__(self):
    xy = np.loadtxt('data/diabetes.csv', delimiter=',', dtype=np.float32)
    self.x_data = torch.from_numpy(xy[:, 0:-1])
    self.y_data = torch.from_numpy(xy[:, [-1]])
    self.len = xy.shape[0]
  
  # return one item given index
  def __getitem__(self, index):
    return self.x_data[index], self.y_data[index]
  # return data length
  def __len__(self):
    return self.len

class Model(torch.nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.l1 = torch.nn.Linear(8, 6)
    self.l2 = torch.nn.Linear(6, 4)
    self.l3 = torch.nn.Linear(4, 1)

  def forward(self, x):
    y_pred = torch.sigmoid(self.l3(self.l2(self.l1(x))))
    return y_pred

# our model
model = Model()

# loss & optimizer
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

dataset = DiabetesDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=16,
                          shuffle=True,
                          num_workers=0)

for epoch in range(10):
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = torch.tensor(inputs), torch.tensor(labels)

        y_pred = model(inputs)

        loss = criterion(y_pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Run your training process
        print(f'Epoch: {epoch} | Iteration: {i} | Loss: {loss.data.item()}')
