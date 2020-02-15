import numpy as np

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = np.random.uniform(0, 4.1)
print("intial w = ",w)

# learning rate
lr = 0.01

# model for the forward pass
def forward(x):
  return x * w

# Loss Fucntion
def loss(x, y):
  y_pred = forward(x)
  return (y_pred - y) * (y_pred - y)

# gradient decendant
def gradient(x, y):
  return 2 * x * (x * w - y)

for epoch in range(0,100):
  for x_val, y_val in zip(x_data, y_data):
    grad = gradient(x_val, y_val)
    w = w - lr * grad
    l = loss(x_val, y_val)
  print('epoch=', epoch, 'loss=', l)



