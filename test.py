import torch

loss = torch.nn.CrossEntropyLoss()

y = torch.LongTensor([0])
print(y)

y_pred1 = torch.tensor([[2.0, 1.0, 0.1]])
y_pred2 = torch.tensor([[0.5, 2.0, 0.3]])

l1 = loss(y_pred1, y)
l2 = loss(y_pred2, y)

print(l1)
print(l2)