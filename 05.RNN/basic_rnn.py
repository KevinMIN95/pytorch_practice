import torch
from torch import nn

# One hot encoding for each char in 'hello'
h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]

cell = nn.RNN(input_size=4, hidden_size=2, batch_first=True)

# Input: (batch, seq_len, input_size) when batch_first=True
inputs = torch.tensor([[h]], dtype=torch.float32, requires_grad=True)

# (num_layers * num_directions, batch, hidden_size) regardless of batch_first
hidden = torch.randn(1, 1, 2, requires_grad=True)

out, hidden = cell(inputs, hidden)
print('out', out)

# batch: 1, seq_len: 4, input_size: 4
inputs = torch.tensor([[h, e, l, o]], dtype=torch.float32, requires_grad=True)

out, hidden = cell(inputs, hidden)
print('out', out.data)
print('out_size', out.data.size()) # (1, 4, 2)

# 3 batches : 'hello', 'elloo', 'lloee'
# input size: (3, 5, 4)
inputs = torch.tensor([[h, e, l, l, o],
                      [e, l, l, o, o],
                      [l, l, o, o, e]], dtype=torch.float32, requires_grad=True)

hidden = torch.randn(1, 3, 2, requires_grad=True)
out, hidden = cell(inputs, hidden)
print('out', out.data)
print('out_size', out.data.size()) # (3, 5, 2)