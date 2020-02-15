import sys
import torch
from torch import nn

# data_preparation
idx2char = ['h', 'e', 'l', 'o', 'EOS']
h = [1, 0, 0, 0, 0]
e = [0, 1, 0, 0, 0]
l = [0, 0, 1, 0, 0]
o = [0, 0, 0, 1, 0]
EOS = [0, 0, 0, 0, 1]

x_data = [0, 1, 2, 2, 3] # hello
x_one_hot = [h, e, l, l, o]

y_data = [1, 2, 2, 3, 4] # ello<EOS>

inputs = torch.tensor(x_one_hot, dtype=torch.float32, requires_grad=True)
labels = torch.tensor([[y_data[y]] for y in range(len(y_data))]).long()

num_classes = 5
input_size = 5  # one-hot size
hidden_size = 5  # output from the RNN. 5 to directly predict one-hot
batch_size = 1   # one sentence
sequence_length = 1  # One by one
num_layers = 1  # one-layer rnn

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.rnn = nn.RNN(input_size=input_size,
                          hidden_size=hidden_size, batch_first=True)

    def forward(self, hidden, x):
        # Reshape input (batch first)
        x = x.view(batch_size, sequence_length, input_size)

        # Propagate input through RNN
        # Input: (batch, seq_len, input_size)
        # hidden: (num_layers * num_directions, batch, hidden_size)
        out, hidden = self.rnn(x, hidden)
        return hidden, out.view(-1, num_classes)

    def init_hidden(self):
        # Initialize hidden and cell states
        # (num_layers * num_directions, batch, hidden_size)
        return torch.zeros(num_layers, batch_size, hidden_size, dtype=torch.float32, requires_grad=True)


# Instantiate RNN model
model = Model()
# print(model)

# Set loss and optimizer function
# CrossEntropyLoss = LogSoftmax + NLLLoss
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# Train the model
for epoch in range(10):
    optimizer.zero_grad()
    loss = 0
    hidden = model.init_hidden()

    sys.stdout.write("predicted string: ")
    for input, label in zip(inputs, labels):
      hidden, output = model(hidden, input)
      val, idx = output.max(1)
      sys.stdout.write(idx2char[idx.data[0]])
      loss += criterion(output, label)

    print(", epoch: %d, loss: %1.3f" % (epoch + 1, loss.data.item()))

    loss.backward()
    optimizer.step()

print("Learning finished!")