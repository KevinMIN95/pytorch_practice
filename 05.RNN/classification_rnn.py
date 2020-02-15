import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_loader.names_data_loader import NameDataset

# config
HIDDEN_SIZE = 100
N_CHARS = 128  # ASCII
N_CLASSES = 18
N_LAYERS = 2
BATCH_SIZE = 256
N_EPOCHS = 100

test_dataset = NameDataset(type='test')
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE, shuffle=True)

train_dataset = NameDataset(type='train')
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE, shuffle=True)

N_COUNTRIES = len(train_dataset.get_countries())
print(N_COUNTRIES, "countries")

# utils
def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def str2ascii_arr(s):
  arr = [ ord(x) for x in s]
  return arr, len(arr)

def countries2tensor(countries):
  countries_id = [test_dataset.get_country_id(country) for country in countries]
  return torch.LongTensor(countries_id)

# pad sequences and sort the tensor
def pad_sequences(vectorized_seqs, seq_lengths, countries):
    seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
    for idx, (seq, seq_len) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

    # Sort tensors by their length
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx]

    # Sort countires in the same order
    if len(countries):
      countries = countries[perm_idx]

    return seq_tensor, seq_lengths, countries

# Create necessary variables, lengths, and target
def make_variables(names, countries):
    sequence_and_length = [str2ascii_arr(name) for name in names]
    vectorized_seqs = [sl[0] for sl in sequence_and_length]
    seq_lengths = torch.LongTensor([sl[1] for sl in sequence_and_length])
    vectorized_countries = countries2tensor(countries)

    return pad_sequences(vectorized_seqs, seq_lengths, vectorized_countries)


class RnnClassification(nn.Module):
  def __init__(self, input_size, hidden_size, output_size, n_layers = 1, bidirectional=True):
    super(RnnClassification, self).__init__()
    self.hidden_size = hidden_size
    self.input_size = input_size

    self.n_layers = n_layers
    self.n_directions = int(bidirectional) + 1

    self.embedding = nn.Embedding(input_size, hidden_size)
    self.gru = nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True, bidirectional=bidirectional)
    self.fc = nn.Linear(hidden_size, output_size)
  
  def forward(self, input):
    batch_size = input.size(0)
    seq_length = input.size(1)

    emb = self.embedding(input)
    emb = emb.view(batch_size, seq_length, -1)

    hidden  = self._init_hidden(batch_size)

    output, hidden = self.gru(emb, hidden)

    fc_output = self.fc(hidden[-1])
    return fc_output
    
  def _init_hidden(self, batch_size):
    hidden = torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)
    return hidden

def train():
  total_loss = 0
  for idx, (names, countries) in enumerate(train_loader, 1):
    inputs, _, targets = make_variables(names, countries)
    outputs = classifier(inputs)

    loss = criterion(outputs, targets)
    total_loss += loss.data.item()

    classifier.zero_grad()
    loss.backward()
    optimizer.step()

    if idx % 10 == 0:
      print('[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.2f}'.format(
        time_since(start), epoch,  idx *
        len(names), len(train_loader.dataset),
        100. * idx * len(names) / len(train_loader.dataset),
        total_loss / idx * len(names)))

  return total_loss

# Testing cycle
def test(name=None):
    # Predict for a given name
    if name:
        input, _, target = make_variables([name], [])
        output = classifier(input)
        pred = output.data.max(1, keepdim=True)[1]
        country_id = pred.cpu().numpy()[0][0]
        print(name, "is", train_dataset.get_country(country_id))
        return

    print("evaluating trained model ...\n****************")
    correct = 0
    test_data_size = len(test_loader.dataset)

    for names, countries in test_loader:
        inputs, _, targets = make_variables(names, countries)
        outputs = classifier(inputs)
        pred = outputs.data.max(1, keepdim=True)[1]
        correct += pred.eq(targets.data.view_as(pred)).cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, test_data_size, 100. * correct / test_data_size))



if __name__ == '__main__':
  classifier = RnnClassification(N_CHARS, HIDDEN_SIZE, N_CLASSES, N_LAYERS)

  optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
  criterion = nn.CrossEntropyLoss()

  start = time.time()
  print("Training for %d epochs..." % N_EPOCHS)
  for epoch in range(1, N_EPOCHS):
      # Train cycle
      train()

      # Testing
      test()

      # Testing several samples
      test("Sung")
      test("Jungwoo")
      test("Soojin")
      test("Nako")


  
