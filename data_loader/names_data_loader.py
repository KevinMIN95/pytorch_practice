import numpy as np
import csv
import torch
from torch.utils.data import Dataset, DataLoader

class NameDataset(Dataset):
  def __init__(self, type = 'train'):
    file_name = 'data/names_train.csv' if type == 'train' else 'data/names_test.csv'
    f = open(file_name, 'r')
    reader = csv.reader(f)
    rows = list(reader)

    self.len = len(rows)
    self.names = [row[0] for row in rows]
    self.countries  = [row[1] for row in rows]

    self.country_list = list(sorted(set(self.countries)))
    f.close()

  # return one item given index
  def __getitem__(self, index):
    return self.names[index], self.countries[index]
  # return data length
  def __len__(self):
    return self.len

  def get_countries(self):
    return self.country_list

  def get_country(self, id):
    return self.country_list[id]

  def get_country_id(self, country):
    return self.country_list.index(country)

# test dataset
if __name__ == '__main__':
  name_dataset = NameDataset()

  train_loader = DataLoader(dataset=name_dataset,
                            batch_size=10,
                            shuffle=True)
  print(len(train_loader.dataset))
  for epoch in range(1):
    for i, (names, countries) in enumerate(train_loader):
      if i==0:
        print(epoch, i, "names", names, "countries", countries)
