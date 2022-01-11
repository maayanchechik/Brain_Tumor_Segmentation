import torch
from torch.utils.data import DataLoader, Dataset, Sampler
import numpy as np

class dataset(Dataset):
  def __init__(self):
    self.data = range(0,9)
  def __getitem__(self, index):
    d = self.data[index]
    return d
  def __len__(self):
    return len(self.data)


class sampler(Sampler):
  def __init__(self, batch_len, data_len):
    self.batch_len = batch_len
    self.data_len = data_len
  def __iter__(self):
    for i in range(self.batch_len):
      index = np.random.randint(low = 0, high=9)
      yield index

data = dataset()
train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(data, [3,3,3])

train_dataloader = DataLoader(train_dataset, batch_size=1)
vali_dataloader = DataLoader(validation_dataset, batch_size=1)
test_dataloader = DataLoader(test_dataset, batch_size=1)

print("train_dataset")
for i in enumerate(train_dataloader):
  print(i)

print("validation_dataset")
for i in enumerate(vali_dataloader):
  print(i)

print("test_dataset")
for i in enumerate(test_dataloader):
  print(i)
