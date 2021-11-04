import torch
from torch.utils.data import DataLoader
from Brats_dataset import BratsDataset
from Brats_sampler import BratsSampler #change it

dataset = BratsDataset(patch_size = 112)
sampler = BratsSampler(batch_len=2)
dataloader = DataLoader(dataset = dataset, batch_sampler = sampler, num_workers = 1)

for batch_i,(patch_image, patch_labels) in enumerate(dataloader):
    print("batch ",batch_i)
    print(patch_image.size())
    print(patch_labels.size())
