import torch
from torch.utils.data import DataLoader
from Brats_dataset import BratsDataset
from Brats_sampler import BratsSampler 

dataset = BratsDataset(patch_size = 112)
# sampler = BratsSampler(batch_size = 4)
#dataloader = DataLoader(dataset = dataset, batch_sampler = sampler, num_workers = 2)

dataloader = DataLoader(dataset = dataset,
                        batch_size=4,
                        num_workers = 2,
                        shuffle=True)

print("GGG: Num batches = ", len(dataloader))

for batch_i, (patch_image, patch_labels) in enumerate(dataloader):
    print("batch ",batch_i)
    print("tester_dataloader: patch_image.size = ", patch_image.size())
    print("tester_dataloader: patch_labels.size = ", patch_labels.size())    
print("here")
