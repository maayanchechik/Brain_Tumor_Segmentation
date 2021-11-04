import torch
from torch.utils.data import BatchSampler
import numpy as np

class BratsSampler(BatchSampler):
    def __init__(self, batch_len):
        self.batch_len = batch_len


    def __iter__(self):
        batch = np.random.randint(low=1, high=370, size = self.batch_len)
        yield batch
        
            
    
