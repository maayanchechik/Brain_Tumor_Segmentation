import torch
import torch.utils.data.BatchSampler
import numpy

class Brats_sampler(BatchSampler):
    def __init__(self, batch_len, patch_size):
        self.batch_len = batch_len
        self.patch_size = patch_size

    def __iter__(self):
        batch = np.random.randint(low=1, high=370, size = batch_len)
        yield batch
        
            
    
