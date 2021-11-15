import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class BratsDataset(Dataset):
    def __init__(self,patch_size):
        #later add a self.cash that has some of the data to save time,
        #bc some of the data is in RAM and not disc
        super().__init__()
        self.data_path = "/home/mc/Brain_Tumor_Segmentation/data/data/BraTS2020_training_data/content/data_patients/"
        self.patch_size = patch_size
        self.brain_size = np.array([4,240,240,154])

    def get_brain(self,index):
        index_path_image = self.data_path + "volume_" + str(index) + "_image.pt"
        index_path_labels = self.data_path + "volume_" + str(index) + "_labels.pt"
        brain_image = torch.load(index_path_image)
        brain_labels = torch.load(index_path_labels)
        return brain_image, brain_labels

    def fix_patch_center(self, center_per_dim, dim):
        if(center_per_dim+(self.patch_size//2) > self.brain_size[dim]):
            return self.brain_size[dim]-self.patch_size/2
        if(center_per_dim -(self.patch_size//2 - 1) < 0):
            return self.patch_size//2 - 1
        else:
            return center_per_dim
        

    def get_patch(self, brain_image, brain_labels):
        #where() returns a list of arrays. Each array represents a dimention.
        #In each array there are the dimention's indices of the labels that are 1.
        #All in all this list is an ordered account of the indices of tumor pixels.
        tumor_indices_per_dim = np.where(brain_labels==1)

        #index is randomly chosen int that chooses which tumor index will be the center 
        index = np.random.randint(0, len(tumor_indices_per_dim[0]))

        #The first cell is the dim of the modules which is not sliced,
        #so will stay empty but is here so it wont confuse the dim count.
        #i_start_patch_per_dim = [] but name to long so:
        i_s = np.array([0,0,0,0])
        #i_end_patch_per_dim = []
        i_e = np.array([0,0,0,0])
        
        for dim in range(1,4):
            cur_i_patch_center = tumor_indices_per_dim[dim][index]
            cur_i_patch_center = self.fix_patch_center(cur_i_patch_center,dim)
            i_s[dim] = int(cur_i_patch_center - self.patch_size/2)
            i_e[dim] = int(cur_i_patch_center + self.patch_size/2)

        print("i_s",i_s)
        print("i_e",i_e)
        patch_image = brain_image[:, i_s[1]:i_e[1], i_s[2]:i_e[2], i_s[3]:i_e[3]]
        patch_labels = brain_labels[:, i_s[1]:i_e[1], i_s[2]:i_e[2], i_s[3]:i_e[3]]
        print("patch_image size", patch_image.shape)
        print("brain_image",brain_image.shape)
        return patch_image, patch_labels


    def __getitem__(self, index):
        brain_image, brain_labels = self.get_brain(index)
        patch_image, patch_labels = self.get_patch(brain_image, brain_labels)
        patch = (patch_image,patch_labels)
        return patch

    def __len__(self):
        return 369

