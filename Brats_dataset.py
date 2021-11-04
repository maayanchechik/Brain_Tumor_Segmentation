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

    def get_brain(self,index):
        index_path_image = self.data_path + "volume_" + str(index) + "_image.pt"
        index_path_labels = self.data_path + "volume_" + str(index) + "_labels.pt"
        brain_image = torch.load(index_path_image)
        brain_labels = torch.load(index_path_labels)
        return brain_image, brain_labels

    def get_patch(self, brain_image, brain_labels):
        #where retuns a list of arrays. Each array represents a dimention.
        #In each array there is the dimentions indices of the labels that are 1.
        #All in all this list is an ordered account of the indices of tumor pixels.
        tumor_indices_per_dim = np.where(brain_labels==1)

        ###instead of : do loop
        #modalities = array_of_indexes[0]
        #x = array_of_indexes[1]
        #y =array_of_indexes[2]
        #z = array_of_indexes[3]
        
        #tumor_indices_list_per_dim = []
        #for dim in range(4):
        #    tumor_indices_list_per_dim.append(list_array_of_i_per_dim[dim])
        

        #index is randomly chosen to give the indices for this patch's center 
        index = np.random.randint(0, len(tumor_indices_per_dim[0]))

        print(tumor_indices_per_dim[1])
        ###instead of: do loop 
        #m_i = modalities[index]
        #x_i = x[index]
        #y_i = y[index]
        #z_i = z[index]

        #i_start_patch_per_dim = [] but name to long so:
        i_s = []
        #i_end_patch_per_dim = []
        i_e = []
        for dim in range(4):
            cur_i_patch_center = tumor_indices_per_dim[dim][index]
            i_patch_start = int(cur_i_patch_center - self.patch_size/2)
            i_patch_end = int(cur_i_patch_center + self.patch_size/2)
            i_s.append(i_patch_start)
            i_e.append(i_patch_end)
            
        patch_image = brain_image[i_s[0]:i_e[0], i_s[1]:i_e[1], i_s[2]:i_e[2], i_s[3]:i_e[3]]
        print("patch_image",patch_image.shape)
        patch_labels = brain_labels[i_s[0]:i_e[0], i_s[1]:i_e[1], i_s[2]:i_e[2], i_s[3]:i_e[3]]

        return patch_image, patch_labels

    def __getitem__(self, index):
        brain_image, brain_labels = self.get_brain(index)
        patch_image, patch_labels = self.get_patch(brain_image, brain_labels)
        patch = (patch_image,patch_labels)
        return patch

    def __len__(self):
        return 369

