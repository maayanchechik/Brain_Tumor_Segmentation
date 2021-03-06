import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

class BratsDataset(Dataset):
    def __init__(self, patch_size, len_dataset, transform, patching, is_test):
        basedir = '/home/mc/Brain_Tumor_Segmentation/data/data/'
        #later add a self.cash that has some of the data to save time,
        #bc some of the data is in RAM and not disc
        super().__init__()
        if is_test:
            self.data_path = basedir + "BraTS2020_test_data/"
        else:
            self.data_path = basedir + "BraTS2020_training_data/content/data_patients/"
        #self.data_path = basedir + "BraTS2020_training_data/content/one_patient/"
        self.patch_size = patch_size
        self.brain_size = [4,240,240,154]
        # self.brains = pd.read_csv(basedir + 'BraTS20\ Training\ Metadata.csv')
        self.len_dataset = len_dataset
        self.transform = transform
        self.patching = patching

        
    def __len__(self):
        return self.len_dataset

    def get_brain(self,index):
        index_path_image = self.data_path + "volume_" + str(index+1) + "_image.pt"
        index_path_labels = self.data_path + "volume_" + str(index+1) + "_labels.pt"
        brain_image = torch.load(index_path_image)
        brain_labels = torch.load(index_path_labels)
        # GGG print("get_brain: brain_image = ", index_path_image)
        return brain_image, brain_labels

    def fix_patch_center(self, center_per_dim, dim):
        if(center_per_dim+1+(self.patch_size//2) > self.brain_size[dim]):
            return self.brain_size[dim]-self.patch_size/2 - 1
        if(center_per_dim -(self.patch_size//2 - 1) < 0):
            return self.patch_size//2 - 1
        else:
            return center_per_dim

    # This could have been the way it was implemented in the paper, yet that way is flawed
    # so here I'd like to add black pixels to the outofbounds part 
    # (the same way it is outside of the brain when it's not out of bounds). 
    def fix_patch(self, center_per_dim, brain_image, brain_labels, dim):
        # (1)Notice that the center index in an even sized patch is left of the two centers.
        # Therefor, right to the center index there will always be patch_size/2 cells,
        # but left to the center index there will be patch_size/2 cells for uneven patch sizes,
        # and patch_size/2-1 cells for even patch sizes.
        if(center_per_dim+1+(self.patch_size//2) > self.brain_size[dim]):
            dim_extra = center_per_dim+1+(self.patch_size//2) - self.brain_size[dim]
            shape_of_extra = self.brain_size.copy()
            #print("shape_of_extra",shape_of_extra,"dim_extra",dim_extra, "self.brain_size", self.brain_size )
            shape_of_extra[dim] = dim_extra
            #print("shape_of_extra after = ",shape_of_extra)
            extra = torch.zeros(shape_of_extra)
            #print("extra.shape", extra.shape, "brain_image", brain_image.shape)
            brain_image = torch.cat((brain_image, extra), dim)
            self.brain_size = list(brain_image.shape)
            extra[3][:][:][:] = 1
            brain_labels = torch.cat((brain_labels, extra),dim)
            return center_per_dim, brain_image, brain_labels

        if(center_per_dim -(self.patch_size/2 - 1) < 0): #/ and not // bc of (1)
            if self.patch_size%2==0:
                dim_extra = 0 - (center_per_dim -(self.patch_size//2 - 1))
            else:
                dim_extra = 0 - (center_per_dim -(self.patch_size//2))
            shape_of_extra = self.brain_size.copy()
            #print("shape_of_extra",shape_of_extra,"dim_extra",dim_extra, "self.brain_size", self.brain_size )
            shape_of_extra[dim] = dim_extra
            #print("shape_of_extra after = ",shape_of_extra)
            extra = torch.zeros(shape_of_extra)
            #print("extra.shape", extra.shape, "brain_image", brain_image.shape)
            brain_image = torch.cat((extra, brain_image), dim)
            self.brain_size = list(brain_image.shape)
            brain_labels[3][:][:][:] = 1
            brain_labels = torch.cat((extra, brain_labels), dim)
            center_per_dim = center_per_dim + dim_extra
            return center_per_dim, brain_image, brain_labels
        else:
            return center_per_dim, brain_image, brain_labels

    def get_patch_random(self, brain_image, brain_labels):
        i_s = np.array([0,0,0,0])
        i_e = np.array([0,0,0,0])
        for dim in range(1,4):
            cur_i_patch_center = np.random.randint(0, self.brain_size[dim])
            cur_i_patch_center = self.fix_patch_center(cur_i_patch_center, dim)
            i_s[dim] = int(cur_i_patch_center +1 - self.patch_size/2)
            i_e[dim] = int(cur_i_patch_center +1+ self.patch_size/2)
        patch_image = brain_image[:, i_s[1]:i_e[1], i_s[2]:i_e[2], i_s[3]:i_e[3]]
        patch_labels = brain_labels[:, i_s[1]:i_e[1], i_s[2]:i_e[2], i_s[3]:i_e[3]]
        return patch_image, patch_labels
    
    def get_patch_random_center_tumor(self, brain_image, brain_labels, p=0.5):
        assert  self.brain_size == [4,240,240,154], "brain_size wrong"
        random = torch.rand(1)
        if random.item() < p:
            tumor_indices_per_dim = np.where(brain_labels==1)
        else:
            tumor_indices_per_dim = np.where(brain_labels==0)
        index = np.random.randint(0, len(tumor_indices_per_dim[0]))

        i_s = np.array([0,0,0,0])
        #i_end_patch_per_dim = []
        i_e = np.array([0,0,0,0])
    
        for dim in range(1,4):
            #print("dim=",dim," self.brain_size=",self.brain_size)
            cur_i_patch_center = tumor_indices_per_dim[dim][index]
            cur_i_patch_center, brain_image, brain_labels = self.fix_patch(cur_i_patch_center, brain_image, brain_labels, dim)
            #print("after fix patch", " self.brain_size=",self.brain_size)
            i_s[dim] = int(cur_i_patch_center +1 - self.patch_size/2)
            i_e[dim] = int(cur_i_patch_center +1+ self.patch_size/2)
            #print("i_s[dim]",i_s[dim],"i_e[dim]",i_e[dim])
        ###print("            get_patch: index_start = ", i_s, " index_end = ", i_e)
              
        patch_image = brain_image[:, i_s[1]:i_e[1], i_s[2]:i_e[2], i_s[3]:i_e[3]]
        patch_labels = brain_labels[:, i_s[1]:i_e[1], i_s[2]:i_e[2], i_s[3]:i_e[3]]
        self.brain_size = [4,240,240,154]#the brain_size needs to be reset for the next brain
        return patch_image, patch_labels

        
    def get_patch_center_tumor(self, brain_image, brain_labels):
        #where() returns a list of arrays. Each array represents a dimention.
        #In each array there are the dimention's indices of the labels that are 1.
        #All in all this list is an ordered account of the indices of tumor pixels.
        tumor_indices_per_dim = np.where(brain_labels==1)

        #index is randomly chosen int that chooses which tumor index will be the center
        
        ######ONLY FOR OVERFITTING
        #index = 0
        index = np.random.randint(0, len(tumor_indices_per_dim[0]))
                
        #The first cell is the dim of the modules which is not sliced,
        #so will stay empty but is here so it wont confuse the dim count.
        #i_start_patch_per_dim = [] but name to long so:
        i_s = np.array([0,0,0,0])
        #i_end_patch_per_dim = []
        i_e = np.array([0,0,0,0])
        
        for dim in range(1,4):
            cur_i_patch_center = tumor_indices_per_dim[dim][index]
            cur_i_patch_center = self.fix_patch_center(cur_i_patch_center, dim)
            i_s[dim] = int(cur_i_patch_center +1 - self.patch_size/2)
            i_e[dim] = int(cur_i_patch_center +1+ self.patch_size/2)
        ###print("            get_patch: index_start = ", i_s, " index_end = ", i_e)
            
        patch_image = brain_image[:, i_s[1]:i_e[1], i_s[2]:i_e[2], i_s[3]:i_e[3]]
        patch_labels = brain_labels[:, i_s[1]:i_e[1], i_s[2]:i_e[2], i_s[3]:i_e[3]]
        return patch_image, patch_labels

    
    def __getitem__(self, brain_index):
        brain_image, brain_labels = self.get_brain(brain_index)
        if self.patching == 'CENTER_TUMOR':
            patch_image, patch_labels = self.get_patch_center_tumor(brain_image, brain_labels)
        elif self.patching == 'RANDOM_CENTER_TUMOR':
            patch_image, patch_labels = self.get_patch_random_center_tumor(brain_image, brain_labels)
        elif self.patching == 'RANDOM':
            patch_image, patch_labels = self.get_patch_random(brain_image, brain_labels)
        else:
            print("ERROR PATCHING UNKNOWN")
            quit()
        patch = (patch_image, patch_labels)
        if self.transform != None:
            patch = self.transform(patch)
        #print(patch)
        #i_p, l_p = patch
        #print("patch type", dtype(i_p), dtype(l_p))
        ### print("__get_item__: brain_index = ", brain_index)
        return patch




