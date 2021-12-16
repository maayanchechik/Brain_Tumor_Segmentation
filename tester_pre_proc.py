import h5py
import torch
import numpy as np
from matplotlib import pyplot as plt

###This file tests the data_preprocessing.py file to see if its group_brain() function groups the brain slices into a brain correctly.
###It does so by drawing a horizontal and a vertical slice and seeing that they both create a brain shape
p1_brain_file_path = '/home/mc/Brain_Tumor_Segmentation/data/data/BraTS2020_training_data/content/data_patients/volume_41_labels.pt'
p2_brain_file_path = '/home/mc/Brain_Tumor_Segmentation/data/data/BraTS2020_training_data/content/data_patients/volume_41_image.pt'
f = h5py.File('/home/mc/Brain_Tumor_Segmentation/data/data/BraTS2020_training_data/content/data_slices/volume_41_slice_80.h5', 'r')
#p1_brain = torch.load(p1_brain_file_path)
brain = torch.load(p1_brain_file_path)
brain_image = torch.load(p2_brain_file_path)
print("brain_image.shape = ",brain_image.shape)
print("brain.shape = ",brain.shape)
print("brain[0].shape = ",brain[0].shape)
#non_tumor = torch.zeros((1,240,240,154))
#new = torch.cat((brain, non_tumor), dim=0)
#print("new shape = ",new.shape)
tumor_voxels = brain[0]+brain[1]+brain[2]
non_tumor = 1-tumor_voxels
print("tumor_voxels[0][0][0]",tumor_voxels[0][0][0])
print("non_tumor[0][0][0]",non_tumor[0][0][0])
non_tumor = torch.unsqueeze(non_tumor, dim=0)
print("non_tumor.shape",non_tumor.shape)
print("torch.max(tumor_voxels)",torch.max(tumor_voxels))
print("torch.min(tumor_voxels)",torch.min(tumor_voxels))
print("torch.max(non_tumor)",torch.max(non_tumor))
print("torch.min(non_tumor)",torch.min(non_tumor))
new_brain = torch.cat((brain, non_tumor), dim=0)
print("new_brain.shape",new_brain.shape)
exit()






p1_brain = p1_brain.permute(1,2,3,0)
np_brain = p1_brain.numpy()

#original data slices
data = np.array(f['image'])
im = data[:][:][:]
plt.imshow(im, interpolation='nearest')
plt.show()

#this should math the original data slices
vertical_slice = np_brain[:, : , 80 , :]
print(vertical_slice.shape)
plt.imshow(vertical_slice, interpolation='nearest')
plt.show()

horizontal_slice = np_brain[ : , 100 , : , :]
print(horizontal_slice.shape)
plt.imshow(horizontal_slice, interpolation='nearest')
plt.show()

