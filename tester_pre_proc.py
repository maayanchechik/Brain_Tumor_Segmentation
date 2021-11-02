import h5py
import torch
import numpy as np
from matplotlib import pyplot as plt

###This file tests the data_preprocessing.py file to see if its group_brain() function groups the brain slices into a brain correctly.
###It does so by drawing a horizontal and a vertical slice and seeing that they both create a brain shape
p1_brain_file_path = '/home/mc/Brain_Tumor_Segmentation/data/data/BraTS2020_training_data/content/data_patients/volume_41_image.pt'
f = h5py.File('/home/mc/Brain_Tumor_Segmentation/data/data/BraTS2020_training_data/content/data_slices/volume_41_slice_80.h5', 'r')
p1_brain = torch.load(p1_brain_file_path)
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

