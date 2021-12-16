import torch
from torch.utils.data import Dataset
import h5py
from torchvision import transforms as T
import numpy as np
import os

def resize_brain_labels(tensor_brain_labels):
    tumor_voxels = tensor_brain_labels[0]+tensor_brain_labels[1]+tensor_brain_labels[2]
    non_tumor = 1-tumor_voxels
    non_tumor = torch.unsqueeze(non_tumor, dim=0)
    new_brain_labels = torch.cat((tensor_brain_labels, non_tumor), dim=0)
    print("tensor_brain_labels.shape", new_brain_labels.shape)
    return new_brain_labels

#This function groups each paitients brain from 0-154 slices to a full brain that is a tensor and saves it
def group_brain():
    file_path = '/home/mc/Brain_Tumor_Segmentation/data/data/BraTS2020_training_data/content/data_slices/volume_'
    new_file_path = '/home/mc/Brain_Tumor_Segmentation/data/data/BraTS2020_training_data/content/data_patients'
    for patient_num in range(1,370):
        for slice_num in range(154):
            #read slice file
            slice_file_path = file_path + str(patient_num) + '_slice_' +str(slice_num) +'.h5'
            slice_file = h5py.File(slice_file_path, 'r')
            
            #create numpy array of the images and the labels from the slice file
            hdf5_slice_image = slice_file['image']
            np_slice_image = np.array(hdf5_slice_image)
            
            hdf5_slice_labels = slice_file['mask']
            np_slice_labels = np.array(hdf5_slice_labels)

            #concat the new slice to the brain
            if slice_num == 0:
                np_brain_image = np_slice_image
                np_brain_labels = np_slice_labels
            elif slice_num == 1:
                np_brain_image = np.stack((np_brain_image, np_slice_image),axis=0)
                np_brain_labels = np.stack((np_brain_labels, np_slice_labels),axis=0)
            else:
                np_slice_image = np.expand_dims(np_slice_image,axis=0)
                np_slice_labels = np.expand_dims(np_slice_labels,axis=0)
                np_brain_image = np.concatenate((np_brain_image, np_slice_image),axis=0)
                np_brain_labels = np.concatenate((np_brain_labels, np_slice_labels),axis=0)

        print("patient ", patient_num)
        tensor_brain_image = torch.FloatTensor(np_brain_image)
        tensor_brain_labels = torch.FloatTensor(np_brain_labels)
        print("tensor_brain_image",tensor_brain_image.type())
        #permute the brain array to the dimentions expected in the network
        print("tensor_brain_image.size() ",tensor_brain_image.size())
        tensor_brain_image = tensor_brain_image.permute(3,1,2,0)
        tensor_brain_labels = tensor_brain_labels.permute(3,1,2,0)
        #resize brain labels to have 4 arrays in the first dimention (representing classes),
        #one for a voxel being non tumor
        tensor_brain_labels = resize_brain_labels(tensor_brain_labels)
        #save brain_image and brain labels of this patient
        brain_name_image = "volume_" + str(patient_num) + "_image.pt"
        brain_name_labels = "volume_" + str(patient_num) + "_labels.pt"
        save_path_brain_image = os.path.join(new_file_path,brain_name_image)        
        save_path_brain_labels = os.path.join(new_file_path,brain_name_labels)
        torch.save(tensor_brain_image, save_path_brain_image)
        torch.save(tensor_brain_labels, save_path_brain_labels)



group_brain()
