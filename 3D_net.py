import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import interpolate
pool_kernal = 2
class 3D_UNet(nn.Module):
    def __init__(self, num_groups = 8, num_channels, padding = 1, dropout_percent):
        super(3D_UNet, self).__init__()
        self.conv1 = nn.Conv3d(4, 32, kernel_size = (3,3,3), padding =padding)
  #      self.gnorm1 = nn.GroupNorm(num_groups,num_channels)
        self.conv2 = nn.Conv3d(32, 32, kernel_size = (3,3,3), padding = padding)
  #      self.gnorm2 = nn.GroupNorm(num_groups,num_channels)
        
        self.conv3 = nn.Conv3d(32, 64, kernel_size = (3,3,3), padding = padding)
  #      self.gnorm3 = nn.GroupNorm(num_groups,num_channels)
        self.conv4 = nn.Conv3d(64, 64, kernel_size = (3,3,3), padding = padding)
  #      self.gnorm4 = nn.GroupNorm(num_groups,num_channels)

        self.conv5 = nn.Conv3d(64, 128, kernel_size = (3,3,3), padding = padding)
  #      self.gnorm5 = nn.GroupNorm(num_groups,num_channels)
        self.conv6 = nn.Conv3d(128, 128, kernel_size = (3,3,3), padding = padding)
  #      self.gnorm6 = nn.GroupNorm(num_groups,num_channels)
        
        self.conv7 = nn.Conv3d(128, 256, kernel_size = (3,3,3), padding = padding)
  #      self.gnorm7 = nn.GroupNorm(num_groups,num_channels)
        self.conv8 = nn.Conv3d(256, 256, kernel_size = (3,3,3), padding = padding)
  #      self.gnorm8 = nn.GroupNorm(num_groups,num_channels)

        self.conv9 = nn.Conv3d(256, 128, kernel_size = (3,3,3), padding = padding)
  #      self.gnorm9 = nn.GroupNorm(num_groups,num_channels)
        self.conv10 = nn.Conv3d(128, 128, kernel_size = (3,3,3), padding = padding)
#      self.gnorm10 = nn.GroupNorm(num_groups,num_channels)

        self.conv11 = nn.Conv3d(128, 64, kernel_size = (3,3,3), padding = padding)
  #      self.gnorm9 = nn.GroupNorm(num_groups,num_channels)
        self.conv12 = nn.Conv3d(64, 64, kernel_size = (3,3,3), padding = padding)
  #      self.gnorm10 = nn.GroupNorm(num_groups,num_channels)

        self.conv13 = nn.Conv3d(64, 32, kernel_size = (3,3,3), padding = padding)
  #      self.gnorm11 = nn.GroupNorm(num_groups,num_channels)
        self.conv14 = nn.Conv3d(32, 32, kernel_size = (3,3,3), padding = padding)
  #      self.gnorm12 = nn.GroupNorm(num_groups,num_channels)

        self.conv_last = nn.Conv3d(32, 4, kernel_size = (1,1,1)) #does this have padding?

        #other
        self.relu = nn.ReLU()
        self.gnorm = nn.GroupNorm(num_groups,num_channels)
        self. dropout = nn.Dropout(dropout_percent)
        self.final_activation = nn.Softmax(dim = 1)

    def forward(self, X):
        y1 = self.gnorm(self.relu(self.conv1(X)))
        y2 = self.gnorm(self.relu(self.conv2(y1)))
        y2_pooled = nn.MaxPool3D(y2, pool_kernal)
        
        y2_drop = self.dropout(y2_pooled)
        y3 = self.gnorm(self.relu(self.conv3(y2_drop)))
        y4 = self.gnorm(self.relu(self.conv4(y3)))
        y4_pooled = nn.MaxPool3D(y4, pool_kernal)

        y4_drop = self.dropout(y4_pooled)
        y5 = self.gnorm(self.relu(self.conv5(y4_drop)))
        y6 = self.gnorm(self.relu(self.conv6(y5)))
        y6_pooled = nn.MaxPool3D(y6, pool_kernal)

        y6_drop = self.dropout(y6_pooled)
        y7 = self.gnorm(self.relu(self.conv7(y6_drop)))
        y8 = self.gnorm(self.relu(self.conv8(y7)))
        y8_up = interpolate(y8, scale_factor = 2, mode='nearest')
        y8_con = torch.cat((y6, y8_up), dim=1)

        y8_drop = self.dropout(y8_con)
        y9 = self.gnorm(self.relu(self.conv9(y8_drop)))
        y10 = self.gnorm(self.relu(self.conv10(y9)))
        y10_up = interpolate(y10, scale_factor = 2, mode='nearest')
        y10_con = torch.cat((y4, y10_up), dim=1)

        y10_drop = self.dropout(y10_con)
        y11 = self.gnorm(self.relu(self.conv11(y10_drop)))
        y12 = self.gnorm(self.relu(self.conv12(y11)))
        y12_up = interpolate(y12, scale_factor = 2, mode='nearest')
        y12_con = torch.cat((y2, y12_up), dim=1)

        y12_drop = self.dropout(y12_con)
        y13 = self.gnorm(self.relu(self.conv13(y12_drop)))
        y14 = self.gnorm(self.relu(self.conv14(y13)))
        cov_out = self.conv_last(y14)
        activation_out = final_activation(y14)
        


        
