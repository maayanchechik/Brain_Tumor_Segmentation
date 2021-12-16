import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import interpolate
pool_kernal = 2


class UNet_3D(nn.Module):
    def __init__(self, num_groups = 8, padding = 1):
        super(UNet_3D, self).__init__()
        self.conv1 = nn.Conv3d(4, 32, kernel_size = (3,3,3), padding =padding)
        self.gnorm1 = nn.GroupNorm(num_groups,32)
        self.conv2 = nn.Conv3d(32, 32, kernel_size = (3,3,3), padding = padding)
        self.gnorm2 = nn.GroupNorm(num_groups,32)
        
        self.conv3 = nn.Conv3d(32, 64, kernel_size = (3,3,3), padding = padding)
        self.gnorm3 = nn.GroupNorm(num_groups,64)
        self.conv4 = nn.Conv3d(64, 64, kernel_size = (3,3,3), padding = padding)
        self.gnorm4 = nn.GroupNorm(num_groups,64)

        self.conv5 = nn.Conv3d(64, 128, kernel_size = (3,3,3), padding = padding)
        self.gnorm5 = nn.GroupNorm(num_groups,128)
        self.conv6 = nn.Conv3d(128, 128, kernel_size = (3,3,3), padding = padding)
        self.gnorm6 = nn.GroupNorm(num_groups,128)
        
        self.conv7 = nn.Conv3d(128, 256, kernel_size = (3,3,3), padding = padding)
        self.gnorm7 = nn.GroupNorm(num_groups,256)
        self.conv8 = nn.Conv3d(256, 256, kernel_size = (3,3,3), padding = padding)
        self.gnorm8 = nn.GroupNorm(num_groups,256)

        self.conv9 = nn.Conv3d(384, 128, kernel_size = (3,3,3), padding = padding)
        self.gnorm9 = nn.GroupNorm(num_groups,128)
        self.conv10 = nn.Conv3d(128, 128, kernel_size = (3,3,3), padding = padding)
        self.gnorm10 = nn.GroupNorm(num_groups,128)

        self.conv11 = nn.Conv3d(192, 64, kernel_size = (3,3,3), padding = padding)
        self.gnorm11 = nn.GroupNorm(num_groups,64)
        self.conv12 = nn.Conv3d(64, 64, kernel_size = (3,3,3), padding = padding)
        self.gnorm12 = nn.GroupNorm(num_groups,64)

        self.conv13 = nn.Conv3d(96, 32, kernel_size = (3,3,3), padding = padding)
        self.gnorm13 = nn.GroupNorm(num_groups,32)
        self.conv14 = nn.Conv3d(32, 32, kernel_size = (3,3,3), padding = padding)
        self.gnorm14 = nn.GroupNorm(num_groups,32)

        self.conv_last = nn.Conv3d(32, 4, kernel_size = (1,1,1))

        #other
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout3d()
        self.final_activation = nn.Softmax(dim = 1)
        self.maxpool = nn.MaxPool3d(pool_kernal)
        

    def forward(self, X):
        y1 = self.gnorm1(self.relu(self.conv1(X)))
        y2 = self.gnorm2(self.relu(self.conv2(y1)))
        y2_pooled = self.maxpool(y2)
        
        y2_drop = self.dropout(y2_pooled)
        y3 = self.gnorm3(self.relu(self.conv3(y2_drop)))
        y4 = self.gnorm4(self.relu(self.conv4(y3)))
        y4_pooled = self.maxpool(y4)

        y4_drop = self.dropout(y4_pooled)
        y5 = self.gnorm5(self.relu(self.conv5(y4_drop)))
        y6 = self.gnorm6(self.relu(self.conv6(y5)))
        y6_pooled = self.maxpool(y6)

        y6_drop = self.dropout(y6_pooled)
        y7 = self.gnorm7(self.relu(self.conv7(y6_drop)))
        y8 = self.gnorm8(self.relu(self.conv8(y7)))
        y8_up = interpolate(y8, scale_factor = 2, mode='nearest')
        y8_con = torch.cat((y6, y8_up), dim=1)
        
        y8_drop = self.dropout(y8_con)
        y91 = self.conv9(y8_drop)
        y9 = self.gnorm9(self.relu(y91))
        y10 = self.gnorm10(self.relu(self.conv10(y9)))
        y10_up = interpolate(y10, scale_factor = 2, mode='nearest')
        y10_con = torch.cat((y4, y10_up), dim=1)

        y10_drop = self.dropout(y10_con)
        y11 = self.gnorm11(self.relu(self.conv11(y10_drop)))
        y12 = self.gnorm12(self.relu(self.conv12(y11)))
        y12_up = interpolate(y12, scale_factor = 2, mode='nearest')
        y12_con = torch.cat((y2, y12_up), dim=1)

        y12_drop = self.dropout(y12_con)
        y13 = self.gnorm13(self.relu(self.conv13(y12_drop)))
        y14 = self.gnorm14(self.relu(self.conv14(y13)))
        conv_out = self.conv_last(y14)
        activation_out = self.final_activation(conv_out)

        return activation_out 


        
