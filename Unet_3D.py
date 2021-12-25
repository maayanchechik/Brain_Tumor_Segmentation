import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import interpolate

def conv_block(in_channels, out_channels, kernel_size=(3,3,3), padding=1, num_groups=8):
     return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.GroupNorm(num_groups,out_channels),
        nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.GroupNorm(num_groups,out_channels)
        )
       
def encoder_block(in_cannels, out_channels, kernel_size, padding, num_groups, pool_kernal):
    return nn.Sequential(
        nn.MaxPool3d(pool_kernal),
        nn.Dropout3d(),
        conv_block(in_cannels, out_channels, kernel_size , padding , num_groups)
        )

#decoder cannot be sequential becuase of concat...        
class decoder_block(nn.Module):
    def __init__(self, in_cannels, out_channels, kernel_size, padding, num_groups):
        super(decoder_block, self).__init__()
        self.drop_conv = nn.Sequential(
            nn.Dropout3d(),
            conv_block(in_cannels, out_channels, kernel_size, padding, num_groups)
            )
    def forward(self, x, encoder_fmap):
        y = interpolate(x, scale_factor = 2, mode='nearest')
        y = torch.cat((encoder_fmap, y), dim=1)
        y = self.drop_conv(y)
        return y

class UNet_3D(nn.Module):
    def __init__(self, in_channels=4, out_channels=[32,64,128,256], kernel_size=(3,3,3), padding=1, num_groups=8, pool_kernal=2):
        super(UNet_3D, self).__init__()
        encoder_list = []
        for i in range(len(out_channels)):
            if i==0:
                encoder = conv_block(in_channels, out_channels[i], kernel_size, padding, num_groups)
            else:
                encoder = encoder_block(out_channels[i-1], out_channels[i], kernel_size, padding, num_groups, pool_kernal)
            encoder_list.append(encoder)
        self.encoders = nn.ModuleList(encoder_list)
       
        decoder_list = []
        reversed_channels = list(reversed(out_channels))
        for i in range(len(reversed_channels)-1):
            curr_in_channels = reversed_channels[i]+reversed_channels[i+1]
            decoder = decoder_block(curr_in_channels, reversed_channels[i+1], kernel_size, padding, num_groups)
            decoder_list.append(decoder)
        self.decoders = nn.ModuleList(decoder_list)
       
        self.final = nn.Sequential(
            nn.Conv3d(out_channels[0], in_channels, kernel_size=(1,1,1)),
            nn.Softmax(dim=1)
            )
           
    def forward(self, x):
        encoder_fmap = []
        for encoder in self.encoders:
            x = encoder(x)
            encoder_fmap.append(x)
        r_encoder_fmap = list(reversed(encoder_fmap))
        r_encoder_fmap = r_encoder_fmap[1:]
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, r_encoder_fmap[i])
       
        x = self.final(x)
        return x    
        
