import torch
import torch.nn as nn
def conv_block(in_channels, out_channels, kernel_size=(3,3,3), padding=1, num_groups=8):
  return nn.Sequential(
    nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
    nn.GroupNorm(num_groups, out_channels),
    nn.ReLU()#inplace? will this enable more memory?
  )
def double_conv(in_channels, out_channels, kernel_size=(3,3,3), padding=1, num_groups=8):
  return nn.Sequential(
    conv_block(in_channels, out_channels, kernel_size, padding, num_groups),
    conv_block(out_channels, out_channels, kernel_size, padding, num_groups)
  )

class res_block(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=(3,3,3), padding=1, num_groups=8):
    super(res_block, self).__init__()
    self.conv1 = conv_block(in_channels, out_channels, kernel_size, padding, num_groups)
    self.conv2 = double_conv(out_channels, out_channels, kernel_size, padding, num_groups)
    self.relu = nn.ReLU()
  def forward(self, x):
    res = self.conv1(x)
    x = self.conv2(res)
    x = x + res
    x = self.relu(x)
    return x

class encoder_block(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=(3,3,3), padding=1, num_groups=8, pool_kernal=2):
    super(encoder_block, self).__init__()
    self.down = nn.Sequential(nn.MaxPool3d(pool_kernal),
                              nn.Dropout3d())
    self.res_block = res_block(in_channels, out_channels, kernel_size, padding, num_groups)
  def forward(self,x):
    #print("returns x.shape = ",x.shape)
    x = self.down(x)
    #print("receives x.shape = ",x.shape)
    x = self.res_block(x)
    return x

class decoder_block(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=(3,3,3), padding=1, num_groups=8):
    super(decoder_block, self).__init__()
    #conv transposed params are calculated with: out = (in-1)*s -2p +k
    #28 = (14-1)*2 -2*1 +4
    self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
    self.drop_conv = nn.Sequential(nn.Dropout3d(),
                                   res_block(out_channels, out_channels, kernel_size, padding, num_groups))
  def forward(self, x, feature_map):
    #print("feature_map.shape=",feature_map.shape)
    x = self.up(x)
    #print("x_up.shape=",x.shape)
    x = x + feature_map
    x = self.drop_conv(x)
    return x

class Res_UNet(nn.Module):
  def __init__(self, in_channels=4, out_channels=[32,64,128,256], kernel_size=(3,3,3), padding=1, num_groups=8, pool_kernal=2):
    super(Res_UNet, self).__init__()
    encoder_list = []
    for i in range(len(out_channels)):
      if i==0:
        encoder = res_block(in_channels, out_channels[i], kernel_size, padding, num_groups)
      else:
        encoder = encoder_block(out_channels[i-1], out_channels[i], kernel_size, padding, num_groups, pool_kernal)
      encoder_list.append(encoder)
    self.encoders = nn.ModuleList(encoder_list)
    
    decoder_list = []
    reversed_channels = list(reversed(out_channels))
    for i in range(len(reversed_channels)-1):
      decoder = decoder_block(reversed_channels[i], reversed_channels[i+1], kernel_size, padding, num_groups)
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
