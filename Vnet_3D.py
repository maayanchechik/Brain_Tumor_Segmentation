import torch
import torch.nn as nn
def first_conv(in_channels, out_channels, kernel_size=(5,5,5), padding=1):
  return nn.Sequential(
    nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
    nn.InstanceNorm3d(out_channels)
  )
  
class first_conv_block(nn.Module):
  def __init__(in_channels, out_channels):
    self.first_conv(in_channels, out_channels)
  def forward(x):
    res = x
    x = first_conv(x)
    x = res + x
    return x

def conv_block(in_channels, out_channels, kernel_size=(5,5,5), padding=2):
  return nn.Sequential(
    nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
    nn.InstanceNorm3d(out_channels),
    nn.PReLU()
  )
  
#this is not for thefirst conv
def create_convs(num_convs, in_channels, out_channels):
  modules = nn.ModuleList([])
  modules.append(conv_block(in_channels, out_channels))
  for i in range(1, num_convs):
    module.append(conv_block(out_channels, out_channels))
  return modules

class encoder_block(nn.Module):
  def __init__(self,num_convs, in_channels, out_channels):
    self.conv = nn.Sequential(create_convs(num_convs, in_channels, out_channels))
    self.down = nn.Sequential(
      nn.Conv3d(in_channels, out_channels, kernel_size=(2,2,2), stride=2),
      nn.InstanceNorm3d(out_channels),
      nn.PReLU(),
      nn.Dropout3d()
    )
    self.prelu = nn.PReLU()
  def forward(self,x):
    x = down(x)
    res = x
    x = conv(x)
    #add
    x = self.prelu(x)
    res = self.prelu(res)
    x = res + x
    return x

class decoder_block(nn.Module):
  def __init__(self, num_convs, in_channels, out_channels):
    self.up = nn.Sequential(
      nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(2,2,2), stride=2),
      nn.InstanceNorm3d(out_channels),
      nn.PReLU(),
      nn.Dropout3d()
    )
    self.conv = nn.Sequential(create_convs(num_convs, in_channels+out_channels, out_channels))
    self.prelu = nn.PReLU()
  def forward(self,x, encoder_fmap):
    x = up(x)
    res = x
    x = torch.cat((encoder_fmap, y), dim=1)
    x = conv(x)
    #add
    x = self.prelu(x)
    res = self.prelu(res)
    x = res + x
    return x

class last_conv_block(nn.Module):
  def __init__(self,in_channels, out_channels):
    self.conv = conv_block(in_channels, out_channels)
    self.up = nn.Sequential(
      nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(2,2,2) stride=2),
      nn.InstanceNorm3d(out_channels),
      nn.PReLU(),
      nn.Dropout3d()
    )
    self.final = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=(1,1,1)),
                               nn.Softmax(dim=1))
  def forward(self, x, encoder_fmap):
    x = up(x)
    x = torch.cat((encoder_fmap, y), dim=1)
    x = conv(x)
    x = final(x)
    return x
  
#################################################
########  vnet  #################################
#################################################

class VNet(nn.Module):
  def __init__(self, in_channels=4, out_channels=[32,64,128,256,521], num_convs=[1,2,3,3,4]):
    super(UNet_3D, self).__init__()
    #encoder
    encoder_list = []
    encoder = first_conv_block(in_channels, out_channels[i])
    encoder_list.append(encoder)
    for i in range(1, len(out_channels)):
      encoder = encoder_block(num_convs=num_convs[i], out_channels[i-1], out_channels[i])
      encoder_list.append(encoder)
    self.encoders = nn.ModuleList(encoder_list)
    #decoder
    decoder_list = []
    reversed_channels = reversed_channels[1:]
    reversed_channels = list(reversed(out_channels))#[521,256,128,64]
    reversed_num_convs = reversed_num_convs[1:]
    reversed_num_convs = list(reversed(out_channels))#[3,3,2,1]
    for i in range(len(reversed_channels)-1):
      decoder = decoder_block(num_convs=reversed_num_convs[i], reversed_channels[i], reversed_channels[i+1])
      decoder_list.append(decoder)
    last_decoder = last_conv_block(num_convs[0], out_channels[0], in_channels)
    decoder_list.append(last_decoder)
    self.decoders = nn.ModuleList(decoder_list)
           
  def forward(self, x):
    encoder_fmap = []
    for encoder in self.encoders:
      x = encoder(x)
      encoder_fmap.append(x)
    r_encoder_fmap = list(reversed(encoder_fmap))
    #the last output is of the middle layer that doesnt get a res
    r_encoder_fmap = r_encoder_fmap[1:]
    for i, decoder in enumerate(self.decoders):
      x = decoder(x, r_encoder_fmap[i])
    return x
