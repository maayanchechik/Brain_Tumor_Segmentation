from Unet_3D import UNet_3D
import torch

input = torch.rand((1,4,64,64,64))
model = UNet_3D()
conv_out, activation_out = model(input)
print("conv_out", conv_out.shape)
print("activation_out", activation_out.shape)
