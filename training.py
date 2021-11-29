import torch
import torch.optim as optim
from Unet_3D import UNet_3D
from dice_loss import GDL
from torch.utils.data import DataLoader
from Brats_dataset import BratsDataset
from Brats_sampler import BratsSampler
move_to_cuda = True

def train_model(model,optimizer, dataloader, class_weight, nepochs, reg_w):
    train_losses = []
    for e in range(nepochs):
        model.train()
        for batch_i,(batch_patch_image, batch_patch_labels) in enumerate(dataloader):
            bp_image = batch_patch_image.to("cuda")#float64
            bp_labels = batch_patch_labels.to("cuda")#uint8
            bp_pred, activation = model(bp_image)#problem with conv() cuz weights and images
                                                 #are different types, bp_image is float64,
                                                 #does conv function expect double type data?
            print("\n\nbefore loss\n\n")
            loss = GDL()(bp_pred, bp_labels, class_weight)
            #reg_loss =
            #total_loss = loss + reg_w* reg_loss #and change the rest to be according to total loss
            print("batch_index = ", batch_i,
                  "loss = ", loss.item())
                  #"reg = ", reg_loss.item(),
                  #"total_loss = ",total_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_losses.append(loss.item())
    return train_losses

def main():
    model = UNet_3D() 
    #print(model)
    for param in model.parameters():
        param.requires_grad = True
    # move the model to the GPU
    if move_to_cuda:
        model = model.to("cuda")
       
    dataset = BratsDataset(patch_size = 112)
    sampler = BratsSampler(batch_len=2)
    dataloader = DataLoader(dataset = dataset, batch_sampler = sampler, num_workers = 1)
    lr = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=lr)
    class_weight = torch.cuda.ByteTensor([1,1,1]) #find the correct weights
    reg_w = 1e-5
    nepochs = 10
    train_losses= train_model(model, optimizer, dataloader, class_weight, nepochs, reg_w)
    print(train_losses.item())



main()
