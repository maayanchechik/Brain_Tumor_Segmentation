import torch
import torch.optim as optim
from Unet_3D import UNet_3D
from dice_loss import GDL
from torch.utils.data import DataLoader
from Brats_dataset import BratsDataset
from Brats_sampler import BratsSampler
import matplotlib.pyplot as plt
move_to_cuda = True

def train_model(model,optimizer, dataloader, class_weight, nepochs, reg_w):
    train_losses = []
    for e in range(nepochs):
        print("epoch = ",e)
        model.train()
        for batch_i, (batch_patch_image, batch_patch_labels) in enumerate(dataloader):
            optimizer.zero_grad()
            
            # bp_image = batch_patch_image.to("cuda")#float64
            # #print("bp_image", bp_image.type())
            # #print("bp_image is float ", isinstance(bp_image[0][0][0][0][0].item(), float))
            # bp_labels = batch_patch_labels.to("cuda")#uint8
            # #       print("bp_image", bp_labels.type())
            # #       print("bp_labels is float " ,
            # #            isinstance(bp_labels[0][0][0][0][0].item(), float))
            # bp_pred, activation = model(bp_image)
            # #       problem with conv() cuz weights and images
            # #       are different types, bp_image is tensor.float64,
            # #       and w is tensor.float32?

            bp_image = batch_patch_image.to("cuda")#float64
            bp_labels = batch_patch_labels.to("cuda")#uint8
            bp_pred, activation = model(bp_image)
            loss = GDL()(bp_pred, bp_labels, class_weight)
            print("batch_index = ", batch_i,
                  "loss before optimizer step = ", loss.item())
            loss.backward()
            optimizer.step()

            bp_pred, activation = model(bp_image)
            loss = GDL()(bp_pred, bp_labels, class_weight)
            print("batch_index = ", batch_i,
                  "loss after optimizer step = ", loss.item())
            
        train_losses.append(loss.item())
    return train_losses


def main():
    torch.manual_seed(0)
    model = UNet_3D() 
    for param in model.parameters():
        param.requires_grad = True
    
    # move the model to the GPU
    if move_to_cuda:
        model = model.to("cuda")
       
    dataset = BratsDataset(patch_size = 112)
    batch_size = 1
    print("batch size =", batch_size)
    ### sampler = BratsSampler(batch_len=batch_size)
    dataloader = DataLoader(dataset = dataset,
                            batch_size=batch_size,
                            num_workers = 2,
                            shuffle=False)
    print("num_batches =", len(dataloader))
    
    lr = 1e-8
    w_decay = 1e-10
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=w_decay)
    class_weight = torch.cuda.FloatTensor([1,1,1]) #find the correct weights
    reg_w = 1e-5
    nepochs = 1
    train_losses= train_model(model, optimizer,
                              dataloader, class_weight,
                              nepochs, reg_w)
    print("train losses", train_losses)
    plt.plot(train_losses, label = "train_losses")
    plt.xlabel('epoch')
    plt.legend()
    plt.show()


main()
