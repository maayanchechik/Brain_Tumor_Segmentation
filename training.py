import torch
import torch.optim as optim
from Unet_3D import UNet_3D
from dice_loss import GDL
from torch.utils.data import DataLoader
from Brats_dataset import BratsDataset
from Brats_sampler import BratsSampler
import matplotlib.pyplot as plt
move_to_cuda = True

def train_model(model,optimizer, dataloader, class_weight, nepochs, batch_size):
    train_losses = []
    torch.cuda.empty_cache()
    for e in range(nepochs):
        if e == 25:
            for g in optimizer.param_groups:
                g['lr'] = 0.001
        print("epoch = ",e)
        model.train()
        for batch_i, (batch_patch_image, batch_patch_labels) in enumerate(dataloader):
            print("bp_labels.shape = ",batch_patch_labels.shape)
            bp_image = batch_patch_image.to("cuda")
            bp_labels = batch_patch_labels.to("cuda")
            bp_pred = model(bp_image)
            max_pred = torch.max(bp_pred)
            min_pred = torch.min(bp_pred)
            print("max_pred= ", max_pred, "min_pred= ", min_pred)
            loss = GDL()(bp_pred, bp_labels, class_weight)
            print("\nbatch_index = ", batch_i,
                  "loss before step = ", loss.item(),"\n")
            
            loss.backward()
            if batch_size == 1:
                optimizer.step()
                optimizer.zero_grad()
            elif batch_i%batch_size == 1:
                optimizer.step()
                optimizer.zero_grad()
            
            
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
    ###[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10 ,1e-11, 1e-12]:
    plt.clf()
    for lr in [1e-3]:
        print("lr ", lr)
        w_decay = 1e-5
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=w_decay)
        class_weight = torch.cuda.FloatTensor([1,1,1,1]) #find the correct weights
        nepochs = 10
        batch_size = 2
        train_losses= train_model(model, optimizer,
                                  dataloader, class_weight,
                                  nepochs, batch_size)
        print("train losses", train_losses)
        plt.plot(train_losses, label = "train_losses")
        plt.xlabel('epoch')
        plt.legend()
        file_path = '/home/mc/Brain_Tumor_Segmentation/loss_figs/softmax_nepoches'+str(nepochs)+'_batch'+str(batch_size)+'_lr'+ str(lr)+'.png'
        plt.savefig(file_path)
        plt.clf()


main()
