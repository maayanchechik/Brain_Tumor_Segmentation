import torch
import torch.optim as optim
from Unet_3D import UNet_3D
from dice_loss import GDL
from torch.utils.data import DataLoader
from Brats_dataset import BratsDataset
from Brats_sampler import BratsSampler
from torchvision import transforms
from transformations import random_flip, random_rotate90, random_intensity_scale, random_intensity_shift
import matplotlib.pyplot as plt
import time
move_to_cuda = True

def train_model(model,optimizer, train_dataloader, validation_dataloader, class_weight, nepochs, batch_size, validation_ratio, dataset_sizes):
    train_losses = []
    valid_losses = []
    start_time = time.time()
    
    torch.cuda.empty_cache()
    for e in range(nepochs):
        if e == 25:
            for g in optimizer.param_groups:
                g['lr'] = 0.001
        print("epoch = ",e)
        model.train()
        for batch_i, (batch_patch_image, batch_patch_labels) in enumerate(train_dataloader):
            bp_image = batch_patch_image.to("cuda")
            bp_labels = batch_patch_labels.to("cuda")
            bp_pred = model(bp_image)
            max_pred = torch.max(bp_pred)
            min_pred = torch.min(bp_pred)
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

            if batch_i % validation_ratio == 0:
                #accumulated valid loss throughout the validation_dataloader, then average
                with torch.no_grad(): #this means the gradients will not be calculated,
                    #yet unlike model.eval() dropout layers will still be implimented
                    accum_valid_loss = 0
                    for bp_image_valid, bp_labels_valid in validation_dataloader:
                        bp_image_valid = bp_image_valid.to("cuda")
                        bp_labels_valid = bp_labels_valid.to("cuda")
                        bp_pred_valid = model(bp_image_valid)
                        valid_loss = GDL()(bp_pred_valid, bp_labels_valid, class_weight)
                        accum_valid_loss += valid_loss.item()
                    
                train_losses.append(loss.item())
                print("\n\ntrain_loss = ",loss.item())
                average_valid_loss = accum_valid_loss / dataset_sizes[1]
                valid_losses.append(average_valid_loss)
                print("validation_loss = ",average_valid_loss, "\n\n")
    time_to_train = time.time() - start_time
    print("how much time this training took:",time_to_train)
    return train_losses, valid_losses


def main():
    torch.manual_seed(0)
    model = UNet_3D() 
    for param in model.parameters():
        param.requires_grad = True
    
    # move the model to the GPU
    if move_to_cuda:
        model = model.to("cuda")
    len_dataset = 369
    batch_size = 1 #for space reasons
    transform = transforms.Compose([random_flip(),
                                    random_rotate90(),
                                    random_intensity_scale(),
                                    random_intensity_shift()])
    dataset = BratsDataset(patch_size = 112, len_dataset = len_dataset, transform = transform)
    dataset_sizes = [295,37,37] #about 80% train
    #sampler = BratsSampler(batch_size = batch_size, len_dataset = len_dataset)
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                                    dataset_sizes)
    ### dataloader:
    train_dataloader = DataLoader(dataset = train_dataset,
                                  batch_size=batch_size, shuffle=False)
    validation_dataloader = DataLoader(dataset = validation_dataset,
                                       batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(dataset = test_dataset,
                                 batch_size=batch_size, shuffle=False)
    validation_ratio = 20
    ###[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10 ,1e-11, 1e-12]:
    plt.clf()
    for lr in [1e-3]:
        print("lr ", lr)
        w_decay = 1e-5
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=w_decay)
        class_weight = torch.cuda.FloatTensor([1,1,1,1]) #find the correct weights
        nepochs = 2
        batch_size = 2
        train_losses, valid_losses = train_model(model, optimizer, train_dataloader,
                                                 validation_dataloader, class_weight,
                                                 nepochs, batch_size, validation_ratio,
                                                 dataset_sizes)
        print("train losses", train_losses)
        print("validation losses", valid_losses)
        plt.plot(train_losses, 'b', label = "train_losses")
        plt.plot(valid_losses, 'g', label = "validation_losses")
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        file_path = '/home/mc/Brain_Tumor_Segmentation/loss_figs/dynamic_unet_dataset_whole_dataset_validwithtorchnograd_nepoches'+str(nepochs)+'_batch'+str(batch_size)+'_lr'+ str(lr)+'.png'
        plt.savefig(file_path)
        plt.clf()


main()
