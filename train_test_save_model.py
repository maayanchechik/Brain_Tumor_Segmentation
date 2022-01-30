import torch
import torch.optim as optim
from Unet_3D import UNet
from Res_Unet_3D import Res_UNet
from Vnet_3D import VNet
from torch.utils.data import DataLoader
from Brats_dataset import BratsDataset
from Brats_sampler import BratsSampler
from torchvision import transforms
from transformations import random_flip, random_rotate90, random_intensity_scale, random_intensity_shift
import matplotlib.pyplot as plt
from training import train_model
from testing import test_model
move_to_cuda = True

def main():
    ##########DEFINE DATASET##########
    torch.manual_seed(0)
    len_dataset = 332 #data that is not test
    transform = transforms.Compose([random_flip(),
                                    random_rotate90(),
                                    random_intensity_scale(),
                                    random_intensity_shift()])
    dataset = BratsDataset(patch_size=96, len_dataset=len_dataset, transform=transform,
                           patching='RANDOM_CENTER_TUMOR', is_test=False)
    dataset_sizes = [295,37] #about 80% train and 37 for test
    train_dataset, validation_dataset = torch.utils.data.random_split(dataset, dataset_sizes)
    test_dataset = BratsDataset(patch_size=96, len_dataset=37, transform = None,
                                patching='RANDOM', is_test=True)
    
    ##########DEFINE DATALOADER##########
    batch_size = 1 #for space reasons
    train_dataloader = DataLoader(dataset = train_dataset,
                                  batch_size=batch_size, shuffle=False)
    validation_dataloader = DataLoader(dataset = validation_dataset,
                                       batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(dataset = test_dataset,
                                 batch_size=batch_size, shuffle=False)
    validation_ratio = 40
    ###[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10 ,1e-11, 1e-12]:
    plt.clf()
    for lr in [1e-3]:
        ##########MODEL##########
        model = Res_UNet() 
        for param in model.parameters():
            param.requires_grad = True
        # move the model to the GPU
        if move_to_cuda:
            model = model.to("cuda")
        ##########TRAIN##########
        print("lr ", lr)
        w_decay = 1e-5
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=w_decay)
        #class_weight = torch.cuda.FloatTensor([1,1,1,1]) #find the correct weights
        class_weight = torch.cuda.FloatTensor([0.1,0.35,0.2,0.35])
        nepochs = 1
        batch_size = 2
        train_losses, valid_losses, model = train_model(model, optimizer, train_dataloader,
                                                 validation_dataloader, class_weight,
                                                 nepochs, batch_size, validation_ratio,
                                                 dataset_sizes)

        #######PATHS##########
        folder_path = '/home/mc/Brain_Tumor_Segmentation/model_stats/'
        model_path = 'FirstDebuggSave_ResUNet_lrDecrease_withClassWeights_nepochs'+str(nepochs)+'_batch'+str(batch_size)+'_lr'+ str(lr)


        ###################
        ###TRAINING DATA###
        ###################
           #######VISUALIZE TRAINING  RESULTS##########
        print("train losses", train_losses)
        print("validation losses", valid_losses)
        plt.plot(train_losses, 'b', label = "train_losses")
        plt.plot(valid_losses, 'g', label = "validation_losses")
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        
           #######SAVE FIGS##########
        fig_path = folder_path + 'Training_Fig_' + model_path + '.png'
        plt.savefig(fig_path)
        plt.clf()

        ###################
        #######TEST########
        ###################
        test_dice_losses, test_hausdorff_losses, average_dice_loss, average_hausdorff_loss = test_model(model, test_dataloader, class_weight)
        
           #######VISUALIZE TEST##########
        print("test dice losses", test_dice_losses)
        plt.plot(test_dice_losses, 'b', label = "test_dice_losses")
        plt.plot(test_hausdorff_losses, 'g', label = "test_hausdorff_losses")
        plt.xlabel('images')
        plt.ylabel('loss')
        plt.legend()

           #######SAVE TEST FIGS##########
        fig_path = folder_path + 'Testing_Fig_' + model_path + '.png'
        plt.savefig(fig_path)
        plt.clf()

        ##########SAVE MODEL##########
        model_name = "Model_" + model_path +".model"
        model.eval()
        save_model_path = os.path.join(folder_path, model_name)
        torch.save(model.state_dict(), save_model_path)
        print("saved model to path")

        ##########SAVE LOSS LISTS##########
        train_loss_file_name = folder_path + 'TrainLosses_'+model_path+ ".pkl"
        train_loss_file = open(train_loss_file_name,"wb")
        pickle.dump(train_losses,train_loss_file)
        train_loss_file.close()
        valid_loss_file_name = folder_path + 'ValidLosses_'+model_path+ ".pkl"
        valid_loss_file = open(valid_loss_file_name,"wb")
        pickle.dump(valid_losses,valid_loss_file)
        valid_loss_file.close()
        #test losses
        test_dice_loss_file_name = folder_path + 'TestDiceLosses_'+model_path+ ".pkl"
        test_dice_loss_file = open(test_dice_loss_file_name,"wb")
        pickle.dump(test_dice_losses,test_dice_loss_file)
        test_dice_loss_file.close()
        test_hausdorff_loss_file_name = folder_path + 'TestHausdorffLosses_'+model_path+ ".pkl"
        test_hausdorff_loss_file = open(test_hausdorff_loss_file_name,"wb")
        pickle.dump(test_hausdorff_losses,test_hausdorff_loss_file)
        test_hausdorff_loss_file.close()

if __name__ == "__main__":
    main()
