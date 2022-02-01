import torch
from Res_Unet_3D import Res_UNet
import os
from Brats_dataset import BratsDataset
from testing import test_model
from torch.utils.data import DataLoader

model = Res_UNet()
folder_path = '/home/mc/Brain_Tumor_Segmentation/model_stats/'
model_name = "Model_FirstDebuggSave_ResUNet_lrDecrease_withClassWeights_nepochs1_batch2_lr0.001.model"
save_model_path = os.path.join(folder_path, model_name)
model.load_state_dict(torch.load(save_model_path))
model.eval()
model.to("cuda")

test_dataset = BratsDataset(patch_size=96, len_dataset=37, transform = None,
                                patching='RANDOM', is_test=True)
test_dataloader = DataLoader(dataset = test_dataset,
                                 batch_size=1, shuffle=False)
class_weight = torch.cuda.FloatTensor([0.1,0.35,0.2,0.35])
test_dice_losses, test_hausdorff_losses, average_dice_loss, average_hausdorff_loss = test_model(model, test_dataloader, class_weight)
