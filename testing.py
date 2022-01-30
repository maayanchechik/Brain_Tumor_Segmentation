from medpy import metric
import torch
from dice_loss import GDL
import time
from statistics import mean

def test_model(model, test_dataloader, class_weight):
    model.eval()
    test_dice_losses = []
    test_hausdorff_losses = []
    for batch_i, (batch_patch_image, batch_patch_labels) in enumerate(test_dataloader):
        bp_image = batch_patch_image.to("cuda")
        bp_labels = batch_patch_labels.to("cuda")
        bp_pred = model(bp_image)
        binary_pred = np.argmax(bp_pred, axis=1)
        dice_loss = metric.dc(bp_pred, bp_labels)
        hausdorff_loss = metric.hd95(bp_pred, bp_labels)
        test_dice_losses.append(dice_loss.item())
        test_hausdorff_losses.append(hausdorff_loss.item())
    return test_dice_losses, test_hausdorff_losses, mean(test_dice_losses), mean(test_hausdorff_losses)
