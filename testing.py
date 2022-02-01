from medpy import metric
import torch
from dice_loss import GDL
import time
from statistics import mean
import numpy as np
from dice_loss import GDL 
def test_model(model, test_dataloader, class_weight):
    model.eval()
    test_dice_scores = []
    test_hausdorff_losses = []
    for batch_i, (batch_patch_image, batch_patch_labels) in enumerate(test_dataloader):
        bp_image = batch_patch_image.to("cuda")
        bp_labels = batch_patch_labels.to("cuda")
        bp_pred = model(bp_image)
        
        my_dice_loss = GDL()(bp_pred, bp_labels, class_weight)
        print("\nmy_dice_loss",my_dice_loss.item())
        
        bp_pred = bp_pred.detach().cpu().clone().numpy()
        bp_labels = bp_labels.detach().cpu().clone().numpy()
        
        binary_pred = np.where(bp_pred==bp_pred.max(axis=1,keepdims=True),1,0)
        first_class_pred = binary_pred[:][0][:][:][:]
        first_class_labels = bp_labels[:][0][:][:][:]
        dice_score = metric.dc(first_class_pred, first_class_labels)
        hausdorff_loss = metric.hd95(first_class_pred, first_class_labels)
        test_dice_scores.append(dice_score)
        print("dice_score",dice_score)
        test_hausdorff_losses.append(hausdorff_loss)
        print("hausdorff_loss",hausdorff_loss)
    return test_dice_scores, test_hausdorff_losses, mean(test_dice_scores), mean(test_hausdorff_losses)
