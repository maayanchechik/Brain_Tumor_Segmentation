import torch
from dice_loss import GDL
import time
from statistics import mean

def test_model(model, test_dataloader, class_weight):
    model.eval()
    test_losses = []
    for batch_i, (batch_patch_image, batch_patch_labels) in enumerate(test_dataloader):
        bp_image = batch_patch_image.to("cuda")
        bp_labels = batch_patch_labels.to("cuda")
        bp_pred = model(bp_image)
        loss = GDL()(bp_pred, bp_labels, class_weight)
        test_losses.append(loss.item())
    return test_losses, mean(train_losses)
