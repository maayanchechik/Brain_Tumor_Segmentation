import torch
import torch.optim as optim
from dice_loss import GDL
import time

def train_model(model,optimizer, train_dataloader, validation_dataloader, class_weight, nepochs, batch_size, validation_ratio, dataset_sizes):
    train_losses = []
    valid_losses = []
    start_time = time.time()
    last_validation_score = 1
    torch.cuda.empty_cache()
    for e in range(nepochs):
        print("epoch = ",e)
        model.train()
        for batch_i, (batch_patch_image, batch_patch_labels) in enumerate(train_dataloader):
            bp_image = batch_patch_image.to("cuda")
            bp_labels = batch_patch_labels.to("cuda")
            bp_pred = model(bp_image)
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

            if (batch_i % validation_ratio == 0) or (batch_i==294):
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
                if average_valid_loss > last_validation_score:
                    for g in optimizer.param_groups:
                        g['lr'] = g['lr']/5
                last_validation_score = average_valid_loss
    time_to_train = time.time() - start_time
    print("how much time this training took:",time_to_train)
    return train_losses, valid_losses, model

