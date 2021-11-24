import torch
import torch.nn as nn

class GDL(nn.Module):
    
    #def intersect(self, m1, m2):
        #print((m1*m2).shape)
        #inter = torch.sum(m1*m2,dim=1) 
        #length = m1.shape[0]
        #inter = torch.tensor([0,0,0], dtype=float)
        #for i in range(length):
        #    inter[i] = torch.matmul(m1[i], m2[i])
        #shape should be [3]
        #print(inter.shape)
        #return inter
    
    #This is written for a single image not whole batch with multiple images
    #needs to change
    def forward(self, pred, gt, class_weight):
        #pred_f = torch.permute((1,0,2,3,4))
        pred_f = torch.flatten(pred, start_dim=1)
        gt_f = torch.flatten(gt, start_dim=1)

        #print(pred_f.shape, pred_f.dtype)
        #print(gt_f.shape, gt_f.dtype)
        #print(inter.dtype)
        #inter = self.intersect(pred_f, gt_f)
        
        inter = torch.sum(pred_f*gt_f,dim=1)
        #print("inter",inter.shape)
        numerator = 2.0 * torch.matmul(class_weight, inter)
       
        union = torch.add(pred_f, gt_f)
        union = torch.sum(union, dim=1)
        denominator = torch.matmul(class_weight, union)
       
        loss = 1-(numerator/denominator)
        print(loss)
        return loss
