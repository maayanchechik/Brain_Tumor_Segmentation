import torch
import torch.nn as nn

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


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

    def forward_simple(self, pred, gt, class_weight):
        pred_f = torch.flatten(pred, start_dim=2)
        gt_f = torch.flatten(gt, start_dim=2)
        print("gt_f sum =", torch.sum(gt_f, dim=2),
              "pred_f_sum = ", torch.sum(pred_f, dim=2),
              pred_f.shape)
        loss = torch.sum(pred)
        print("simple loss = ", loss.cpu().detach().numpy())
        return loss

        
    #This is written for a whole batch
    def forward(self, pred, gt, class_weight):
        
        batch_size = pred.shape[0]
        #print("batch_size ",batch_size)
        pred_f = torch.flatten(pred, start_dim=2)
        gt_f = torch.flatten(gt, start_dim=2)

        #print("pred_f.max", torch.max(gt_f))
        #print("pred_f.min", torch.min(gt_f))
        #print("pred_f", pred_f.shape, pred_f.dtype)
        #print("gt_f", gt_f.shape, gt_f.dtype)
        #print("p_f*g_f",(pred_f*gt_f).shape, (pred_f*gt_f).dtype)
        ##print(inter.dtype)
        ##inter = self.intersect(pred_f, gt_f)
        
        inter = torch.sum(pred_f*gt_f,dim=2)
        print("gt_f sum =", torch.sum(gt_f, dim=2),
              "pred_f_sum = ", torch.sum(pred_f, dim=2),
              #"pred_f = ", pred_f.shape, pred_f.dtype,
              "intersection = ", inter, inter.dtype)
        #print("class_weight.shape", class_weight.shape, class_weight.dtype)
        numerator = 2.0 * torch.matmul(inter, class_weight)
       
        union = torch.add(pred_f, gt_f)
        union = torch.sum(union, dim=2)
        denominator = torch.matmul(union, class_weight)
        ratio = numerator/denominator.clamp(min=1e-6)
        loss = batch_size - torch.sum(numerator/denominator.clamp(min=1e-6))

        print("numerator =", numerator.cpu().detach().numpy(), 
              "union", union.cpu().detach().numpy(),
              "denominator = ", denominator.cpu().detach().numpy(),
              "ratio = ", ratio.cpu().detach().numpy(),
              "loss = ", loss.cpu().detach().numpy())

        return loss
