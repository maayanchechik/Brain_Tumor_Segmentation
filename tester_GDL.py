import torch
from dice_loss import GDL
#a = torch.ones([4,122,122,122])
#f_a = torch.flatten(a, start_dim=1)
#print(f_a.shape)
#a = torch.tensor([[1,2,3,4],[4,5,6,2],[1,2,3,10]])
#b = torch.tensor([[1,2,3,4],[4,5,6,1],[7,8,9,10]])
#c = torch.add(a,b)
#print(c)
#d = torch.sum(c,dim=1)
#print(d)
#e = a*b
#print(e)

run = True
if(run == True):
    pred = torch.ones([2,3,2,2,2], dtype=float)
    gt = torch.ones([2,3,2,2,2], dtype=float)
    class_weight = torch.tensor([0.2,0.3,0.5],dtype=float)
    loss = GDL()(pred,gt,class_weight)
else:
    bc = torch.ones([2,3], dtype=float)
    c = torch.tensor([4,2,3], dtype=float)
    print("bc = ", bc, bc.shape)
    print("\n c = ", c, c.shape)
    #mul = bc*c
    #print("\nbc * c = ", mul)
    matmul = torch.matmul(bc,c)
    print("\nmatmul(bc,c) = ", matmul)
