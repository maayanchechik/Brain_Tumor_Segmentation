import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

####################################
#Transformations for 2d data #######
####################################
class random_flip(object):
  def __init__(self, p=0.5):
    super().__init__()
    print("init")
    self.p = p
  def __call__(self, image):
    print("in flip:")
    random = torch.rand(1)
    if random.item() < self.p:
      print("#####before- stuck here???")
      image = image.detach().numpy()
      print("#####after-stuck here???")
      print("#####image=numpy=",isinstance(image, np.ndarray))
      image = np.flip(image,(1,2))
      image = torch.from_numpy(image)
      print("#####end if")
    print("end flip")
    return image

class random_rotate90(object):
  def __init__(self, p=0.5):
    super().__init__()
    self.p = p
    self.num_rot = (1,2,3)
    self.axes_rot = (1,2)
  def rotate90(self, sample):
    axes_rot = np.random.choice(self.axes_rot, size=2, replace=False)
    axes_rot.sort()
    num_rot = np.random.choice(self.num_rot)
    image = np.rot90(sample, num_rot, axes=axes_rot)
    return sample
  def __call__(self, image):
    random = torch.rand(1)
    if random.item() < self.p:
      image = self.rotate90(image)
    return image

class random_intensity_scale(object):
  def __init__(self, min=0.9, max=1.1):
    super().__init__()
    self.min = min
    self.max = max
  def __call__(self, image):
    #random = torch.rand(1)
    scale = np.random.uniform(self.min,self.max)
    image = image*scale
    return image

class random_intensity_shift(object):
  def __init__(self, min=-0.1, max=0.1):
    super().__init__()
    self.min = min
    self.max = max
  def __call__(self, image):
    scale = np.random.uniform(self.min,self.max)
    image = image.numpy()
    std = np.std(image[image>0])
    image = image + (scale*std)
    return image

#####################################################

transform = transforms.Compose([transforms.ToTensor(),
                                random_flip()])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#####################################################
import matplotlib.pyplot as plt
import numpy as np

# functions to show an image

print("##########1here#############")
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
