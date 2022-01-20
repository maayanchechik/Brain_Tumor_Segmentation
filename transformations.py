import numpy as np
import torch
class random_flip(object):
  def __init__(self, p=0.5):
    super().__init__()
    self.p = p
  def __call__(self, image_labels_tuple):
    random = torch.rand(1)
    image, labels = image_labels_tuple
    if random.item() < self.p:
      image = np.flip(image,(1,2,3))
      labels = np.flip(labels,(1,2,3))#why does the paper do (0,1,2)?
    return (image, labels)

class random_rotate90(object):
  def __init__(self, p=0.5):
    super().__init__()
    self.p = p
    self.num_rot = (1,2,3)
    self.axes_rot = (1,2,3)
  def rotate90(self, sample):
    axes_rot = np.random.choice(self.axes_rot, size=2, replace=False)
    axes_rot.sort()
    num_rot = np.random.choice(self.num_rot)
    image = np.rot90(sample, num_rot, axes=axes_rot)
    return sample
  def __call__(self, image_labels_tuple):
    random = torch.rand(1)
    image, labels = image_labels_tuple
    if random.item() < self.p:
      image = self.rotate90(image)
      labels = self.rotate90(labels)
    return (image, labels)

class random_intensity_scale(object):
  def __init__(self, min=0.9, max=1.1):
    super().__init__()
    self.min = min
    self.max = max
  def __call__(self, image_labels_tuple):
    #random = torch.rand(1)
    image, labels = image_labels_tuple
    scale = np.random.uniform(self.min,self.max)
    image = image*scale
    return (image, labels)

class random_intensity_shift(object):
  def __init__(self, min=-0.1, max=0.1):
    super().__init__()
    self.min = min
    self.max = max
  def __call__(self, image_labels_tuple):
    image, labels = image_labels_tuple
    scale = np.random.uniform(self.min,self.max)
    image_np = image.numpy()
    std = np.std(image_np[image_np>0])
    image = image + (scale*std)
    return (image, labels)
