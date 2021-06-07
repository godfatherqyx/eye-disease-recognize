import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms as tfs
from torchvision.transforms.transforms import CenterCrop, RandomRotation, RandomVerticalFlip, Resize, TenCrop, ToPILImage, ToTensor
# 使用数据增强
class train_augment:
    def __init__(self,size):
        self.size = size
        self.augment = tfs.Compose([
        tfs.Resize((self.size,self.size)),
        #tfs.TenCrop((self.size,self.size)),
        tfs.ToTensor(),
        tfs.Normalize([0.45485725, 0.29716743, 0.17132092], [0.28256197, 0.20084267, 0.13919644]),
        ])
    def __call__(self,img):
        return self.augment(img)
class val_augment:
    def __init__(self,size):
        self.size = size
        self.augment = tfs.Compose([        
        tfs.Resize((self.size,self.size)),
        #tfs.ColorJitter(brightness=0.2, contrast=0.2, hue=0.2,saturation=0.2),
        #tfs.RandomRotation(90),
        #tfs.CenterCrop((self.size,self.size)),
        tfs.ToTensor(),
        tfs.Normalize([0.45483129, 0.29719642, 0.1718384 ], [0.28157003,0.20086884, 0.14085859]),
        ])
    def __call__(self,img):
        return self.augment(img) 