import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import pandas as pd
import torch as t
from torchvision.transforms.transforms import ToPILImage
import glob
from sklearn.preprocessing import label_binarize
class Eye(data.Dataset):
    def __init__(self ,img_root,tag_root ,transform=None):
        self.img_root = img_root
        self.tag_root = tag_root
        self.transform = transform
        fh = open(tag_root,'r',encoding='utf-8')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],int(words[1])))
        self.imgs = imgs
    def __getitem__(self,index):
        image_name,label = self.imgs[index]
        image = Image.open(os.path.join(self.img_root,image_name))
        if self.transform:
            image = self.transform(image)
        # label_list=[]
        # label_list.append(label)
        # label = label_binarize(np.array(label_list),classes=[0,1,2,3,4,5])
        return image,label
    def __len__(self):
        return len(self.imgs)
