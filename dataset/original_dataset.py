import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import pandas as pd
import torch as t
from torchvision.transforms.transforms import ToPILImage
import glob

class Eye(data.Dataset):
    class_names = ('A','D','G','C','H','M','N')
    def __init__(self ,img_root,tag_root='no_O.csv' ,transform=None ,train=False, test=False,val=False):
        self.img_root = img_root
        self.tag_root = tag_root
        self.train = train
        self.test = test
        self.val = val
        self.transform = transform
        self.imgs = glob.glob(os.path.join(img_root, '*.jpg'))
        self.class_dict = {class_name:i for i,class_name in enumerate(self.class_names)}
        self.names=self._read_csv()
        self.name = []
        for i in range(len(self.imgs)):
            if(self.imgs[i].split('/')[-1] in self.names):
                self.name.append(self.imgs[i].split('/')[-1])
        
    def __getitem__(self,index):
        image_name = self.name[index]
        image = self._read_image(image_name)
        label = self._read_label(image_name)
        if self.transform:
            image = self.transform(image)
        return image,label
    def __len__(self):
        return len(self.name)
    @staticmethod
    def _read_image_names(imgs):
        image_names=[]
        for root in imgs:
            image_names.append(root.split('/')[-1])
        return image_names
    def _read_image(self,image_name):
        image_file = os.path.join(self.img_root,image_name)
        image = Image.open(image_file)
        image = np.array(image)
        image = ToPILImage()(image)
        return image
    def _read_csv(self):
        tag_csv = pd.read_csv(self.tag_root)
        # tag_csv = tag_csv[['labels','filename']]
        # tag_csv = tag_csv[tag_csv['filename']==image_name]
        # label = tag_csv['labels']
        # label = label.tolist()
        # try:
        #   new_key=label[0][1:4]
        #   new_key=self.class_dict[new_key[1:2]]
        #   return new_key
        # except IndexError:
        #   return image_name
        tag_csv = tag_csv[['labels','filename']]
        img_namelist=[]  
        for i in range(len(tag_csv)):
            img_name=tag_csv.iat[i,1]
            img_namelist.append(img_name)
        return img_namelist
    def _read_label(self,image_name):
        tag_csv = pd.read_csv(self.tag_root)
        tag_csv = tag_csv[['labels','filename']]
        tag_csv = tag_csv[tag_csv['filename']==image_name]
        label = tag_csv['labels']
        label = label.tolist() 
        try:
          new_key=label[0][1:4]
          new_key=self.class_dict[new_key[1:2]]
          return new_key
        except IndexError:
          return image_name
