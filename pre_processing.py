import collections
import os
from matplotlib.pyplot import imshow
import numpy as np
import cv2
from torchvision.transforms.transforms import ToPILImage
from skimage import io
import glob
def corp_margin(path):
        img = cv2.imread(path,1) 
        img2=img.sum(axis=2)
        (row,col)=img2.shape
        row_top=0
        raw_down=0
        col_top=0
        col_down=0
        for r in range(0,row):
                if img2.sum(axis=1)[r]>0:
                        row_top=r
                        break
 
        for r in range(row-1,0,-1):
                if img2.sum(axis=1)[r]>0:
                        raw_down=r
                        break
 
        for c in range(0,col):
                if img2.sum(axis=0)[c]>0:
                        col_top=c
                        break
 
        for c in range(col-1,0,-1):
                if img2.sum(axis=0)[c]>0:
                        col_down=c
                        break
        if row_top<1 or col_top<1:
            new_img=img[row_top:raw_down,col_top:col_down,0:3] 
        else:
            new_img=img[row_top-1:raw_down,col_top-1:col_down,0:3]
        return new_img
def process(img):
  b,g,r = cv2.split(img)
  lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
  lab_planes = cv2.split(lab)
  clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
  lab_planes[0] = clahe.apply(lab_planes[0])
  lab = cv2.merge(lab_planes)
  bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
  return bgr

original_path = 'Training Images'
processed_path = 'dataset/temp'
number_counter=0
img_list=glob.glob(os.path.join(processed_path, '*.jpg'))
for root, dirs, files in os.walk(original_path):
    for file in files:
        file_path = os.path.join(root, file)
        if os.path.join(processed_path,file) in img_list:
                continue
        else:
                #original_image = cv2.imread(file_path, 1)
                img = corp_margin(file_path)
                #cv2.imshow('original',img)
                #cv2.waitKey(0)
                img = process(img)
                img = cv2.resize(img,(1024,1024))
                cv2.imwrite(os.path.join(processed_path,file),img)
                number_counter+=1
                #cv2.imshow('processd',img)
                #cv2.waitKey(0)
                if number_counter%100==0:
                        print(str(number_counter)+"张完成")