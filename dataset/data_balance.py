import os
from matplotlib.pyplot import imshow
import numpy as np
import cv2
from skimage import io
import glob
from torchvision import transforms as tfs
import pandas as pd
original_path = 'dataset/temp'
processed_path = 'dataset/balanced_data'
csv_path = '/mnt/data/qyx_data/torch/full_df.csv'
tag_csv = pd.read_csv(csv_path)
tag_csv = tag_csv[['labels','filename']] 
class_names = ('A','D','G','C','H','M','N','O')
class_dict = {class_name:i for i,class_name in enumerate(class_names)}
number_counter=0
img_list=glob.glob(os.path.join(processed_path, '*.jpg'))
csv_dict=dict()
PIL_list=[]
# for root in img_path:
#     real_img_list.append(root.split('/')[-1])
for i in range(len(tag_csv)):
    img_name=tag_csv.iat[i,1]
    label = tag_csv.iat[i,0]
    label = label[1:4]
    label = class_dict[label[1:2]]
    csv_dict[img_name] = label
for root, dirs, files in os.walk(original_path):
    for file in files:
        file_path = os.path.join(root, file)
        if os.path.join(processed_path,file) in img_list:
            continue
        else:
            if file in csv_dict.keys():
                label = csv_dict[file]
                if label == 7:
                    img = cv2.imread(file_path)
                    img = tfs.ToPILImage(img)
                    PIL_list = tfs.TenCrop(img)
