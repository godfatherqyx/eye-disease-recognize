import os
import random
import shutil
import time
import glob
import pandas as pd
csv_path = '/mnt/data/qyx_data/torch/no_ON.csv'
img_root = 'dataset/test'
class_names = ('A','D','G','C','H','M')
class_dict = {class_name:i for i,class_name in enumerate(class_names)}
img_path = glob.glob(os.path.join(img_root, '*.jpg'))
tag_csv = pd.read_csv(csv_path)
tag_csv = tag_csv[['labels','filename']]
real_img_list=[]
img_namelist=[]
label_list=[]
for root in img_path:
    real_img_list.append(root.split('/')[-1])
for i in range(len(tag_csv)):
    img_name=tag_csv.iat[i,1]
    label = tag_csv.iat[i,0]
    label = label[1:4]
    label = class_dict[label[1:2]]
    img_namelist.append(img_name)
    label_list.append(label)
with open('/mnt/data/qyx_data/torch/dataset/test.txt',"w",encoding="utf-8") as f:
    for i in range(len(img_namelist)):
        if img_namelist[i] in real_img_list:
            f.write(img_namelist[i]+' '+str(label_list[i])+'\n')
f.close()
