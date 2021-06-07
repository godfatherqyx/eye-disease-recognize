import os
import cv2
import numpy as np
img_path='dataset/train'
index=1
test_index=0
delete_list=[]
path_list=[]
random_list = [i for i in range(7000)]
random_list = np.random.choice(random_list,1000,replace=False)
for root, dirs, files in os.walk(img_path):
    for file in files:
        path_list.append(file)
for i in random_list:
    file_path = os.path.join('dataset/train', path_list[i])
    img=cv2.imread(file_path,1)
    cv2.imwrite(os.path.join('dataset/test/'+path_list[i]),img)
    delete_list.append(file_path)
        # if index%10==0 and test_index<1000:
        #     img=cv2.imread(file_path,1)
        #     cv2.imwrite(os.path.join('dataset/test/'+file),img)
        #     delete_list.append(file_path)
        #     test_index+=1
        #     index+=1
        # else:
        #     index+=1
for file in delete_list:
    os.remove(file)