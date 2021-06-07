import os
import cv2
import numpy as np
def calculate_mean_and_std(data_path):
    pixels = [0,0,0]
    pixel_num = [0,0,0]
    mean = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            file_path = os.path.join(root, file)
            image = cv2.imread(file_path, 1)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            h, w ,c = image.shape
            color_list = cv2.split(image)
            for i in range(len(color_list)):
              pixels[i] += np.sum(color_list[i][:, :] / 255)
              pixel_num [i] += h * w
    mean = np.array(pixels) / np.array(pixel_num)
    pixels = [0,0,0]
    for root, dirs, files in os.walk(data_path):
        for file in files:
            file_path = os.path.join(root, file)
            image = cv2.imread(file_path, 1)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            color_list = cv2.split(image)
            for i in range(len(color_list)): 
                pixels[i]+= np.sum((color_list[i][:, :] / 255 - mean[i]) ** 2)

    std = np.sqrt(np.array(pixels) / np.array(pixel_num))
    print(f'normMean={mean},\tnormStd={std}.')
    return mean, std
calculate_mean_and_std('dataset/temp')