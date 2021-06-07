import csv
import os
import glob
def get_k_fold_data(k, k1, image_dir):
    # 返回第i折交叉验证时所需要的训练和验证数据
    assert k > 1
    # if k1==0:#第一次需要打开文件
    file = open(image_dir, 'r', encoding='utf-8')
    reader=csv.reader(file)
    imgs_ls = []
    for line in file.readlines():
        # if len(line):
        imgs_ls.append(line)
    file.close()
    #print(len(imgs_ls))
    avg = len(imgs_ls) // k
    #print(avg)
    f1 = open('dataset/train_k.txt', 'w',encoding="utf-8")
    f2 = open('dataset/val_k.txt', 'w', encoding="utf-8")
    # writer1 = csv.writer(f1)
    # writer2 = csv.writer(f2)
    for i, row in enumerate(imgs_ls):
        if (i // avg) == k1:
            f2.write(row)
        else:
            f1.write(row)
    f1.close()
    f2.close()
get_k_fold_data(10,9,'dataset/all_shuffle_datas.txt')