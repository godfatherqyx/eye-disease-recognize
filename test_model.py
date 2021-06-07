from dataset.preprocess import train_augment
from dataset.preprocess import val_augment
from torch.utils.data import DataLoader,WeightedRandomSampler
import torch as t
from torch.autograd import Variable as v
from config.default import cfg
import torchnet.meter as meter
from torchvision import models
from utils.plot import plot
import torch.nn as nn
from dataset.dataset import Eye
from cnn_finetune import make_model
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from utils.plot import plot
import os
import argparse
test_root=cfg.IMGROOT
parser = argparse.ArgumentParser(description='Test model')
parser.add_argument('--model',default='resnet18',type=str,help='the test model')
parser.add_argument('--pretrained',default=1, type=int, help='if the model is pretrained')
parser.add_argument('--epoch',default=50,type=int,help='the test epoch')
parser.add_argument('--batchsize',default=64,type=int,help='the test batchsize')
args = parser.parse_args()
if not isinstance(args.pretrained,bool):
    if args.pretrained==1:
        args.pretrained=True
    elif args.pretrained==0:
        args.pretrained=False
n_classes=6
disease_class=['A','D','G','C','H','M']
resume_model='/mnt/data/qyx_data/torch/saveModel2/'+args.model+'/batchsize'+str(args.batchsize)+'_epoch{:2d}.pth'.format(args.epoch)
#resume_model='/mnt/data/qyx_data/torch/saveModel2/'+args.model+'/epoch{:2d}.pth'.format(args.epoch)
test_transform = train_augment(cfg.IMAGE_SIZE)
test_data=Eye(img_root=test_root,tag_root='dataset/test.txt',transform=test_transform)
# data_len=test_data.__len__()
# weight_prob=[data_len/w for w in [1,6,1,1,0.4,0.8]]
# weight_list=[weight_prob[label] for data,label in test_data]
# test_sampler = WeightedRandomSampler(weights=weight_list,num_samples=6*1000,replacement=True)
if os.path.exists('/mnt/data/qyx_data/torch/test_model/'+args.model+'_'+str(args.pretrained)):
    pass
else:
    os.mkdir('/mnt/data/qyx_data/torch/test_model/'+args.model+'_'+str(args.pretrained))
test_dataloader=DataLoader(test_data,batch_size=cfg.BATCHSIZE,drop_last=True,shuffle=True,num_workers=8)
model = make_model(args.model,num_classes=6,pretrained=args.pretrained,input_size = (cfg.IMAGE_SIZE,cfg.IMAGE_SIZE)) 
device = t.device("cuda" if t.cuda.is_available() else "cpu")
model = nn.DataParallel(model)
model = model.to(device)
model.load_state_dict(t.load(resume_model))
model.eval()
confusion_matrix = meter.ConfusionMeter(6)
fpr = dict()
tpr = dict()
roc_auc = dict()
thresholds = dict()
labels = np.zeros((cfg.BATCHSIZE,6))
scores = np.zeros((cfg.BATCHSIZE,6))
for ii, (input,label) in enumerate(test_dataloader):
    with t.no_grad():
        val_input = v(input)
        val_input=val_input.to(device)
        label=label.to(device)
        score = model(val_input)
        confusion_matrix.add(score.data,label.data)
        label = label_binarize(label.cpu().detach().numpy(),classes=[0,1,2,3,4,5])
        score = score.cpu().detach().numpy()
        labels = np.concatenate((labels,label),axis=0)
        scores = np.concatenate((scores,score),axis=0)
for i in range(6):
    fpr[i], tpr[i], _ = roc_curve(labels[:,i],scores[:,i])
    roc_auc[i] = auc(fpr[i],tpr[i]) 
# Compute macro-average ROC curve and ROC area（方法一）
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(6)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(6):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
lw=2
plt.figure()
# plt.plot(fpr["micro"], tpr["micro"],
#          label='micro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["micro"]),
#          color='deeppink', linestyle=':', linewidth=4)
 
plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)
 
colors = cycle(['aqua', 'darkorange', 'cornflowerblue','darkred','blueviolet','coral'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='{0} (area = {1:0.2f})'
             ''.format(disease_class[i], roc_auc[i]))
 
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontsize=15)
plt.ylabel('True Positive Rate',fontsize=15)
plt.title('ROC of model',fontsize=15)
plt.legend(loc="lower right")
plt.savefig('/mnt/data/qyx_data/torch/test_model/'+args.model+'_'+str(args.pretrained)+'/ROC.png')

cm_value = confusion_matrix.value()
plot_util = plot()
print(cm_value)
plot_util.plot_cm_matrix(cm_matrix=confusion_matrix,savepath='/mnt/data/qyx_data/torch/test_model/'+args.model+'_'+str(args.pretrained)+'/cm_matrix.png')
correct = 0
sensitive=dict()
specificity=dict()
for i in range(6):
    correct+=cm_value[i][i]
    sensitive[i] = cm_value[i][i]/cm_value.sum(axis=1)[i]
    specificity[i] = (cm_value.sum(axis=1)[i]-cm_value[i][i])/((cm_value.sum(axis=0)[i]-cm_value[i][i])+(cm_value.sum(axis=1)[i]-cm_value[i][i]))
accuracy = 100.*(correct)/(cm_value.sum())
print('accuracy={}'.format(accuracy))
print(sensitive)
print(specificity)

