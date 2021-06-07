import sys
import os

from numpy import mod
from dataset.dataset import Eye
from dataset.preprocess import train_augment
from dataset.preprocess import val_augment
from dataset.k_cross import get_k_fold_data
from torch.utils.data import DataLoader,WeightedRandomSampler
import torch as t
from torch.autograd import Variable as v
import logging
import time
import argparse
import datetime
from config.default import cfg
from utils.adjust_learningrate import adjust_learning_rate
import torchnet.meter as meter
from utils.Focalloss import FocalLossV1
from torchvision import models
from utils.plot import plot
from torchsummary import summary
import torch.nn as nn
import matplotlib as plt
from cnn_finetune import make_model
lr=cfg.LR
lr_step=cfg.LR_STEPS

def val(model,dataloder,device):
    model.eval()
    confusion_matrix = meter.ConfusionMeter(7)
    for ii, (input,label) in enumerate(dataloder):
        val_input = v(input)
        val_input=val_input.to(device)
        label=label.to(device)
        score = model(val_input)
        confusion_matrix.add(score.data,label.data)
    model.train()
    cm_value = confusion_matrix.value()
    print(cm_value)
    correct = 0
    for i in range(7):
         correct+=cm_value[i][i]
    accuracy = 100.*(correct)/(cm_value.sum())
    return confusion_matrix, accuracy
def train(train_dataloader,critertion,optimizer,model,device,loss_meter,confusion_matrix):
    train_confusion_matrix = meter.ConfusionMeter(7)
    train_confusion_matrix.reset()
    #scaler = t.cuda.amp.GradScaler()
    for iteration,(data,label) in enumerate(train_dataloader):
        if(isinstance(label,str)):
            continue
        else:
            #训练参数模型
            iteration=iteration+1
            data = t.tensor(data,requires_grad=True)
            input=data.reshape(cfg.BATCHSIZE,3,cfg.IMAGE_SIZE,cfg.IMAGE_SIZE)
            input=input.to(device)
            label=label.to(device)
            optimizer.zero_grad()
            with t.cuda.amp.autocast():
                score = model(input)
                loss=critertion.forward(score,label.long())
            loss.backward()
            optimizer.step()
            #scaler.scale(score).backward()
            #scaler.step(optimizer)
            #scaler.update()
            # score=model(input.float())
            # loss=critertion.forward(score,label.long())
            # loss.backward()
            # optimizer.step()
            
            #更新统计指标
            loss_meter.add(loss.item())
            confusion_matrix.add(score.data,label.data)
            train_confusion_matrix.add(score.data,label.data)
            correct = 0
            for i in range(7):
                correct+=train_confusion_matrix.value()[i][i]
            accuracy = 100.*(correct)/(train_confusion_matrix.value().sum())
            print(confusion_matrix.value())
            class_list=[]
            for i in confusion_matrix.value():
                class_list.append(i.sum())
            print(class_list)
    return model,loss.item(),accuracy
def k_fold(logger,k,root,val_root,epoch,args,critertion,optimizer,k_model,device,loss_meter,confusion_matrix,train_acc,loss_list,val_acc):
    trained_time = 0
    best_accracy = 0
    best_model = None
    avg_accuracy = 0
    avg_loss = 0
    avg_train_acc = 0
    end = time.time()
    for i in range(k):
        get_k_fold_data(k,i,'dataset/all_shuffle_datas.txt')
        train_transform = train_augment(cfg.IMAGE_SIZE)
        train_data=Eye(img_root=root,tag_root='dataset/train_k.txt',transform=train_transform)
        data_len=train_data.__len__()
        weight_prob=[data_len/w for w in [1,6,1,1,0.4,0.8,2.5]]
        weight_list=[weight_prob[label] for data,label in train_data]
        train_sampler = WeightedRandomSampler(weights=weight_list,num_samples=7*2000,replacement=True)
        train_dataloader=DataLoader(train_data,batch_size=cfg.BATCHSIZE,shuffle=(train_sampler==None),drop_last=True,sampler=train_sampler,num_workers=8)
        
        val_transform = val_augment(cfg.IMAGE_SIZE)
        val_data=Eye(img_root=root,tag_root='dataset/val_k.txt',transform=val_transform)
        val_dataloader=DataLoader(val_data,batch_size=cfg.BATCHSIZE,shuffle=False,drop_last=True,num_workers=8)

        k_model[i],train_loss,train_accuracy = train(train_dataloader,critertion,optimizer,k_model[i],device,loss_meter,confusion_matrix)

        val_cm,val_accuracy = val(k_model[i],val_dataloader,device)

        avg_accuracy+=val_accuracy
        avg_train_acc+=train_accuracy
        avg_loss+=train_loss
        trained_time = time.time() - end
        end = time.time()
        log_str = [
            "Epoch:{:02d}, Fold:{:02d}, Lr:{:.8f}, Cost:{:.2f}s".format(epoch,i,
                optimizer.param_groups[0]['lr'], trained_time),
            "Loss:{:.2f}".format(train_loss),
            "train_acc:{:.2f}".format(train_accuracy),
            "val_acc:{:.2f}".format(val_accuracy)
                ]
        logger.info(log_str)
        if val_accuracy>best_accracy:
            best_model = k_model[i]

    avg_accuracy = avg_accuracy/k
    avg_train_acc = avg_train_acc/k
    avg_loss = avg_loss/k

    val_acc.append(avg_accuracy)
    train_acc.append(avg_train_acc)
    loss_list.append(avg_loss)

    log_str = "Epoch:{:2d}".format(epoch)+"--"+"avg_loss:{:2f}".format(avg_loss)+"--"+"avg_train_accuracy:{:2f}".format(avg_train_acc)+"--"+"avg_val_accuracy:{:2f}".format(avg_accuracy)
    logger.info(log_str)
    t.save(best_model.state_dict(),os.path.join(cfg.OUTPUT_MODEL_DIR, args.model+'test{:2d}.pth'.format(epoch)))
    return k_model
def main():
    parser = argparse.ArgumentParser(description='Eye Picture Classification Training With PyTorch')
    parser.add_argument('--resume_model',
                        default='/mnt/data/qyx_data/torch/saveModel/Eye_resnet18_test.pth',
                        type=str,
                        help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--resume',
                        default=None,
                        type=str,
                        help='Checkpoint state_dict file to resume training from')

    parser.add_argument('--model',default='resnet18',type=str,help='the train model')

    parser.add_argument('--start_epoch', default=0, type=int,help='the start epoch of training')
    parser.add_argument('--max_epoch', default=5, type=int, help='the epoch to end training')
    args = parser.parse_args()
    


    root=cfg.IMGROOT
    val_root=cfg.VALROOT
    
    if args.model == 'resnet18':
        model = models.resnet18(pretrained=True)
        model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 7)
    elif args.model == 'vgg16':
        model = make_model('vgg16_bn',num_classes=7,pretrained=True,input_size = (224,224))
    elif args.model =='alexnet':
        model = make_model('alexnet',num_classes=7,pretrained=True,input_size = (224,224)) 
    elif args.model =='inception_v3':
        model = make_model('inception_v3',num_classes=7,pretrained=True,input_size = (224,224))
    elif args.model =='inceptionresnetv2':
        model = make_model('inceptionresnetv2',num_classes=7,pretrained=True,input_size = (224,224)) 
    elif args.model =='googlenet':
        model = make_model('googlenet',num_classes=7,pretrained=True,input_size = (224,224)) 
    elif args.model =='densenet121':
        model = make_model('densenet121',num_classes=7,pretrained=True,input_size = (224,224)) 
    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    logger = logging.getLogger("Eye")
    logger.setLevel(logging.DEBUG)
    fileHanlder = logging.FileHandler('Eye_'+args.model+'.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fileHanlder.setFormatter(formatter)
    logger.addHandler(fileHanlder)
    logger.info('Train Dataset uploaded!')
    logger.info('device:{}'.format(device))
    
    plot_util=plot()

    loss_list=[]
    train_acc=[]
    epoch_list=[]
    val_acc=[]

    critertion=t.nn.CrossEntropyLoss()
    #critertion=FocalLossV1()

    optimizer=t.optim.Adam(model.parameters(),lr=lr, weight_decay=cfg.WEIGHT_DECAY)
    model = nn.DataParallel(model)
    model = model.to(device)
    summary(model,(3,cfg.IMAGE_SIZE,cfg.IMAGE_SIZE),batch_size=cfg.BATCHSIZE)
    # logger.info("Resume from the model {}".format(args.resume_model))
    # model.load_state_dict(t.load(args.resume_model))
    model.train()
    k_model= []
    for i in range(cfg.K):
       k_model.append(model)

    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(7)
    loss_meter.reset()
    confusion_matrix.reset()

    for epoch in range(args.start_epoch,args.max_epoch):
        k_model = k_fold(logger,cfg.K,root,val_root,epoch,args,critertion,optimizer,k_model,device,loss_meter,confusion_matrix,train_acc,loss_list,val_acc)
        epoch_list.append(epoch)
        #调节学习率
        if epoch_list[-1] in lr_step:
            step_index = lr_step.index(epoch_list[-1]) + 1
            adjust_learning_rate(optimizer, cfg.GAMMA, step_index)
    #plot
    plot_util.plot_cm_matrix(confusion_matrix,savepath='/mnt/data/qyx_data/torch/'+args.model+' cm_matrix.png')
    plot_util.plot_accuracy(savepath='/mnt/data/qyx_data/torch/'+args.model+' train_acc.png',epoch=epoch_list,train_accuracy=train_acc,val_accuracy=val_acc)
    plot_util.plot_loss(savepath='/mnt/data/qyx_data/torch/'+args.model+' train_loss.png',iters=epoch_list,loss=loss_list,title='train_loss')
if __name__ == '__main__':
    main()