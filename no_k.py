import sys
import os
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

def val(args,model,dataloder,device,critertion):
    model.eval()
    confusion_matrix = meter.ConfusionMeter(cfg.num_classes)
    for ii, (input,label) in enumerate(dataloder):
        with t.no_grad():
            val_input = v(input)
            val_input=val_input.to(device)
            #label = label.reshape(args.batchsize,6)
            label=label.to(device)
            score = model(val_input)
            val_loss=critertion.forward(score,label.long())
            confusion_matrix.add(score.data,label.data)
    cm_value = confusion_matrix.value()
    print(cm_value)
    correct = 0
    for i in range(6):
         correct+=cm_value[i][i]
    accuracy = 100.*(correct)/(cm_value.sum())
    return confusion_matrix, accuracy,val_loss.item()
def train(epoch,logger,args,train_dataloader,critertion,optimizer,model,device,loss_meter,confusion_matrix,iter_list,loss_list):
    train_confusion_matrix = meter.ConfusionMeter(cfg.num_classes)
    train_confusion_matrix.reset()
    #scaler = t.cuda.amp.GradScaler()
    model.train()
    for iteration,(data,label) in enumerate(train_dataloader):
        if(isinstance(label,str)):
            continue
        else:
            #训练参数模型
            iter_list.append(len(train_dataloader)*epoch+iteration)
            iteration=iteration+1
            data = t.tensor(data,requires_grad=True)
            input=data.reshape(args.batchsize,3,cfg.IMAGE_SIZE,cfg.IMAGE_SIZE)
            #label = label.reshape(args.batchsize,6)
            input=input.to(device)
            label=label.to(device)
            optimizer.zero_grad()
            with t.cuda.amp.autocast():
                score = model(input)
                loss=critertion.forward(score,label.long())
            loss.backward()
            optimizer.step()
   
            
            #更新统计指标
            loss_list.append(loss.item())
            loss_meter.add(loss.item())
            confusion_matrix.add(score.data,label.data)
            train_confusion_matrix.add(score.data,label.data)
            correct = 0
            for i in range(cfg.num_classes):
                correct+=train_confusion_matrix.value()[i][i]
            accuracy = 100.*(correct)/(train_confusion_matrix.value().sum())
            print(confusion_matrix.value())
            class_list=[]
            for i in confusion_matrix.value():
                class_list.append(i.sum())
            print(class_list)
            
            #log
            if iteration%args.log_step==0:
                log_str = [
                            "Epoch:{:02d}, Iter:{:3d}, Lr:{:.8f}".format(epoch,iteration,
                                optimizer.param_groups[0]['lr']),
                            "Loss:{:.4f}".format(loss.item()),
                            "train_acc:{:.2f}".format(accuracy),
                                ]
                logger.info(log_str)
    return model,loss.item(),accuracy
def epoch_train(train_dataloader,val_dataloader,logger,epoch,args,critertion,optimizer,model,device,loss_meter,confusion_matrix,train_acc,loss_list,val_acc,iter_list,val_losslist):
    trained_time = 0
    end = time.time()

    model,train_loss,train_accuracy = train(epoch,logger,args,train_dataloader,critertion,optimizer,model,device,loss_meter,confusion_matrix,iter_list,loss_list)

    val_cm,val_accuracy,val_loss = val(args,model,val_dataloader,device,critertion)

  
    trained_time = time.time() - end
    end = time.time()
    log_str = [
        "Epoch:{:02d}, Lr:{:.8f}, Cost:{:.2f}s".format(epoch,
            optimizer.param_groups[0]['lr'], trained_time),
        "Loss:{:.4f}".format(train_loss),
        "train_acc:{:.2f}".format(train_accuracy),
        "val_acc:{:.2f}".format(val_accuracy),
        "val_loss:{:.2f}".format(val_loss)
            ]
    logger.info(log_str)

    val_acc.append(val_accuracy)
    train_acc.append(train_accuracy)
    val_losslist.append(val_loss)
    
    if args.pretrained==True:
        if os.path.exists(os.path.join(cfg.OUTPUT_MODEL_DIR2,args.model)):
            t.save(model.state_dict(),os.path.join(cfg.OUTPUT_MODEL_DIR,args.model)+"/batchsize{:2d}_".format(args.batchsize)+"epoch{:2d}.pth".format(epoch))
        else:
            os.mkdir(os.path.join(cfg.OUTPUT_MODEL_DIR2,args.model))
            t.save(model.state_dict(),os.path.join(cfg.OUTPUT_MODEL_DIR,args.model)+"/batchsize{:2d}_".format(args.batchsize)+"epoch{:2d}.pth".format(epoch)) 
    else:
        if os.path.exists(os.path.join(cfg.OUTPUT_MODEL_DIR,args.model)):
            t.save(model.state_dict(),os.path.join(cfg.OUTPUT_MODEL_DIR2,args.model)+"/batchsize{:2d}_".format(args.batchsize)+"epoch{:2d}.pth".format(epoch))
        else:
            os.mkdir(os.path.join(cfg.OUTPUT_MODEL_DIR,args.model))
            t.save(model.state_dict(),os.path.join(cfg.OUTPUT_MODEL_DIR2,args.model)+"/batchsize{:2d}_".format(args.batchsize)+"epoch{:2d}.pth".format(epoch))
    return model
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
    parser.add_argument('--max_epoch', default=1, type=int, help='the epoch to end training')
    parser.add_argument('--log_step',default=20, type=int, help='log_step')
    parser.add_argument('--batchsize',default=128, type=int, help='batchsize')
    parser.add_argument('--pretrained',default=1, type=int, help='if the model is pretrained')
    args = parser.parse_args()
    if not isinstance(args.pretrained,bool):
        if args.pretrained==1:
            args.pretrained=True
        elif args.pretrained==0:
            args.pretrained=False
    root=cfg.IMGROOT
    val_root=cfg.VALROOT
    
    if args.model == 'resnet18':
        model = make_model('resnet18',num_classes=cfg.num_classes,pretrained=args.pretrained,input_size = (cfg.IMAGE_SIZE,cfg.IMAGE_SIZE))
    if args.model == 'resnet101':
        model = make_model('resnet101',num_classes=cfg.num_classes,pretrained=args.pretrained,input_size = (cfg.IMAGE_SIZE,cfg.IMAGE_SIZE))
    elif args.model == 'vgg16':
        model = make_model('vgg16',num_classes=cfg.num_classes,pretrained=args.pretrained,input_size = (cfg.IMAGE_SIZE,cfg.IMAGE_SIZE))
    elif args.model =='alexnet':
        model = make_model('alexnet',num_classes=cfg.num_classes,pretrained=args.pretrained,input_size = (cfg.IMAGE_SIZE,cfg.IMAGE_SIZE)) 
    elif args.model =='inception_v3':
        model = make_model('inception_v3',num_classes=cfg.num_classes,pretrained=args.pretrained,input_size = (cfg.IMAGE_SIZE,cfg.IMAGE_SIZE))
    elif args.model =='inceptionresnetv2':
        model = make_model('inceptionresnetv2',num_classes=cfg.num_classes,pretrained=args.pretrained,input_size = (cfg.IMAGE_SIZE,cfg.IMAGE_SIZE)) 
    elif args.model =='googlenet':
        model = make_model('googlenet',num_classes=cfg.num_classes,pretrained=args.pretrained,input_size = (cfg.IMAGE_SIZE,cfg.IMAGE_SIZE)) 
    elif args.model =='densenet121':
        model = make_model('densenet121',num_classes=cfg.num_classes,pretrained=args.pretrained,input_size = (cfg.IMAGE_SIZE,cfg.IMAGE_SIZE)) 

    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    logger = logging.getLogger("Eye")
    logger.setLevel(logging.DEBUG)
    fileHanlder = logging.FileHandler(cfg.LOG+time.asctime( time.localtime(time.time()))+'Eye_'+args.model+'.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fileHanlder.setFormatter(formatter)
    logger.addHandler(fileHanlder)
    logger.info('Train Dataset uploaded!')
    logger.info('device:{}'.format(device))

    train_transform = train_augment(cfg.IMAGE_SIZE)
    train_data=Eye(img_root=root,tag_root='dataset/train.txt',transform=train_transform)
    data_len=train_data.__len__()
    weight_prob=[data_len/w for w in [1,6,1,1,0.4,0.8]]
    weight_list=[weight_prob[label] for data,label in train_data]
    train_sampler = WeightedRandomSampler(weights=weight_list,num_samples=6*3000,replacement=True)
    train_dataloader=DataLoader(train_data,batch_size=args.batchsize,drop_last=True,num_workers=8,sampler=train_sampler)
    
    val_transform = val_augment(cfg.IMAGE_SIZE)
    val_data=Eye(img_root=root,tag_root='dataset/test.txt',transform=val_transform)
    val_dataloader=DataLoader(val_data,batch_size=args.batchsize,shuffle=False,drop_last=True,num_workers=8)
    
    plot_util=plot()

    loss_list=[]
    train_acc=[]
    epoch_list=[]
    val_acc=[]
    iter_list=[]
    val_losslist=[]

    critertion=t.nn.CrossEntropyLoss()
    #critertion=FocalLossV1()

    optimizer=t.optim.Adam(model.parameters(),lr=lr, weight_decay=cfg.WEIGHT_DECAY)
    model = nn.DataParallel(model)
    model = model.to(device)
    #summary(model,(3,cfg.IMAGE_SIZE,cfg.IMAGE_SIZE),batch_size=cfg.BATCHSIZE)
    # logger.info("Resume from the model {}".format(args.resume_model))
    # model.load_state_dict(t.load(args.resume_model))
    #model.train()

    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(cfg.num_classes)
    loss_meter.reset()
    confusion_matrix.reset()

    for epoch in range(args.start_epoch,args.max_epoch):
        model = epoch_train(train_dataloader,val_dataloader,logger,epoch,args,critertion,optimizer,model,device,loss_meter,confusion_matrix,train_acc,loss_list,val_acc,iter_list,val_losslist)
        epoch_list.append(epoch)
        #调节学习率
        if epoch_list[-1] in lr_step:
            step_index = lr_step.index(epoch_list[-1]) + 1
            adjust_learning_rate(optimizer, cfg.GAMMA, step_index)
    #plot
    if os.path.exists('/mnt/data/qyx_data/torch/log/'+args.model+'_'+str(args.pretrained)):
        plot_util.plot_cm_matrix(confusion_matrix,savepath='/mnt/data/qyx_data/torch/log/'+args.model+'_'+str(args.pretrained)+'/cm_matrix.png')
        plot_util.plot_accuracy(savepath='/mnt/data/qyx_data/torch/log/'+args.model+'_'+str(args.pretrained)+'/train_acc.png',epoch=epoch_list,train_accuracy=train_acc,val_accuracy=val_acc)
        plot_util.plot_loss(savepath='/mnt/data/qyx_data/torch/log/'+args.model+'_'+str(args.pretrained)+'/loss.png',iters=iter_list,epoch=epoch_list,loss=loss_list,val_loss=val_losslist,title='loss')
    else:
        os.mkdir('/mnt/data/qyx_data/torch/log/'+args.model+'_'+str(args.pretrained))
        plot_util.plot_cm_matrix(confusion_matrix,savepath='/mnt/data/qyx_data/torch/log/'+args.model+'_'+str(args.pretrained)+'/cm_matrix.png')
        plot_util.plot_accuracy(savepath='/mnt/data/qyx_data/torch/log/'+args.model+'_'+str(args.pretrained)+'/train_acc.png',epoch=epoch_list,train_accuracy=train_acc,val_accuracy=val_acc)
        plot_util.plot_loss(savepath='/mnt/data/qyx_data/torch/log/'+args.model+'_'+str(args.pretrained)+'/train_loss.png',iters=iter_list,epoch=epoch_list,loss=loss_list,val_loss=val_losslist,title='loss')
if __name__ == '__main__':
    main()