import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt 
import numpy as np
class plot:
    def __init__(self):
        self.classes=('A','D','G','C','H','M')
    def plot_cm_matrix(self,cm_matrix,savepath,title='confusion matrix'):
        plt.figure(figsize=(12,8),dpi=100)
        plt.imshow(cm_matrix.value(),interpolation='nearest',cmap='Oranges')
        plt.title(title,fontsize=20)
        plt.colorbar()
        xlocations=np.array(range(len(self.classes)))
        plt.xticks(xlocations,self.classes,rotation=0,fontsize=20)
        plt.yticks(xlocations,self.classes,fontsize=20)
        plt.ylabel('Actual label',fontsize=20)
        plt.xlabel('Predict label',fontsize=20)
        #text
        for i in range(len(self.classes)):
            for j in range(len(self.classes)):
                text = plt.text(j,i,cm_matrix.value()[i,j],ha='center',va='center',color='k',fontsize=20)
        #offsite the tick
        tick_marks = np.array(range(len(self.classes)))+0.5
        plt.gca().set_xticks(tick_marks, minor=True)
        plt.gca().set_yticks(tick_marks, minor=True)
        plt.gca().xaxis.set_ticks_position('none')
        plt.gca().yaxis.set_ticks_position('none')
        plt.grid(True, which='minor', linestyle='-')
        plt.gcf().subplots_adjust(bottom=0.15)
        #show
        plt.savefig(savepath,format='png')
        plt.close()
    def plot_accuracy(self,savepath,epoch,train_accuracy,val_accuracy):
        #plt.subplot(211)
        plt.plot(epoch,train_accuracy,marker='o',label='train_acc')
        #plt.xlabel('epoch')
        #plt.xticks(epoch)
        #plt.ylabel('train_accuracy')
        #plt.title("train")
        #plt.subplot(212)
        plt.plot(epoch,val_accuracy,marker='o',label='val_acc')
        plt.xlabel('epoch')
        #plt.xticks(epoch)
        plt.ylabel('accuracy')
        plt.title("accuracy")
        plt.legend(loc ="lower right")
        plt.savefig(savepath,format='png')
        plt.close()
    def plot_loss(self,savepath,iters,epoch,loss,val_loss,title):
        plt.subplot(211)
        plt.plot(iters,loss)
        plt.xlabel('iteration')
        #plt.xticks(iters)
        plt.ylabel('train_loss')
        plt.title(title)
        plt.subplot(212)
        plt.plot(epoch,val_loss,marker='o')
        plt.xlabel('epoch')
        plt.ylabel('val_loss')
        plt.tight_layout()
        plt.savefig(savepath,format='png')
        plt.close()
# #cm=np.array([[1,2,3,4,5,6],
#              [2,2,5,3,4,5],
#              [2,2,5,3,4,5],
#              [2,2,5,3,4,5],
#              [2,2,5,3,4,5],
#              [2,2,5,3,4,5]          
#                             ])
# iters=[1,2,3,4,5]
# accuracy=[0.1,0.4,0.8,0.9,1]
# loss=[0.9,0.8,0.7,0.6,0.5]
#plot_util=plot()
# # fig = plt.figure()
# plt.subplot(211)
# plt.plot(iters,accuracy)
# plt.subplot(212)
# plt.plot(iters,loss)
# plt.savefig('/mnt/data/qyx_data/torch/test.png')
# plot_util.plot_cm_matrix(savepath='/mnt/data/qyx_data/torch/cm_matrix.png',cm_matrix=cm)
# plot_util.plot_accuracy(savepath='/mnt/data/qyx_data/torch/accuracy_iter.png',epoch=iters,train_accuracy=accuracy,val_accuracy=loss)
# plt.subplot(212)
# plot_util.plot_loss(savepath='/mnt/data/qyx_data/torch/accuracy_iter.png',iters=iters,loss=loss,title='2')
