import torch
from easydict import EasyDict as edict
_C = edict()
cfg = _C
_C.MODEL = edict()
_C.BATCHSIZE=32
_C.LR=0.0001
_C.WEIGHT_DECAY=0.95
_C.LR_STEPS = [0,1,3,5,7,9,12,15,20,25,30,50]
_C.GAMMA = 0.5
_C.IMAGE_SIZE = 512
_C.K = 5
_C.num_classes= 6
_C.IMGROOT='dataset/temp'
_C.VALROOT='dataset/val'
_C.TESTROOT='dataset/test'
_C.OUTPUT_MODEL_DIR = '/mnt/data/qyx_data/torch/saveModel/'
_C.OUTPUT_MODEL_DIR2 = '/mnt/data/qyx_data/torch/saveModel2/'
_C.LOG = '/mnt/data/qyx_data/torch/log/'