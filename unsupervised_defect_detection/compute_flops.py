
import os,sys
import time
import torch
import torchsummary
from torch import optim
import torch.nn as nn
import timeit
import math
from skimage import measure
from statistics import mean
import pandas as pd
import numpy as np
import cv2
import time
import matplotlib
import copy
from numpy import ndarray
from torchvision.utils import save_image
from torchsummary import summary
from sklearn import metrics
from sklearn.metrics import roc_auc_score,average_precision_score,auc
from sklearn.metrics import precision_recall_curve, auc
import torch.nn.functional as F
import pytorch_ssim
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from thop import profile

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser
# user
from builders.model_detect_builder import build_model
from utils.utils import setup_seed, init_weight, netParams
from utils.metric.metric import get_iou, Metrics
from utils.metric.score import SegmentationMetric
from utils.losses.loss import LovaszSoftmax, CrossEntropyLoss2d, CrossEntropyLoss2dLabelSmooth,\
    ProbOhemCrossEntropy2d, FocalLoss2d, DiceLoss
from utils.optim import RAdam, Ranger, AdamW
from utils.scheduler.lr_scheduler import WarmupPolyLR
from builders.dataset_detect_builder import build_dataset_train
from builders.dataset_detect_builder import build_dataset_test
from scipy.spatial import distance


if __name__ == '__main__':
    model_path = './log/syringe/Finish_P_UNet_SFP_SSIM_0.7Rate_0.05Var_1Flag_1Epoch/best_model.pth'
    small_model_path = './log/syringe/Finish_P_UNet_HFP_SSIM_0.7Rate_0.05Var_1Flag_1Epoch/best_model.pth'
    # model_path = './log/syringe/Finish_P_SegNet_FPGM_SSIM_0.1Rate_0.15Var_1Flag_100Epoch/best_model.pth'
    # small_model_path = './log/syringe/Finish_P_SegNet_HFP_SSIM_0.1Rate_0.15Var_1Flag_100Epoch/best_model.pth'

    model = torch.load(model_path)
    small_model = torch.load(small_model_path)

    input = torch.randn(1, 1, 120, 350)
    input = input.cuda()

    Flops, params = profile(model, inputs=(input,))


    small_Flops, small_params = profile(small_model, inputs=(input,))


    flops_drop = (Flops-small_Flops)/Flops * 100
    parasm_drop = (params - small_params)/params * 100

    print('Params Drop %.2f%% Flops Drop %.2f%%'%(parasm_drop, flops_drop))