import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
import numpy as np
from PIL import Image
import torch.utils.data as Data
import os
import time
import argparse
from torch.nn import init
import math


__all__ = ["CAE"]
class CAE(nn.Module):
    def __init__(self,input_nbr,label_nbr):
        super(CAE, self).__init__()

        batchNorm_momentum = 0.1
        self.weights_new = self.state_dict()


        self.conv11 = nn.Conv2d(input_nbr, 64, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.re11 = nn.ReLU()
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.re12 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2, return_indices=True)

        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        self.re21 = nn.ReLU()
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        self.re22 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2, return_indices=True)

        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.re31 = nn.ReLU()
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.re32 = nn.ReLU()
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.re33 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2, return_indices=True)

        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.re41 = nn.ReLU()
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.re42 = nn.ReLU()
        self.conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.re43 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2, return_indices=True)

        self.conv51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.re51 = nn.ReLU()
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.re52 = nn.ReLU()
        self.conv53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53 = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.re53 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(2, 2, return_indices=True)


        self.pool5d = nn.MaxUnpool2d(2, 2)
        self.conv53d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.re53d = nn.ReLU()
        self.conv52d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.re52d = nn.ReLU()
        self.conv51d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.re51d = nn.ReLU()


        self.pool4d = nn.MaxUnpool2d(2, 2)
        self.conv43d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.re43d = nn.ReLU()
        self.conv42d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(512, momentum= batchNorm_momentum)
        self.re42d = nn.ReLU()
        self.conv41d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.re41d = nn.ReLU()

        self.pool3d = nn.MaxUnpool2d(2, 2)
        self.conv33d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.re33d = nn.ReLU()
        self.conv32d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(256, momentum= batchNorm_momentum)
        self.re32d = nn.ReLU()
        self.conv31d = nn.Conv2d(256,  128, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        self.re31d = nn.ReLU()

        self.pool2d = nn.MaxUnpool2d(2, 2)
        self.conv22d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(128, momentum= batchNorm_momentum)
        self.re22d = nn.ReLU()
        self.conv21d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.re21d = nn.ReLU()

        self.pool1d = nn.MaxUnpool2d(2, 2)
        self.conv12d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(64, momentum= batchNorm_momentum)
        self.re12d = nn.ReLU()
        self.conv11d = nn.Conv2d(64, label_nbr, kernel_size=3, padding=1)
        self.bn11d = nn.BatchNorm2d(label_nbr, momentum=batchNorm_momentum)
        self.re11d = nn.ReLU()


    def forward(self, x):
        # Stage 1
        x11 = self.re11(self.bn11(self.conv11(x)))
        x12 = self.re12(self.bn12(self.conv12(x11)))
        x1p, id1 = self.pool1(x12)


        # Stage 2
        x21 = self.re21(self.bn21(self.conv21(x1p)))
        x22 = self.re22(self.bn22(self.conv22(x21)))
        x2p, id2 = self.pool1(x22)

        # Stage 3
        x31 = self.re31(self.bn31(self.conv31(x2p)))
        x32 = self.re32(self.bn32(self.conv32(x31)))
        x33 = self.re33(self.bn33(self.conv33(x32)))
        x3p, id3 = self.pool3(x33)

        # Stage 4
        x41 = self.re41(self.bn41(self.conv41(x3p)))
        x42 = self.re42(self.bn42(self.conv42(x41)))
        x43 = self.re43(self.bn43(self.conv43(x42)))
        x4p, id4 = self.pool4(x43)

        # Stage 5
        x51 = self.re51(self.bn51(self.conv51(x4p)))
        x52 = self.re52(self.bn52(self.conv52(x51)))
        x53 = self.re53(self.bn53(self.conv53(x52)))
        x5p, id5 = self.pool5(x53)


        # Stage 5d
        x5d = self.pool5d(x5p, id5, x53.size())
        x53d = self.re53d(self.bn53d(self.conv53d(x5d)))
        x52d = self.re52d(self.bn52d(self.conv52d(x53d)))
        x51d = self.re51d(self.bn51d(self.conv51d(x52d)))

        # Stage 4d
        x4d = self.pool4d(x51d, id4, x43.size())
        x43d = self.re43d(self.bn43d(self.conv43d(x4d)))
        x42d = self.re42d(self.bn42d(self.conv42d(x43d)))
        x41d = self.re41d(self.bn41d(self.conv41d(x42d)))

        # Stage 3d
        x3d = self.pool3d(x41d, id3, x33.size())
        x33d = self.re33d(self.bn33d(self.conv33d(x3d)))
        x32d = self.re32d(self.bn32d(self.conv32d(x33d)))
        x31d = self.re31d(self.bn31d(self.conv31d(x32d)))

        # Stage 2d
        x2d = self.pool2d(x31d, id2, x22.size())
        x22d = self.re22d(self.bn22d(self.conv22d(x2d)))
        x21d = self.re21d(self.bn21d(self.conv21d(x22d)))

        # Stage 1d
        x1d = self.pool1d(x21d, id1, x12.size())
        x12d = self.re12d(self.bn12d(self.conv12d(x1d)))
        x11d = self.re11d(self.bn11d(self.conv11d(x12d)))

        return x11d

    def replace_layer(self, index, layer):

        if index == 0:
            self.conv11 = layer
        elif index == 1:
            self.bn11 = layer
        elif index == 2:
            self.re11 = layer
        elif index == 3:
            self.conv12 = layer
        elif index == 4:
            self.bn12 = layer
        elif index == 5:
            self.re12 = layer
        elif index == 6:
            self.pool1 = layer

        elif index == 7:
            self.conv21 = layer
        elif index == 8:
            self.bn21 = layer
        elif index == 9:
            self.re21 = layer
        elif index == 10:
            self.conv22 = layer
        elif index == 11:
            self.bn22 = layer
        elif index == 12:
            self.re22 = layer
        elif index == 13:
            self.pool2 = layer
        elif index == 14:
            self.conv31 = layer
        elif index == 15:
            self.bn31 = layer
        elif index == 16:
            self.re31 = layer
        elif index == 17:
            self.conv32 = layer
        elif index == 18:
            self.bn32 = layer
        elif index == 19:
            self.re32 = layer
        elif index == 20:
            self.conv33 = layer
        elif index == 21:
            self.bn33 = layer
        elif index == 22:
            self.re33 = layer
        elif index == 23:
            self.pool3 = layer

        elif index == 24:
            self.conv41 = layer
        elif index == 25:
            self.bn41 = layer
        elif index == 26:
            self.re41 = layer
        elif index == 27:
            self.conv42 = layer
        elif index == 28:
            self.bn42 = layer
        elif index == 29:
            self.re42 = layer
        elif index == 30:
            self.conv43 = layer
        elif index == 31:
            self.bn43 = layer
        elif index == 32:
            self.re43 = layer
        elif index == 33:
            self.pool4 = layer

        elif index == 34:
            self.conv51 = layer
        elif index == 35:
            self.bn51 = layer
        elif index == 36:
            self.re51 = layer
        elif index == 37:
            self.conv52 = layer
        elif index == 38:
            self.bn52 = layer
        elif index == 39:
            self.re52 = layer
        elif index == 40:
            self.conv53 = layer
        elif index == 41:
            self.bn53 = layer
        elif index == 42:
            self.re53 = layer
        elif index == 43:
            self.pool5 = layer

        elif index == 44:
            self.pool5d = layer
        elif index == 45:
            self.conv53d = layer
        elif index == 46:
            self.bn53d = layer
        elif index == 47:
            self.re53d = layer
        elif index == 48:
            self.conv52d = layer
        elif index == 49:
            self.bn52d = layer
        elif index == 50:
            self.re52d = layer
        elif index == 51:
            self.conv51d = layer
        elif index == 52:
            self.bn51d = layer
        elif index == 53:
            self.re51d = layer

        elif index == 54:
            self.pool4d = layer
        elif index == 55:
            self.conv43d = layer
        elif index == 56:
            self.bn43d = layer
        elif index == 57:
            self.re43d = layer
        elif index == 58:
            self.conv42d = layer
        elif index == 59:
            self.bn42d = layer
        elif index == 60:
            self.re42d = layer
        elif index == 61:
            self.conv41d = layer
        elif index == 62:
            self.bn41d = layer
        elif index == 63:
            self.re41d = layer

        elif index == 64:
            self.pool3d = layer
        elif index == 65:
            self.conv33d = layer
        elif index == 66:
            self.bn33d = layer
        elif index == 67:
            self.re33d = layer
        elif index == 68:
            self.conv32d = layer
        elif index == 69:
            self.bn32d = layer
        elif index == 70:
            self.re32d = layer
        elif index == 71:
            self.conv31d = layer
        elif index == 72:
            self.bn31d = layer
        elif index == 73:
            self.re31d = layer

        elif index == 74:
            self.pool2d = layer
        elif index == 75:
            self.conv22d = layer
        elif index == 76:
            self.bn22d = layer
        elif index == 77:
            self.re22d = layer
        elif index == 78:
            self.conv21d = layer
        elif index == 79:
            self.bn21d = layer
        elif index == 80:
            self.re21d = layer

        elif index == 81:
            self.pool1d = layer
        elif index == 82:
            self.conv12d = layer
        elif index == 83:
            self.bn12d = layer
        elif index == 84:
            self.re12d = layer
        elif index == 85:
            self.conv11d = layer
        elif index == 86:
            self.bn11d = layer
        elif index == 87:
            self.re11d = layer


    def load_from_segnet(self, model_path):
        s_dict = self.state_dict()# create a copy of the state dict
        th = torch.load(model_path).state_dict() # load the weigths
        # for name in th:
            # s_dict[corresp_name[name]] = th[name]
        self.load_state_dict(th)

    def init_weight_pytorch_default(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
                if layer.bias is not None:
                    fan_in, _ = init._calculate_fan_in_and_fan_out(layer.weight)
                    bound = 1 / math.sqrt(fan_in)
                    init.uniform_(layer.bias, -bound, bound)
            if isinstance(layer, nn.BatchNorm2d):
                if layer.track_running_stats:
                    layer.running_mean.zero_()
                    layer.running_var.fill_(1)
                    layer.num_batches_tracked.zero_()
                if layer.affine:
                    init.ones_(layer.weight)
                    init.zeros_(layer.bias)

    def restore_weight(self,pre_model):
        pre_model = pre_model.cuda()
        for i, layer in enumerate(self.modules()):
            if isinstance(layer, nn.Conv2d):

                layer.weight = list(pre_model.modules())[i].weight

                layer.bias = list(pre_model.modules())[i].bias
            if isinstance(layer, nn.BatchNorm2d):

                layer.weight = list(pre_model.modules())[i].weight
                layer.bias = list(pre_model.modules())[i].bias
                layer.running_mean = list(pre_model.modules())[i].running_mean
                layer.running_var = list(pre_model.modules())[i].running_var
                layer.num_batches_tracked = list(pre_model.modules())[i].num_batches_tracked

    def init_vgg16_bn_weight(self, model_path):

        weights = torch.load(model_path)
        del weights["classifier.0.weight"]
        del weights["classifier.0.bias"]
        del weights["classifier.3.weight"]
        del weights["classifier.3.bias"]
        del weights["classifier.6.weight"]
        del weights["classifier.6.bias"]

        index = 0

        for i, layer in enumerate(self.modules()):
            if isinstance(layer, nn.Conv2d):
                if index == 0:
                    layer.weight.data = list(weights.items())[index][1][:,0:1,:,:]
                else:
                    layer.weight.data = list(weights.items())[index][1]
                index += 1
                layer.bias.data = list(weights.items())[index][1]
                index += 1
            if isinstance(layer, nn.BatchNorm2d):
                layer.weight.data = list(weights.items())[index][1]
                index += 1
                layer.bias.data = list(weights.items())[index][1]
                index += 1
                layer.running_mean = list(weights.items())[index][1]
                index += 1
                layer.running_var = list(weights.items())[index][1]
                index += 1
            if index == 78:
                break
