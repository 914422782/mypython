import os
import numpy as np
import random
import cv2
from torch.utils import data
import pickle
import torch
import torchvision.transforms as transforms
from PIL import Image


# torch.set_default_dtype(torch.float64)

"""
CamVid is a road scene understanding dataset with 367 training images and 233 testing images of day and dusk scenes. 
The challenge is to segment 11 classes such as road, building, cars, pedestrians, signs, poles, side-walk etc. We 
resize images to 360x480 pixels for training and testing.
"""


class LabelToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()

class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0):

        self.mean = mean
        self.variance = variance

    def __call__(self, img):
        img = np.array(img,dtype= np.float32)
        #第二个参数为标准差 variance为方差
        noise = np.random.normal(self.mean, self.variance ** 0.5, img.shape)
        out = img + noise
        if out.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.
        out = np.clip(out, low_clip, 1.0)
        return torch.from_numpy(out).type(torch.FloatTensor)


class SyringeDataSet(data.Dataset):
    """
       CamVidDataSet is employed to load train set
       Args:
        root: the CamVid dataset path,
        list_path: camvid_train_list.txt, include partial path

    """
    def __init__(self, root='',var=0.01, crop_size=(120, 350)):
        # 所有图片的绝对路径
        self.var = var
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, k) for k in imgs]
        imgs.sort()

        self.transform = transforms.Compose(
            [transforms.Resize(crop_size),
             transforms.ToTensor()]
        )
        self.noise_transform = transforms.Compose(
            [
                AddGaussianNoise(mean=0.0, variance=self.var),
            ]
        )

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        pil_img = Image.open(img_path)
        data = self.transform(pil_img)
        noise_data = self.noise_transform(data)
        return data, noise_data


class SyringeTestDataSet(data.Dataset):
    """
       CamVidValDataSet is employed to load val set
       Args:
        root: the CamVid dataset path,
        list_path: camvid_val_list.txt, include partial path

    """

    def __init__(self, root='',  label_root='', var=0.01, crop_size=(120, 350)):
        # 所有图片的绝对路径
        self.var = var
        self.root = root
        self.label_root = label_root
        imgs = os.listdir(root)
        label_imgs = os.listdir(label_root)
        self.imgs = [os.path.join(root, k) for k in imgs]
        self.label_imgs = [os.path.join(label_root, k) for k in label_imgs]
        imgs.sort()
        label_imgs.sort()

        self.transform = transforms.Compose(
            [transforms.Resize(crop_size),
             transforms.ToTensor()]
        )
        self.noise_transform = transforms.Compose(
            [
                AddGaussianNoise(mean=0.0, variance=self.var),
            ]
        )

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label_img_path = self.label_imgs[index]
        pil_img = Image.open(img_path)
        label_pic_img = Image.open(label_img_path)
        data = self.transform(pil_img)
        label_data = self.transform(label_pic_img)
        noise_data = self.noise_transform(data)
        return data, noise_data, label_data

