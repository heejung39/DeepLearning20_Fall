import os
import numpy as np
import random
import torch
import torchvision.transforms as transforms
import torchvision
import cv2
from torch.utils import data

class chase_stare(data.Dataset):
    def __init__(self, root, crop_size=(500,500), train= True, datatype= None, transforms= None):
        self.root = root
        self.crop_size = crop_size
        self.train = train
        self.datatype = datatype
        self.transforms= transforms
        self.img_list = os.listdir(self.root + 'images/')
        self.files = []
        self.img_ids = []
        self.label_ids =[]
        if train:
            self.img_list[:15] # total 28(chasedb1), 20(stare)
        else:
            self.img_list[15:]
        for name in self.img_list:
            img_file = os.path.join(self.root, 'images/',name)
            if self.datatype == chasedb1:
                label_file = os.path.join(self.root, 'labels/',name[:-4] + "_1stHO.png")
            else:
                label_file = os.path.join(self.root, 'labels/', name[:-3] +"vk.ppm")

        self.files.append({
            "img": img_file,
            "label": label_file
            "name": name
        })
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        datafiles = self.files[idx]
        img = cv2.imread(datafiles["img"])
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        name = datafiles["name"]
        w,h,_ = img.shape
        if (w,h) != self.crop_size:
            img = cv2.resize(img, self.crop_size, interpolations= cv2.INTER_AREA)
            label = cv2.resize(label, self.crop_size, interpolation= cv2.INTER_AREA)
            datafiles= {
                "img": img,
                "label": label,
                "name": name
            }

        if self.transforms:
            X = self.transforms(datafiles)

        return datafiles

class normalize(object):
    def __call__(self, datafiles):
        X = datafiles["img"] / 255
        Y = datafiles["label"] / 255
        name = datafiles["name"]
        mean = torch.mean(X)
        std = torch.std(X)
        X = (X - mean) / std
        X = np.clip(X, 0, 1)
        datafiles = {
            "img": X,
            "label": Y,
            "name": name
        }
        return datafiles

class ToTensor(object):
    def __call__(self, datafiles):
        X = datafiles["img"]
        Y = datafiles["label"]
        name = datafiles["name"]
        Y = np.expand_dims(Y,3)
        X = np.transpose(X, [2, 0, 1])
        Y = np.transpose(Y, [2, 0, 1])
        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y).float()
        datafiles = {
            "img": X,
            "label": Y,
            "name": name
        }
        return datafiles


class rotation(object):
    def __call__(self, datafiles):
        X = datafiles["img"]
        Y = datafiles["label"]
        name = datafiles["name"]
        width, height = X.shape
        rm = np.random.rand()
        if rm > 0.8:
            M = cv2.getRotationMatrix2D((height / 2, width / 2), 90, 1)
            X = cv2.warpAffine(X, M, (height, width))
            Y = cv2.warpAffine(Y, M, (height, width))
        elif rm > 0.6:
            M = cv2.getRotationMatrix2D((height / 2, width / 2), -90, 1)
            X = cv2.warpAffine(X, M, (height, width))
            Y = cv2.warpAffine(Y, M, (height, width))
        elif rm > 0.4:
            M = cv2.getRotationMatrix2D((height / 2, width / 2), 45, 1)
            X = cv2.warpAffine(X, M, (height, width))
            Y = cv2.warpAffine(Y, M, (height, width))
        elif rm > 0.2:
            M = cv2.getRotationMatrix2D((height / 2, width / 2), -45, 1)
            X = cv2.warpAffine(X, M, (height, width))
            Y = cv2.warpAffine(Y, M, (height, width))
        datafiles = {
            "img": X,
            "label": Y,
            "name": name
        }
        return datafiles

class flip(object):
    def __call__(self, datafiles):
        X = datafiles["img"]
        Y = datafiles["label"]
        name = datafiles["name"]
        rm = np.random.rand()
        if rm > 0.7:
            X = cv2.flip(X, 1)
            Y = cv2.flip(Y, 1)
        elif rm > 0.3:
            X = cv2.flip(X, 0)
            Y = cv2.flip(Y, 0)

        datafiles = {
            "img": X,
            "label": Y,
            "name": name
        }
        return datafiles