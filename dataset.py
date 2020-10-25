import cv2
import numpy as np
import os
import torch
from torch.utils import data

class dataset(data.Dataset):
    def __init__(self, img_dir, train= True, img_count = 50, height=700, width=700, type='stare', transforms=None):
        self.dir = img_dir
        self.img_count = img_count
        self.height = height
        self.width = width
        self.type = type
        self.transforms = transforms
        img_p = self.dir + '/images'
        label_p = self.dir + '/labels'
        list_img = os.listdir(img_p)
        self.name_set = []
        if train:
            list_img = list_img[:img_count]
        else:
            list_img = list_img[img_count:]

        list_img = sorted(list_img)
        self.dataset_img = np.zeros(shape=[len(list_img), self.height, self.width, 3])
        self.dataset_label = np.zeros(shape=[len(list_img), self.height, self.width])

        for idx, name in enumerate(list_img):
            tmp_img = img_p + '/' + name
            img = cv2.imread(tmp_img, cv2.IMREAD_COLOR)
            if type == 'stare':
                tmp_label = label_p + '/' + name[:-3] + 'vk.ppm'
            elif type == 'chasedb1':
                tmp_label = label_p +'/' + name[:-4] + '_1stHO.png'
            # elif type == 'UWF':
            #     tmp_label = label_p + '/' + name
            label = cv2.imread(tmp_label, cv2.IMREAD_GRAYSCALE)
            #605x700
            #height =  605, width = 700
            #h,w,c
            if img.shape != (self.height, self.width):
                img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
                label = cv2.resize(label, (self.width, self.height), interpolation=cv2.INTER_AREA)
            # img = cv2.split(img)[1] #green channel
            self.dataset_img[idx, ...] = img
            # label = np.expand_dims(label, 3)
            self.dataset_label[idx, ...] = label
            self.name_set.append(os.path.basename(name))

    def __len__(self):
        return self.dataset_img.shape[0]

    def __getitem__(self, idx):
        X = self.dataset_img[idx]
        Y = self.dataset_label[idx]
        name = self.name_set[idx].split('.')[0]
        sample = {
            "img": X,
            "label": Y,
            "name": name
        }
        if self.transforms:
            sample = self.transforms(sample)
        return sample

class dataset_UWF(data.Dataset):
    def __init__(self, img_dir, train=True, height = 605, width = 605, transforms = None):
        self.dir = img_dir
        self.height = height
        self.width = width
        self.transforms = transforms
        self.train = train
        self.list_img = os.listdir(img_dir)
        self.name_set = []
        self.list_img = self.list_img[:50]
        self.dataset_img = np.zeros(shape=[len(self.list_img), self.height, self.width])
        for idx, name in enumerate(self.list_img):
            tmp_img = self.dir + '/' + name
            img = cv2.imread(tmp_img, cv2.IMREAD_GRAYSCALE)
            #img = 605x700x3 width x height
            if img.shape != (height, width):
                img = cv2.resize(img, (height, width), interpolation=cv2.INTER_AREA)
            self.dataset_img[idx, ...] = img
            self.name_set.append(os.path.basename(name))

    def __len__(self):
        return self.dataset_img.shape[0]

    def __getitem__(self, idx):
        X = self.dataset_img[idx]
        name = self.name_set[idx]

        if self.transforms:
            X = self.transforms(X)
        return X, name

class normalize(object):
    def __call__(self, sample):
        X = sample["img"]
        Y = sample["label"]
        name = sample["name"]
        X = X / 255
        Y = Y / 255
        mean = torch.mean(X)
        std = torch.std(X)
        X = (X - mean) / std
        X = np.clip(X, 0, 1)
        sample= {
            "img": X,
            "label": Y,
            "name" : name
        }
        return sample

class normalize_UWF(object):
    def __call__(self, X):
        X = X / 255
        mean = torch.mean(X)
        std = torch.std(X)
        X = (X - mean) / std
        X = np.clip(X, 0, 1)
        return X

class flip(object):
    def __call__(self, sample):
        X = sample["img"]
        Y = sample["label"]
        name = sample["name"]

        rm = np.random.rand()
        if rm > 0.7:
            X = cv2.flip(X, 1)
            Y = cv2.flip(Y, 1)

        elif rm > 0.3:
            X = cv2.flip(X, 0)
            Y = cv2.flip(Y, 0)

        sample= {
            "img": X,
            "label": Y,
            "name" : name
        }
        return sample

class flip_UWF(object):
    def __call__(self, X):
        rm = np.random.rand()
        if rm > 0.7:
            X = cv2.flip(X, 1)
        elif rm > 0.3:
            X = cv2.flip(X, 0)
        return X

class rotation(object):
    def __call__(self, sample):
        X = sample["img"]
        Y = sample["label"]
        name = sample["name"]
        height = len(X[0])
        width = len(X[1])
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
        sample= {
            "img": X,
            "label": Y,
            "name" : name
        }
        return sample
    
class rotation_UWF(object):
    def __call__(self, X):
        height = len(X[0])
        width = len(X[1])
        rm = np.random.rand()
        if rm > 0.8:
            M = cv2.getRotationMatrix2D((height / 2, width / 2), 90, 1)
            X = cv2.warpAffine(X, M, (height, width))
        elif rm > 0.6:
            M = cv2.getRotationMatrix2D((height / 2, width / 2), -90, 1)
            X = cv2.warpAffine(X, M, (height, width))
        elif rm > 0.4:
            M = cv2.getRotationMatrix2D((height / 2, width / 2), 45, 1)
            X = cv2.warpAffine(X, M, (height, width))
        elif rm > 0.2:
            M = cv2.getRotationMatrix2D((height / 2, width / 2), -45, 1)
            X = cv2.warpAffine(X, M, (height, width))
        return X

class translate(object):
    def __call__(self,sample):
        X = sample["img"]
        Y = sample["label"]
        name = sample["name"]
        height, width, _ = X.shape
        rm = np.random.rand()
        if rm > 0.5:
            rm2 = np.random.uniform(-0.3, 0.3)
            rm3 = np.random.uniform(-0.3, 0.3)
            move_x = int(width*rm2)
            move_y = int(height*rm3)
            M = np.float32([[1,0, move_x], [0,1, move_y]])
            X = cv2.warpAffine(X, M, (width, height))
            Y = cv2.warpAffine(Y, M, (width, height))
        sample = {
            "img": X,
            "label": Y,
            "name": name
        }
        return sample

class ToTensor(object):
    def __call__(self, sample):
        X = sample["img"]
        Y = sample["label"]
        name = sample["name"]
        # X = np.expand_dims(X,3)
        Y = np.expand_dims(Y,3)
        X = np.transpose(X, [2, 0, 1])
        Y = np.transpose(Y, [2, 0, 1])
        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y).float()
        sample= {
            "img": X,
            "label": Y,
            "name" : name
        }
        return sample

class ToTensor_UWF(object):
    def __call__(self, X):
        X = np.expand_dims(X,3)
        X = np.transpose(X, [2, 0, 1])
        X = torch.from_numpy(X).float()
        return X