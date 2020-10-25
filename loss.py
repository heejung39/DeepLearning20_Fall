from model import Unet_C, UNet_G
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.bce = nn.BCELoss(weight, size_average)

    def forward(self, inputs, targets):
        loss = self.bce(inputs, targets)
        # loss = torch.mean(-targets * torch.log(inputs + 1e-8) - (1 - targets) * torch.log(1 - inputs + 1e-8))
        return loss


class Diff2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Diff2d, self).__init__()
        self.weight = weight
        self.L1 = nn.L1Loss()

    def forward(self, inputs1, inputs2):
        return torch.mean(self.L1(inputs1, inputs2))