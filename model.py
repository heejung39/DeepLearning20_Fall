import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class UNet_G(nn.Module):
    def __init__(self, n_channels=3, ch=[32, 64, 128, 256, 512]):
        super(UNet_G, self).__init__()
        self.inc = self.root(n_channels, ch[0], kernel_size=7, stride=1, padding=3)  # conv2번: 1->64->64 channel
        self.ch = ch
        layersE = [self.ENC_L(in_ch, out_ch, kernel_size=3, stride=1, padding=1, pool_kernel_size=3) \
                   for in_ch, out_ch in zip(self.ch, self.ch[1:])]
        self.ENC = nn.Sequential(*layersE)
        # 64->128->256->512->1024 conv

        self.ch_r = self.ch.copy()
        self.ch_r.reverse()
        layersD = [self.DEC_L(in_ch + out_ch, out_ch, cnt, kernel_size=3, stride=1, padding=1) \
                   for in_ch, out_ch, cnt in zip(self.ch_r, self.ch_r[1:], range(1, len(ch) + 1))]
        self.DEC = nn.Sequential(*layersD)

        # 1024->512->256->128->64
        # self.outc = self.outconv(ch[0], n_classes)
        # 64 -> 3(n_classes) conv

    # conv 2번 - relu
    class root(nn.Module):
        def __init__(self, in_ch, out_ch, *args, **kwargs):
            super().__init__()
            # pdb.set_trace()
            self.root_blk = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, **kwargs),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, **kwargs),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            x = self.root_blk(x)
            return x

    # xD ['x'], ['feature_map']
    # first(conv in->out) + second(conv out->out) repeat만큼
    class ENC_L(nn.Module):
        def __init__(self, in_ch, out_ch, pool_kernel_size=2, repeat=2, *args, **kwargs):
            super().__init__()
            self.process = []
            self.process.append(nn.MaxPool2d(kernel_size=pool_kernel_size, stride=2))

            first = [nn.Conv2d(in_ch, out_ch, **kwargs),
                     nn.BatchNorm2d(out_ch),
                     nn.ReLU(inplace=True)]
            self.process.extend(first)
            # process = [maxpool2d, conv2d, batchnorm, relu]

            second = []
            for i in range(repeat):
                second.append(nn.Conv2d(out_ch, out_ch, **kwargs))
                second.append(nn.BatchNorm2d(out_ch))
                second.append(nn.ReLU(inplace=True))

            self.process.extend(second)
            self.seq = nn.Sequential(*self.process)

        def forward(self, xD):
            xD['x'] = self.seq(xD['x'])
            xD['feature_map'].append(xD['x'])
            return xD

    # conv 2번
    class DEC_L(nn.Module):
        def __init__(self, in_ch, out_ch, cnt, *args, **kwargs):
            super().__init__()
            self.cnt = cnt
            self.seq = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, **kwargs),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, **kwargs),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True))
            # nn.MaxPool2d(kernel_size=2, stride=2),

        # feature_map에 저장되어 있는 사이즈를 기준으로 x에 저장되어 있는 현재 크기 늘려주기
        # feature_map과 x를 합치기
        def forward(self, xD):
            sh = xD['feature_map'][-self.cnt - 1].size()[2:]
            xD['x'] = F.interpolate(xD['x'], size=sh, mode='bilinear', align_corners=True)
            xD['x'] = torch.cat([xD['x'], xD['feature_map'][-self.cnt - 1]], dim=1)  # 텐서 결합
            # xD['feature_map'].append(xD['x'])
            xD['x'] = self.seq(xD['x'])

            return xD
    # conv 1 -> classifier
    # class outconv(nn.Module):
    #     def __init__(self, in_ch, out_ch):
    #         super().__init__()
    #         self.conv = nn.Conv2d(in_ch, out_ch, 1)
    #
    #     def forward(self, x):
    #         x = self.conv(x)
    #         self.out = nn.Sigmoid()
    #         return x

    def forward(self, xt):
        # pdb.set_trace()
        x0 = self.inc(xt)
        # feature_map = []
        xD = {
            'feature_map': [x0],
            'x': x0
        }
        xD = self.ENC(xD)
        xD = self.DEC(xD)  # 반대로
        # xo = self.outc(xD['x'])
        # return self.out(xo)  # softmax
        return xD

class Unet_C(nn.Module):
    def __init__(self, in_ch = 32, out_ch = 1):
        super(Unet_C, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
        # self.seq = nn.Sequential(
        #     nn.Conv2d(in_ch, out_ch, 1),
        #     nn.BatchNorm2d(out_ch),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(out_ch, out_ch, 1))
        self.sigmoid = nn.Sigmoid()

    # conv 1 -> classifier
    def forward(self, xD):
        # xo = self.seq(xD['x'])
        xo = self.conv(xD['x'])
        xo = self.sigmoid(xo)
        return xo

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, ch=[32,64,128,196,256]): #32,64,128,196,256-3번째꺼
        super(UNet, self).__init__()
        self.inc = self.root(n_channels, ch[0], kernel_size=7, stride=1, padding=3)  # conv2번: 1->64->64 channel
        self.ch = ch
        layersE = [self.ENC_L(in_ch, out_ch, kernel_size=3, stride=1, padding=1, pool_kernel_size=3) \
                   for in_ch, out_ch in zip(self.ch, self.ch[1:])]
        self.ENC = nn.Sequential(*layersE)
        # 64->128->256->512->1024 conv

        self.ch_r = self.ch.copy()
        self.ch_r.reverse()
        layersD = [self.DEC_L(in_ch + out_ch, out_ch, cnt, kernel_size=3, stride=1, padding=1) \
                   for in_ch, out_ch, cnt in zip(self.ch_r, self.ch_r[1:], range(1, len(ch) + 1))]
        self.DEC = nn.Sequential(*layersD)
        # 1024->512->256->128->64

        self.outc = self.outconv(ch[0], n_classes)
        # 64 -> 3(n_classes) conv
        if n_classes > 1:
            self.out = nn.Softmax()
        else:
            self.out = nn.Sigmoid()

    # conv 2번 - relu
    class root(nn.Module):
        def __init__(self, in_ch, out_ch, *args, **kwargs):
            super().__init__()

            self.root_blk = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, **kwargs),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, **kwargs),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            x = self.root_blk(x)
            return x

    # xD ['x'], ['feature_map']
    # first(conv in->out) + second(conv out->out) repeat만큼
    class ENC_L(nn.Module):
        def __init__(self, in_ch, out_ch, pool_kernel_size=2, repeat=2, *args, **kwargs):
            super().__init__()
            self.process = []
            self.process.append(nn.MaxPool2d(kernel_size=pool_kernel_size, stride=2))

            first = [nn.Conv2d(in_ch, out_ch, **kwargs),
                     nn.BatchNorm2d(out_ch),
                     nn.ReLU(inplace=True)]
            self.process.extend(first)
            # process = [maxpool2d, conv2d, batchnorm, relu]

            second = []
            for i in range(repeat):
                second.append(nn.Conv2d(out_ch, out_ch, **kwargs))
                second.append(nn.BatchNorm2d(out_ch))
                second.append(nn.ReLU(inplace=True))

            self.process.extend(second)
            self.seq = nn.Sequential(*self.process)

        def forward(self, xD):
            xD['x'] = self.seq(xD['x'])
            xD['feature_map'].append(xD['x'])
            return xD

    # conv 2번
    class DEC_L(nn.Module):
        def __init__(self, in_ch, out_ch, cnt, *args, **kwargs):
            super().__init__()
            self.cnt = cnt
            self.seq = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, **kwargs),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, **kwargs),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True))
            # nn.MaxPool2d(kernel_size=2, stride=2),

        # feature_map에 저장되어 있는 사이즈를 기준으로 x에 저장되어 있는 현재 크기 늘려주기
        # feature_map과 x를 합치기
        def forward(self, xD):
            sh = xD['feature_map'][-self.cnt - 1].size()[2:]
            xD['x'] = F.interpolate(xD['x'], size=sh, mode='bilinear', align_corners=True)
            xD['x'] = torch.cat([xD['x'], xD['feature_map'][-self.cnt - 1]], dim=1)  # 텐서 결합
            # xD['feature_map'].append(xD['x'])
            xD['x'] = self.seq(xD['x'])

            return xD

    # conv 1
    class outconv(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.conv = nn.Conv2d(in_ch, out_ch, 1)

        def forward(self, x):
            x = self.conv(x)
            return x

    def forward(self, xt):
        x0 = self.inc(xt)
        # feature_map = []
        xD = {
            'feature_map': [x0],
            'x': x0
        }
        xD = self.ENC(xD)
        xD = self.DEC(xD)  # 반대로
        xo = self.outc(xD['x'])
        return self.out(xo)  # softmax