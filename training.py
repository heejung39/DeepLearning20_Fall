from model import Unet_C, UNet_G
from loss import Diff2d, CrossEntropyLoss2d
from dataset import *
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
import cv2
# import visdom
import torchvision
from torchvision import transforms, utils
import argparse


model_g = UNet_G()
model_f1 = Unet_C()
model_f2 = Unet_C()
model_mc = Unet_C()

def dice_loss(pred, label):
    p = torch.sum(pred, (-1, -2))
    tx = torch.sum(label, (-1, -2))
    pt = torch.sum(pred * label, (-1, -2))

    num = 2 * pt
    den = p + tx
    dsc = torch.mean(num / den, 0)
    loss = 1-dsc
    return dsc

optimizer_g = torch.optim.Adam(model_g.parameters(), lr = 1e-3)
optimizer_f = torch.optim.Adam(list(model_f1.parameters()) + list(model_f2.parameters()), lr = 1e-3)
optimizer_mc = torch.optim.Adam(model_mc.parameters(), lr = 1e-3)
EPOCHs = 20000
BATCH_SIZE = 2

# vis = visdom.Visdom()

# def loss_tracker(loss_plot, loss_value, num):
#     '''num, loss_value, are Tensor'''
#     vis.line(X=num,
#              Y=loss_value,
#              win = loss_plot,
#              update='append'
#              )


# def __init__(self, img_dir, train=True, img_count=50, height=700, width=700, type='stare', transforms=None):

#stare 700x605 chasebd1 999x960
source_train_data = data.DataLoader(
    dataset('images/stare', train= True, img_count= 15, height= 605, width = 605, type= 'stare', transforms=transforms.Compose([
        flip(),
        rotation(),
        translate(),
        ToTensor(),
        normalize()
    ])),
    batch_size=BATCH_SIZE,
    shuffle=True
)
target_train_data = data.DataLoader(
    dataset('images/chasedb1',train= True, img_count=15, height = 605, width = 605,type= 'chasedb1', transforms= transforms.Compose([
        flip(),
        rotation(),
        translate(),
        ToTensor(),
        normalize()
    ])),
    batch_size=BATCH_SIZE,
    shuffle=True
)
source_test_data = data.DataLoader(
    dataset('images/stare', train=False, img_count=15, height = 605, width = 605,transforms= transforms.Compose([
        ToTensor(),
        normalize()
    ])),
    batch_size = BATCH_SIZE,
    shuffle = False
)

target_test_data = data.DataLoader(
    dataset('images/chasedb1', train= False, img_count=15, height = 605, width = 605, type= 'chasedb1', transforms= transforms.Compose([
        ToTensor(),
        normalize()
    ])),
    batch_size = BATCH_SIZE,
    shuffle = False
)


USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
model_g, model_f1, model_f2, model_mc = model_g.to(DEVICE), model_f1.to(DEVICE), model_f2.to(DEVICE), model_mc.to(DEVICE)

criterion = CrossEntropyLoss2d()
criterion_d = Diff2d()

# checkpoint = torch.load('check/281.tar')
# model_g.load_state_dict(checkpoint['g_state_dict'])
# model_f1.load_state_dict(checkpoint['f1_state_dict'])
# model_f2.load_state_dict(checkpoint['f2_state_dict'])
# optimizer_mc.load_state_dict(checkpoint['optimizer_mc'])
# optimizer_g.load_state_dict(checkpoint['optimizer_g'])
# optimizer_f.load_state_dict(checkpoint['optimizer_f'])

#step 1; generator, classifier update w/ source data
#step 2; classifier update w/ target data(generator freeze)
#main is only trained on source data
#step 3; generator update w/ target data(classifier freeze)

def evaluate(model_g, model_mc, source_data, target_data):
    model_g.eval()
    model_mc.eval()
    source_loss_list = []
    target_loss_list = []
    with torch.no_grad():
        for sample in source_data:
            data = sample["img"]
            mask = sample["label"]
            if torch.cuda.is_available():
                data, mask = data.cuda(), mask.cuda()
            output = model_g(data)
            output = model_mc(output)
            dsc = dice_loss(output, mask)
            source_loss_list.append(dsc)
            # tensor [2, 1, 512, 512] -> [512, 512] -> cpu().numpy() * 255

        for sample in target_data:
            data = sample["img"]
            mask = sample["label"]
            if torch.cuda.is_available():
                data, mask = data.cuda(), mask.cuda()
            output = model_g(data)
            output = model_mc(output)
            dsc = dice_loss(output, mask)
            target_loss_list.append(dsc)
            pred = output.squeeze().cpu().numpy()

    source_loss = sum(source_loss_list) / len(source_loss_list)
    target_loss = sum(target_loss_list) / len(target_loss_list)
    return source_loss.item(), target_loss.item()
#
# d_loss_plt = vis.line(Y=torch.Tensor(1).zero_(),opts=dict(title='d_loss_tracker', legend=['d_loss'], showlegend=True))
# c_loss_plt = vis.line(Y=torch.Tensor(1).zero_(),opts=dict(title='c_loss_tracker', legend=['c_loss'], showlegend=True))
# source_plt = vis.line(Y=torch.Tensor(1).zero_(), opts=dict(title='Test Accuracy(Source)', legend = ['Source'], showlegend = True))
# target_plt = vis.line(Y=torch.Tensor(1).zero_(), opts=dict(title='Test Accuracy(Target)', legend = ['Target'], showlegend = True))
#

for epoch in range(EPOCHs):
    model_g.train()
    model_f1.train()
    model_f2.train()
    model_mc.train()

    d_loss_per_epoch = 0
    c_loss_per_epoch = 0
    for ind, (source, target) in enumerate(zip(source_train_data, target_train_data)):
        src_img, src_lb = source["img"], source["label"]
        tar_img, tar_lb = target["img"], target["label"]
        src_img, src_lb = src_img.to(DEVICE), src_lb.to(DEVICE)
        tar_img, tar_lb = tar_img.to(DEVICE), tar_lb.to(DEVICE)
        # print("Step 1:", torch.min(src_img).item(), torch.max(src_img).item(), torch.min(src_lb).item(), torch.max(src_lb).item(),
        #                  torch.min(tar_img).item(), torch.max(tar_img).item(), torch.min(tar_lb).item(), torch.max(tar_lb).item())

        #step 1
        model_g.train()
        model_f1.train()
        model_f2.train()
        model_mc.train()

        loss = 0
        outputs = model_g(src_img)
        outputs1 = model_f1(outputs)
        outputs2 = model_f2(outputs)
        outputs3 = model_mc(outputs)
        # print("Step 1:", torch.min(outputs1).item(), torch.max(outputs1).item(),
        #                  torch.min(outputs2).item(), torch.max(outputs2).item(),
        #                  torch.min(outputs3).item(), torch.max(outputs3).item())

        loss += criterion(outputs1, src_lb)
        loss += criterion(outputs2, src_lb)
        loss += criterion(outputs3, src_lb)

        c_loss = loss.item()
        c_loss_per_epoch += c_loss

        optimizer_f.zero_grad()
        optimizer_g.zero_grad()
        optimizer_mc.zero_grad()
        loss.backward()
        optimizer_g.step()
        optimizer_f.step()
        optimizer_mc.step()

        # step2; update for classifiers
        model_g.eval()
        outputs = model_g(src_img)
        outputs1 = model_f1(outputs)
        outputs2 = model_f2(outputs)

        loss = 0
        loss += criterion(outputs1, src_lb)
        loss += criterion(outputs2, src_lb)
        outputs = model_g(tar_img)
        outputs1 = model_f1(outputs)
        outputs2 = model_f2(outputs)
        loss -= criterion_d(outputs1, outputs2)

        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
        loss.backward()
        optimizer_f.step()
        # print("Step 2:", torch.min(outputs1).item(), torch.max(outputs1).item(),
        #                  torch.min(outputs2).item(), torch.max(outputs2).item(),
        #                  torch.min(outputs3).item(), torch.max(outputs3).item())

        #step3; update generator by discrepancy, 세번씩 update
        d_loss = 0.0
        model_g.train()
        model_f1.eval()
        model_f2.eval()
        model_mc.eval()
        for i in range(3):
            loss = 0
            output = model_g(tar_img)
            output1 = model_f1(output)
            output2 = model_f2(output)
            output3 = model_mc(output)

            loss += criterion_d(output1, output2)
            loss += criterion_d(output3, output1)
            loss += criterion_d(output3, output2)

            optimizer_g.zero_grad()
            loss.backward()
            optimizer_g.step()
        d_loss += loss.item()
        d_loss_per_epoch += d_loss

        # print("Step 3:", torch.min(outputs1).item(), torch.max(outputs1).item(),
        #                  torch.min(outputs2).item(), torch.max(outputs2).item(),
        #                  torch.min(outputs3).item(), torch.max(outputs3).item())

        print('iter [%d] | DLoss: %.4f | CLoss: %.4f' %(ind, d_loss, c_loss))
    source_eval, target_eval = evaluate(model_g, model_mc, source_test_data, target_test_data)
    print("Epoch [%d] | DLoss: %.4f | CLoss: %.4f" % (epoch, d_loss_per_epoch, c_loss_per_epoch))
    print("Source accuracy: %.4f | Target accuracy : %.4f" %(source_eval, target_eval))
    print()
    # loss_tracker(d_loss_plt, torch.Tensor([d_loss_per_epoch * 100]), torch.Tensor([epoch]))
    # loss_tracker(c_loss_plt, torch.Tensor([c_loss_per_epoch * 100]), torch.Tensor([epoch]))
    # loss_tracker(source_plt, torch.Tensor([source_eval * 100]), torch.Tensor([epoch]))
    # loss_tracker(target_plt, torch.Tensor([target_eval * 100]), torch.Tensor([epoch]))

    save_dic = {
        'epoch': epoch + 1,
        'g_state_dict': model_g.state_dict(),
        'f1_state_dict': model_f1.state_dict(),
        'f2_state_dict': model_f2.state_dict(),
        'mc_state_dict' : model_mc.state_dict(),
        'optimizer_g': optimizer_g.state_dict(),
        'optimizer_f': optimizer_f.state_dict(),
        'optimizer_mc' : optimizer_mc.state_dict(),
        'loss_c' : c_loss,
        'loss_d' : d_loss
    }
    torch.save(save_dic, 'check/'+str(epoch)+'.tar')

# def main(args):
#     training(args)

# if __name__ == '__main__':
#     # parser = argparse.ArgumentParser()
#     # parser.add_argument('--device', type=str, default=0)
#     main()