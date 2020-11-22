import argparse
import sys
from packaging import version
import time
import os
import os.path as osp
import timeit
import torch
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo
import torch.nn as nn
import torch.optim as optim
from operator import itemgetter

import scipy
from scipy import ndimage
import math
from PIL import Image
import numpy as np
import shutil
import random
from dataset import *
sys.path.append('Untitle')
from U_model import *
from U_loss import *

#data
##image directory
IMG_DIRECTORY = '/Untitle/images/'
SOURCE_DIR = 'chasedb1/'
TARGET_DIR = 'stare/'
MODEL = 'UNET'
MODEL_SAVE = 'check/'
GPU = 0
BATCH_SIZE = 2
INPUT_SIZE = (600,600)
LEARNING_RATE = 1e-3
SAVE_PATH = 'result/'
EPOCH = 400
NUM_ROUNDS = 5

def get_arguments():
    """Parse all the arguments provided from the CLI."""
    parser = argparse.ArgumentParser(description="UNet")
    ### shared by train & val
    # data
    parser.add_argument("--img_root_dir", type=str, default=IMG_DIRECTORY,
                        help="Root Directory")
    parser.add_argument("--source_dir", type=str, default=SOURCE_DIR,
                        help="Source directory.")
    parser.add_argument("--target_dir", type=str, default=TARGET_DIR,
                        help="Target directory.")

    # model
    parser.add_argument("--model", type=str, default=MODEL,
                        help="Model Choice.")
    parser.add_argument("--save_model", type=str, default=MODEL_SAVE,
                        help="Where restore model parameters from.")
    parser.add_argument("--save_pretrained", type=str, default= 'pretrained',
                        help="Save pretrained model")

    parser.add_argument("--pretrained", type=str, default=None,
                        help="Pretrained model directory")
    # gpu
    parser.add_argument("--gpu", type=int, default=GPU,
                        help="choose gpu device.")
    # log files
    # parser.add_argument("--log-file", type=str, default=LOG_FILE,
    #                     help="The name of log file.")
    # parser.add_argument('--debug',help='True means logging debug info.',
    #                     default=False, action='store_true')
    ### train ###
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    # params for optimizor
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")

     ### self-training params
    parser.add_argument("--save_ST", type=str, default=SAVE_PATH,
                        help="Path to save result for self-training.")
    parser.add_argument("--epoch", type=int, default=EPOCH,
                        help="Number of epochs per round for self-training.")
    parser.add_argument("--num_round", type=int, default= NUM_ROUNDS,
                        help="Number of rounds")
    parser.add_argument("--tgt_port", type= int, default= 0.5,
                        help="tgt portion for retraining")
    parser.add_argument("--save_initial_label", type= str, default= None,
                        help= "Initial label save dir")
    return parser.parse_args()
args = get_arguments()




def training(model, dataloader, optimizer, EPOCHS, save_dir, loss):
    model.train()
    best_loss = 100
    for epoch in range(EPOCHS):
        total_loss = 0
        for idx, datafiles in enumerate(dataloader):
            img, label, name = datafiles["img"], datafiles["label"], datafiles["name"]
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()
            pred = model(img)

            if loss == 'dice':
                dsc, sum_loss = dice_loss(pred, label)
                sum_loss += BCE(pred, label)

            # BCE loss 동일
            elif loss == 'get_metric':
                sum_loss = _eval_func(label, pred)
            total_loss += sum_loss
            sum_loss.backward()
            optimizer.step()
        print('iter = {} of {} completed, loss = {}'.format(epoch, EPOCHS, total_loss))
        if best_loss > total_loss:
            best_loss = total_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, save_dir + 'best_pretrained.pth')

        if epoch % 10 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss' : loss
            }, save_dir + 'pretrained' + str(epoch+1) + '.pth')


def val(model, device, testloader, round_idx, save_dir):
    """Create the model and start the evaluation process."""
    ## model for evaluation
    model.eval()
    ## evaluation process
    print('###### Start evaluating target domain train set in round {}! ######'.format(round_idx))
    with torch.no_grad():
        total_loss = 0
        for index, datafiles in enumerate(testloader):
            image, label, name = datafiles["img"], datafiles["label"], datafiles["name"]
            image, label = image.to(device), label.to(device)
            output = model(image)
            dsc, loss = dice_loss(output, label)
            # loss += BCE(output, label)
            print('loss {}'.format(loss))
            total_loss += loss
            pred = output.squeeze().cpu().numpy()
            pred[pred >= 0.5] = 1.0
            pred[pred < 0.5] = 0.0
            final_pred = pred * 255
            for i in range(len(name)):
                if final_pred.shape == (600,600):
                    final_pred_each = final_pred
                    cv2.imwrite(save_dir + name[i][:-3] + 'jpg', final_pred_each)
                else:
                    final_pred_each = final_pred[i]
                    cv2.imwrite(save_dir + name[i][:-3] + 'jpg', final_pred_each)

        print("loss for value", total_loss/len(testloader))

# def _eval_func(label, pred):
#     gt_label = label.flatten()
#     gt_label = gt_label.cpu().numpy()
#     pred = pred.cpu().numpy()
#     valid_flag = gt_label != 255
#     labels = gt_l   abel[valid_flag].astype(int)
#     n,c,h,w = pred.shape
#     valid_inds = np.where(valid_flag)[0]
#     probmap = np.rollaxis(pred.astype(np.float32),1).reshape((c, -1))
#     print(valid_inds.shape)
#     print(labels.shape)
#     valid_probmap = probmap[labels, valid_inds]
#     log_valid_probmap = -np.log(valid_probmap+1e-32)
#     sum_metric = log_valid_probmap.sum()
#     num_inst = valid_flag.sum()
#
#     return (sum_metric, num_inst + (num_inst == 0))

if __name__ =='__main__':

    root_dir = args.img_root_dir
    source_dir = args.source_dir
    target_dir = args.target_dir
    device = torch.device("cuda:" + str(args.gpu))
    result_save_path = args.save_ST
    save_pseudo_label_path = args.save_initial_label
    pretrained = args.pretrained  # dir pretrained model saved
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epoch
    NUM_ROUNDS = args.num_round

    if not os.path.exists(save_pseudo_label_path):
        os.makedirs(save_pseudo_label_path)
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)
    ##model pre-training
    for round_idx in range(NUM_ROUNDS):
        save_round_eval_path = os.path.join(args.save_model, str(round_idx))
        if not os.path.exists(save_round_eval_path):
            os.makedirs(save_round_eval_path)

        if pretrained == None:
            source_dataloader = data.DataLoader(
                chase_stare(source_dir, crop_size=(600, 600), train=True, datatype="chasedb1",
                            transforms=transforms.Compose([
                                flip(),
                                rotation(),
                                ToTensor(),
                                normalize()
                            ])),
                batch_size=BATCH_SIZE,
                shuffle=True
            )
            # pretraining w/ source data
            print("--------------------------------Only Source------------------------------------------------")
            if not os.path.exists(args.save_pretrained):
                os.makedirs(args.save_pretrained)
            training(model, source_dataloader, optimizer=optimizer, EPOCHS=EPOCHS, save_dir=args.save_pretrained,
                     loss='dice')

        else:
            checkpoint = torch.load(args.pretrained)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            loss = checkpoint['loss']
            pretrained = args.pretrained

        ###pseudo label generation
        testloader = data.DataLoader(
            chase_stare(target_dir, train=False, datatype="stare", source_target='target', transforms=transforms.Compose([
                # flip(),
                # rotation(),
                ToTensor(),
                normalize()
            ])),
            batch_size=1, shuffle=False)

        print("=====================make pseudo label============================")
        save_path = save_pseudo_label_path+ str(round_idx) +'/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        val(model, device, testloader, round_idx, save_path)

        # retraining
        print('=======================retraining start============================')
        source_dataset = chase_stare(source_dir, crop_size=(600, 600), train=True, datatype="chasedb1",
                                     source_target='source', tgt_label=None, transforms=transforms.Compose([
                flip(),
                rotation(),
                ToTensor(),
                normalize()
            ]))
        target_dataset = chase_stare(target_dir, crop_size=(600, 600), train=True, datatype="stare",
                                     source_target='target', tgt_label=save_path,
                                     transforms=transforms.Compose([
                                         flip(),
                                         rotation(),
                                         ToTensor(),
                                         normalize()
                                     ]))
        train_size = int(len(target_dataset) * 0.5)
        target_dataset_for_training, _ = torch.utils.data.random_split(target_dataset, [train_size, len(
            target_dataset) - train_size])
        trainloader = data.DataLoader(
            torch.utils.data.ConcatDataset([source_dataset, target_dataset_for_training]),
            batch_size=1, shuffle=True
        )

        result_save_path2 = result_save_path + str(round_idx) + '/'
        if not os.path.exists(result_save_path2):
            os.makedirs(result_save_path2)

        training(model, trainloader, optimizer, EPOCHS // 2, result_save_path2, loss='dice')
        pretrained = result_save_path2 + 'best_pretrained.pth'
