import argparse
import sys
from packaging import version
import time
import util
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

sys.path.append('Untitle')
from model import Unet, dice_loss
from loss import CrossEntropyLoss2d

#data
##image directory
IMG_DIRECTORY = '/Untitle/images/'
SOURCE_DIR = 'chasedb1/'
TARGET_DIR = 'stare/'
MODEL = 'UNET'
MODEL_SAVE = 'check/'
GPU = 0
BATCH_SIZE = 5
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
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result for self-training.")
    parser.add_argument("--epoch", type=int, default=EPOCH,
                        help="Number of epochs per round for self-training.")
    parser.add_argument("--num_round", type=int, default= NUM_ROUNDS,
                        help="Number of rounds")
    return parser.parse_args()
args = get_arguments()


def main():
    root_dir = args.img_root_dir
    source_dir = root_dir + args.source_dir
    target_dir = root_dir + args.target_dir
    device = torch.device("cuda:" + str(args.gpu))
    result_save_path = args.save
    save_pseudo_label_path = os.path.join(target_dir, 'pseudo_label')
    save_stats_path = os.path.join(save_path, 'stats')
    pretrained = args.pretrained
    BATCH_SIZE = args.batch_size
    NUM_ROUND = args.num_round
    if not os.path.exists(result_save_path):
        os.makedirs(save_path)
    if not os.path.exists(save_pseudo_label_path):
        os.makedirs(save_pseudo_label_path)
    if not os.path.exists(save_stats_path):
        os.makedirs(save_stats_path)

    if args.model == 'Unet':

        model = Unet()
        optimizer = torch.optim.Adam(model.paramerters(), lr= LEARNING_RATE)

    ##model pre-training
    for round_idx in range(NUM_ROUND):
        save_round_eval_path = os.path.join(args.save_model,str(round_idx))
        if not os.path.exists(save_round_eval_path):
            os.makedirs(save_round_eval_path)

        if pretrained == None:
            source_dataloader = data.DataLoader(
                chase_stare(root_dir, crop_size= (600,600), train= True, datatype= "chasedb1", transforms=transforms.Compose([
                    flip(),
                    rotation(),
                    ToTensor(),
                    normalize()
                ])),
                batch_size=BATCH_SIZE,
                shuffle=True
            )
            LEARNING_RATE = args.learning_rate
            EPOCH = args.epoch
            for i in range(EPOCH):
                print("--------------------------------Only Source------------------------------------------------")
                training(model, source_dataloader, epoch_idx= i, optimizer= optimizer)
        else:
            checkpoint = torch.load(args.pretrained)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint[optimizer_state_dict])
            loss = checkpoint['loss']
            pretrained = args.pretrained

        ###pseudo label generation
        if round_idx != args.num_round -1:
            conf_dict, save_prob_path, save_pred_path = val(model, device, save_round_eval_path, round_idx, args)


def training(model, dataloader, epoch_idx, optimizer):
    model.train()
    total_loss = 0
    for idx, datafiles in enumerate(dataloader):
        img, label, name = datafiles["img"], datafiles["label"], datafiles["name"]
        img, label = img.to(device), label.to(device)

        optimizer.zero_grad()
        pred = model(img)
        loss = CrossEntropyLoss2d(img, pred)
        total_loss += loss
        loss.backward()
        optimizer.step()
    print('iter = {} of {} completed, loss = {:.4f}'.format(epoch_idx, EPOCH, total_loss))

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss' : loss
    }, os.path.join(args.save_pretrained, 'pretrained' + str(epoch_idx+1) + '.pth'))

def val(model, device, save_round_eval_path, round_idx, tgt_num, label_2_id, valid_labels, args):
    """Create the model and start the evaluation process."""
    ## scorer
    scorer = ScoreUpdater(valid_labels, args.num_classes, tgt_num, logger)
    scorer.reset()
    h, w = map(int, args.input_size.split(','))

    ## test data loader
    testloader = data.DataLoader(chase_stare(args.root, train= False, datatype= "stare", transforms=transforms.Compose([
                    flip(),
                    rotation(),
                    ToTensor(),
                    normalize()
                ])),
                                    batch_size=1, shuffle=False)

    ## model for evaluation
    model.eval()
    model.to(device)
    ## output folder
    save_pred_vis_path = osp.join(save_round_eval_path, 'pred_vis')
    save_prob_path = osp.join(save_round_eval_path, 'prob')
    save_pred_path = osp.join(save_round_eval_path, 'pred')
    if not os.path.exists(save_pred_vis_path):
        os.makedirs(save_pred_vis_path)
    if not os.path.exists(save_prob_path):
        os.makedirs(save_prob_path)
    if not os.path.exists(save_pred_path):
        os.makedirs(save_pred_path)

    ## evaluation process
    print('###### Start evaluating target domain train set in round {}! ######'.format(round_idx))
    with torch.no_grad():
        for index, datafiles in enumerate(testloader):
            image, label, name = datafiles["img"], datafiles["label"], datafiles["name"]
            image, label = image.to(device), label.to(device)
            output = model(image)
            amax_output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
            conf = np.amax(output,axis=2)
            # score
            pred_label = amax_output.copy()
            label = label_2_id[np.asarray(label.numpy(), dtype=np.uint8)]
            scorer.update(pred_label.flatten(), label.flatten(), index)

            # save visualized seg maps & predication prob map
            amax_output_col = colorize_mask(amax_output)
            name = name[0].split('/')[-1]
            image_name = name.split('.')[0]
            # prob
            np.save('%s/%s.npy' % (save_prob_path, image_name), output)
            # trainIDs/vis seg maps
            amax_output = Image.fromarray(amax_output)
            amax_output.save('%s/%s.png' % (save_pred_path, image_name))
            amax_output_col.save('%s/%s_color.png' % (save_pred_vis_path, image_name))

            # save class-wise confidence maps
            if args.kc_value == 'conf':
                for idx_cls in range(args.num_classes):
                    idx_temp = pred_label == idx_cls
                    pred_cls_num[idx_cls] = pred_cls_num[idx_cls] + np.sum(idx_temp)
                    if idx_temp.any():
                        conf_cls_temp = conf[idx_temp].astype(np.float32)
                        len_cls_temp = conf_cls_temp.size
                        # downsampling by ds_rate
                        conf_cls = conf_cls_temp[0:len_cls_temp:args.ds_rate]
                        conf_dict[idx_cls].extend(conf_cls)
            elif args.kc_value == 'prob':
                for idx_cls in range(args.num_classes):
                    idx_temp = pred_label == idx_cls
                    pred_cls_num[idx_cls] = pred_cls_num[idx_cls] + np.sum(idx_temp)
                    # prob slice
                    prob_cls_temp = output[:,:,idx_cls].astype(np.float32).ravel()
                    len_cls_temp = prob_cls_temp.size
                    # downsampling by ds_rate
                    prob_cls = prob_cls_temp[0:len_cls_temp:args.ds_rate]
                    conf_dict[idx_cls].extend(prob_cls) # it should be prob_dict; but for unification, use conf_dict
    logger.info('###### Finish evaluating target domain train set in round {}! Time cost: {:.2f} seconds. ######'.format(round_idx, time.time()-start_eval))

    return conf_dict, pred_cls_num, save_prob_path, save_pred_path  # return the dictionary containing all the class-wise confidence vectors

