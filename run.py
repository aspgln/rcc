from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, RandomSampler
import mrnet.torchsample.transforms

from torchvision import datasets, models, transforms, utils
import time
import os  
import sys
import copy
import pandas as pd
import visdom
import random
import matplotlib.image as mpimg 
import warnings
import nrrd
import shutil
from datetime import datetime
from tensorboardX import SummaryWriter
import kornia
import argparse



warnings.filterwarnings("ignore")


import mrnet.mrnet_dataloader
import mrnet.mrnet_train
import mrnet.mrnet_model

import src.dataloader 
import src.train3d
import src.model



def run(args):

    ### Data Loading 
    
    if args.task == 0:
        print('Task 0: MR Dataset Prediction')
        augmentor = transforms.Compose([
            transforms.Lambda(lambda x: torch.Tensor(x)),
            mrnet.torchsample.transforms.RandomRotate(25),
            mrnet.torchsample.transforms.RandomTranslate([0.11, 0.11]),
            mrnet.torchsample.transforms.RandomFlip(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1, 1).permute(1, 0, 2, 3)),
        ])
        job = 'acl'
        plane = 'sagittal'
        train_ds = mrnet.mrnet_dataloader.MRDataset('/data/larson2/RCC_dl/MRNet-v1.0/data/', job,
                                  plane, transform=augmentor, train=True)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=1, shuffle=True, num_workers=11, drop_last=False)

        val_ds = mrnet.mrnet_dataloader.MRDataset(
            '/data/larson2/RCC_dl/MRNet-v1.0/data/', job, plane, train=False)
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=1, shuffle=-True, num_workers=11, drop_last=False)
        
    elif args.task == 1:
        print('Task 1: clear cell grade prediction')
        path = '/data/larson2/RCC_dl/new/clear_cell/'
    
        augmentor =  transforms.Compose([
            transforms.Lambda(lambda x: torch.Tensor(x)),
            src.dataloader.Rescale(-160, 240), # rset dynamic range
            transforms.Lambda(lambda x: x.repeat(3, 1, 1, 1).permute(3, 0, 1, 2)),
#             src.dataloader.Normalize(),
#             src.dataloader.Crop(90),
#             src.dataloader.RandomCenterCrop(90),

            src.dataloader.RandomHorizontalFlip(),
            src.dataloader.RandomRotate(25),
            src.dataloader.Resize(256),
        ])

        augmentor2 =  transforms.Compose([
            transforms.Lambda(lambda x: torch.Tensor(x)),
            src.dataloader.Rescale(-160, 240), # rset dynamic range
            transforms.Lambda(lambda x: x.repeat(3, 1, 1, 1).permute(3, 0, 1, 2)),
    #         src.dataloader.Normalize(),
    #         src.dataloader.Crop(90),
            src.dataloader.Resize(256),
        ])
        
        train_ds = src.dataloader.RCCDataset_h5(path, mode='train', transform=augmentor)
        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=1, drop_last=False)

        val_ds = src.dataloader.RCCDataset_h5(path, mode='val', transform=augmentor2)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=True, num_workers=1, drop_last=False)
        print(f'train size: {len(train_loader)}')
        print(f'val size: {len(val_loader)}')

        pos_weight = args.weight


    ### Some Checkers
    print('Summary: ')

    print(f'\ttrain size: {len(train_loader)}')
    print(f'\tval size: {len(val_loader)}')
    print('\tDatatype = ', train_ds[1][0].dtype)
    print('\tMin = ', train_ds[1][0].min())
    print('\tMax = ', train_ds[1][0].max())
    print('\tInput size', train_ds[0][0].shape)
    print('\tweight = ', args.weight)


    
    
    ### Some trackers 
    log_root_folder = "/data/larson2/RCC_dl/logs/"

    now = datetime.now()
    now = now.strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join(log_root_folder , f"task_{args.task}_{args.prefix_name}_model{args.model}_{now}")
    os.makedirs(logdir)
    print(f'logdir = {logdir}')
    
    writer = SummaryWriter(logdir)

    
    
    ### Model Construction
    
    ## Select Model    
    if args.model == 1:
        model = src.model.MRNet()
    elif args.model == 2:
        model = src.model.MRNet2()
    elif args.model == 3:
        model = src.model.MRNetBN()    
    elif args.model == 4:
        model = src.model.MRResNet()
    elif args.model == 5:
        model = src.model.MRNetScratch()
    elif args.model == 6:
        model = src.model.TDNet()
    else:
        print('Invalid model name')
        return
    
    ## Weight Initialization
    
    
    ## Training Stretegy
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    print('\tCuda:', torch.cuda.is_available(), f'\n\tdevice = {device}')
        
        


    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.1)

    if args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=.3, threshold=1e-4, verbose=True)
    elif args.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=3, gamma=args.gamma)

    model = model.to(device)
    

    ### Ready?
    best_val_loss = float('inf')
    best_val_auc = float(0)
    iteration_change_loss = 0
    t_start_training = time.time()
    
    ### Here we go
    for epoch in range(args.epochs):
        current_lr = src.train3d.get_lr(optimizer)

        t_start = time.time()

        train_loss, train_auc = src.train3d.train_model(
            model, train_loader, device, epoch, args.epochs, optimizer, writer, current_lr, args.log_every, args.weight)
        val_loss, val_auc = src.train3d.evaluate_model(
            model, val_loader, device, epoch, args.epochs, writer, current_lr,args.log_every,)

        if args.lr_scheduler == 'plateau':
            scheduler.step(val_loss)
        elif args.lr_scheduler == 'step':
            scheduler.step()

        t_end = time.time()
        delta = t_end - t_start

        print("train loss : {0} | train auc {1} | val loss {2} | val auc {3} | elapsed time {4} s".format(
            train_loss, train_auc, val_loss, val_auc, delta))

        iteration_change_loss += 1
        print('-' * 30)

        model_root_dir = "/data/larson2/RCC_dl/models/"


        if val_auc > best_val_auc:
            best_val_auc = val_auc
            if bool(args.save_model):
                file_name = f'task_{args.task}_model_{args.model}_{args.prefix_name}_val_auc_{val_auc:0.4f}_train_auc_{train_auc:0.4f}_epoch_{epoch+1}_weight_{args.weight}_lr_{args.lr}_gamma_{args.gamma}_lrsche_{args.lr_scheduler}.pth'
#                 for f in os.listdir(model_root_dir):
#                     if  (args.prefix_name in f):
#                         os.remove(os.path.join(model_root_dir, f))
                torch.save(model, os.path.join(model_root_dir, file_name))

    
    
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            iteration_change_loss = 0

        if iteration_change_loss == args.patience:
            print('Early stopping after {0} iterations without the decrease of the val loss'.
                  format(iteration_change_loss))
            break
            
        
    t_end_training = time.time()
    print(f'training took {t_end_training - t_start_training} s')
    #     --------------------




def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default=None)
    parser.add_argument('--prefix_name', type=str, required=True)
    parser.add_argument('--task', type=int, required=True) # 0->mrnet, 1->clear cell
    parser.add_argument('--model', type=int, required=True)


#     parser.add_argument('--augment', type=int, choices=[0, 1], default=1)
    parser.add_argument('--lr_scheduler', type=str,
                        default='plateau', choices=['plateau', 'step'])
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--flush_history', type=int, choices=[0, 1], default=0)
    parser.add_argument('--save_model', type=int, choices=[0, 1], default=1)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--log_every', type=int, default=25)
    parser.add_argument('--weight', type=float, default=1)


    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    run(args)





















