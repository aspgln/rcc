from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, RandomSampler
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



import src.dataloader 
import src.train3d
import src.model



def run(args):
    
    
    
# task='abnormal'
# plane='sagittal'
# augment=1
# lr_scheduler='step'
# gamma=0.8
# epochs=20
# lr=1e-4
# flush_history=0
# save_model=1
# patience=5
# log_every=100
# prefix_name='test'
    
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")


    augmentor =  transforms.Compose([
        transforms.Lambda(lambda x: torch.Tensor(x)),
        src.dataloader.Rescale(-160, 240), # rset dynamic range
        transforms.Lambda(lambda x: x.repeat(3, 1, 1, 1).permute(3, 0, 1, 2)),
#         src.dataloader.Normalize(), 
        src.dataloader.Crop(90), 
#         src.dataloader.RandomCenterCrop(90), 

        src.dataloader.RandomHorizontalFlip(), 
        src.dataloader.RandomRotate(25), 
        src.dataloader.Resize(256), 
    ])
    
    augmentor2 =  transforms.Compose([
        transforms.Lambda(lambda x: torch.Tensor(x)),
        src.dataloader.Rescale(-160, 240), # rset dynamic range
        transforms.Lambda(lambda x: x.repeat(3, 1, 1, 1).permute(3, 0, 1, 2)),
#         src.dataloader.Normalize(), 
        src.dataloader.Crop(90), 
        src.dataloader.Resize(256), 
    ])
    
    
    if args.task == 1:
        print('Task 1: clear cell grade prediction')
        path = '/data/larson2/RCC_dl/7.3D_clearcell/'
    
        train_ds = src.dataloader.RCCDataset3D(path, mode='train', transform=augmentor)
        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=16, drop_last=False)

        val_ds = src.dataloader.RCCDataset3D(path, mode='test', transform=augmentor2)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=True, num_workers=4, drop_last=False)
        
        pos_weight = args.weight
    
    elif args.task == 2:
        print('Task 2: differentiate between oncocytoma vs papilary RCC')
        
        path = '/data/larson2/RCC_dl/7.3D_onc_papillary/'

        train_ds = src.dataloader.RCCDataset3DTest(path, mode='train', transform=augmentor)
        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=16, drop_last=False)

        val_ds = src.dataloader.RCCDataset3DTest(path, mode='val', transform=augmentor2)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=True, num_workers=4, drop_last=False)
        pos_weight = args.weight





    
    print('Path = ', path)
    print('Datatype = ', train_ds[1][0].dtype)
    print('Min = ', train_ds[1][0].min())
    print('Max = ', train_ds[1][0].max())
    print('Input size', train_ds[0][0].shape)
    

#     task='abnormal'
#     plane='sagittal'
#     augment=1
#     lr_scheduler='plateau'
#     gamma=0.5
#     epochs=20
#     lr=1e-5
#     flush_history=0
#     save_model=1
#     patience=5
#     log_every=100
#     prefix_name='test'

    log_root_folder = "./logs/"
#         if args.flush_history == 1:
#         objects = os.listdir(log_root_folder)
#         for f in objects:
#             if os.path.isdir(log_root_folder + f):
#                 shutil.rmtree(log_root_folder + f)


    now = datetime.now()
#     logdir = log_root_folder + "test"
    logdir = os.path.join(log_root_folder , args.prefix_name+now.strftime("%Y%m%d-%H%M%S") + "/")
    os.makedirs(logdir)
    
    
    writer = SummaryWriter(logdir)

    model = src.model.MRNet()
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.1)

    if args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=.3, threshold=1e-4, verbose=True)
    elif args.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=3, gamma=args.gamma)
        
        
    best_val_loss = float('inf')
    best_val_auc = float(0)

    num_epochs = args.epochs
    iteration_change_loss = 0
    patience = args.patience
    log_every = args.log_every
    t_start_training = time.time()
    weight = args.weight
    print('weight = ', weight)
    for epoch in range(num_epochs):
        current_lr = src.train3d.get_lr(optimizer)

        t_start = time.time()

        train_loss, train_auc = src.train3d.train_model(
            model, train_loader, device, epoch, num_epochs, optimizer, writer, current_lr, log_every, weight)
        val_loss, val_auc = src.train3d.evaluate_model(
            model, val_loader, device, epoch, num_epochs, writer, current_lr,log_every,  weight)

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

        if val_auc > best_val_auc and epoch > 5:
            best_val_auc = val_auc
            if bool(args.save_model):
                file_name = f'model_{args.task}_{args.prefix_name}_val_auc_{val_auc:0.4f}_train_auc_{train_auc:0.4f}_epoch_{epoch+1}.pth'
#                 for f in os.listdir('/working/larson/qdai/RCC/repo/models/'):
#                     if (args.task in f) :
#                         os.remove(f'/working/larson/qdai/RCC/repo/models/{f}')
                torch.save(model, f'/working/larson/qdai/RCC/repo/models/{file_name}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            iteration_change_loss = 0

        if iteration_change_loss == patience:
            print('Early stopping after {0} iterations without the decrease of the val loss'.
                  format(iteration_change_loss))
            break
            
        
    t_end_training = time.time()
    print(f'training took {t_end_training - t_start_training} s')
    #     --------------------
   



#     task='abnormal'
#     plane='sagittal'
#     augment=1
#     lr_scheduler='plateau'
#     gamma=0.5
#     epochs=20
#     lr=1e-5
#     flush_history=0
#     save_model=1
#     patience=5
#     log_every=100
#     prefix_name='test'



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--prefix_name', type=str, required=True)
    parser.add_argument('--task', type=int, required=True)



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





















