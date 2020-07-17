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

from sklearn import metrics
from sklearn.metrics import confusion_matrix


import src.dataloader 
import src.train3d
import src.model
import src.helper

def run(args):
    print('Task 1: clear cell grade prediction')
    path = '/data/larson2/RCC_dl/new/clear_cell/'
    
    
    transform =  {'train': transforms.Compose([
        transforms.Lambda(lambda x: torch.Tensor(x)),
        src.dataloader.Rescale(-160, 240, zero_center=True), # rset dynamic range
        transforms.Lambda(lambda x: x.repeat(3, 1, 1, 1).permute(3, 0, 1, 2)),
    #     src.dataloader.Normalize(),
    #     src.dataloader.Crop(110),
    #     src.dataloader.RandomCenterCrop(90),
        src.dataloader.RandomHorizontalFlip(),
    #     src.dataloader.RandomRotate(25),
        src.dataloader.Resize(256)]), 

                  'val': transforms.Compose([
        transforms.Lambda(lambda x: torch.Tensor(x)),
        src.dataloader.Rescale(-160, 240, zero_center=True), # rset dynamic range
        transforms.Lambda(lambda x: x.repeat(3, 1, 1, 1).permute(3, 0, 1, 2)),
    #       src.dataloader.Normalize(),
    #       src.dataloader.Crop(90),
        src.dataloader.Resize(256)])
                 }

    my_dataset = {'train':src.dataloader.RCCDataset_h5(path, mode='train', transform=transform['train']), 
                 'val':  src.dataloader.RCCDataset_h5(path, mode='val', transform=transform['train'])}

    my_loader = {x: DataLoader(my_dataset[x], batch_size=1, shuffle=True, num_workers=4) 
                 for x in ['train', 'val']}

    print('train size: ', len(my_loader['train']))
    print('train size: ', len(my_loader['val']))


    ### Some Checkers
    print('Summary: ')
    print('\ttrain size: ', len(my_loader['train']))
    print('\ttrain size: ', len(my_loader['val']))
    print('\tDatatype = ', next(iter(my_loader['train']))[0].dtype)
    print('\tMin = ', next(iter(my_loader['train']))[0].min())
    print('\tMax = ', next(iter(my_loader['train']))[0].max())
    print('\tInput size', next(iter(my_loader['train']))[0].shape)
#     print('\tweight = ', args.weight)
    
    
    ### Tensorboard Log Setup 
    log_root_folder = "/data/larson2/RCC_dl/logs/"
    now = datetime.now()
    now = now.strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join(log_root_folder , f"{now}_model_{args.model}_{args.prefix_name}_epoch_{args.epochs}_weight_{args.weight}_lr_{args.lr}_gamma_{args.gamma}_lrsche_{args.lr_scheduler}_{now}")
#     os.makedirs(logdir)
    print(f'\tlogdir = {logdir}')
    
    writer = SummaryWriter(logdir)
    
    ### Model Selection
    
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

    model = src.model.TDNet() 
    model = model.to(device)
    
    writer.add_graph(model, my_dataset['train'][0][0].to(device))
    
    print('\tCuda:', torch.cuda.is_available(), f'\n\tdevice = {device}')
        
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.1)

    if args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=.3, threshold=1e-4, verbose=True)
    elif args.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=3, gamma=args.gamma)
        
    pos_weight = torch.FloatTensor([args.weight]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    
    ### Ready?
    best_val_loss = float('inf')
    best_val_auc = float(0)
    best_model_wts = copy.deepcopy(model.state_dict())
    iteration_change_loss = 0
    t_start_training = time.time()
    
    
    
    
    ### Here we go
    for epoch in range(args.epochs):
        current_lr = get_lr(optimizer)
        t_start = time.time()
        
        epoch_loss = {'train':0., 'val':0.}
        epoch_corrects = {'train':0., 'val':0.}


        epoch_acc = 0.0
        epoch_AUC = 0.0
        
        for phase in ['train', 'val']:
            if phase == 'train':
                if args.lr_scheduler == "step":
                    scheduler.step()
                model.train()
            else:
                model.eval()

            running_losses = []
            running_corrects = 0.
            y_trues = []
            y_probs = []
            y_preds = []
            
            print('lr: ', current_lr)
            for i, (inputs, labels, header) in enumerate(my_loader[phase]):
                optimizer.zero_grad()
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # forward
                # track history only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs.float()) # raw logits
                    probs = torch.sigmoid(outputs) # [0, 1] probability, shape = s * 1
                    preds = torch.round(probs) # 0 or 1, shape = s * 1, prediction for each slice
                    pt_pred, _ = torch.mode(preds, 0) # take majority vote, shape = 1, prediction for each patient
                    
                    count0 = (preds==0).sum().float()
                    count1 = (preds==1).sum().float()
                    pt_prob = count1/(preds.shape[0])

                    # convert label to slice level
                    loss = criterion(outputs, labels.repeat(inputs.shape[1], 1)) # inputs shape = 1*s*3*256*256
                    
                    # backward + optimize only if in training phases
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                
                # multiple loss by slice num per batch? 
                running_losses.append(loss.item()) # * inputs.size(0) 
                running_corrects += torch.sum(preds == labels.data)
                
                y_trues.append(int(labels.item()))
                y_probs.append(pt_prob.item()) # use ratio to get probability
                y_preds.append(pt_pred.item())
    
                writer.add_scalar(f'{phase}/Loss', loss.item(), epoch * len(my_loader[phase]) + i)
                writer.add_pr_curve('{phase}pr_curve', y_trues, y_probs, 0)


                
                if (i % args.log_every == 0) & (i > 0):
                    print('Epoch: {0}/{1} | Single batch number : {2}/{3} | avg loss:{4} | Acc: {5:.4f} | lr: {6}'.format(
                        epoch+1, 
                        args.epochs, 
                        i,
                        len(my_loader[phase]), 
                        np.round(np.mean(running_losses), 4), 
                        (running_corrects/len(my_loader[phase])), 
                        current_lr))

        
        
            # epoch statistics    
            epoch_loss[phase] = np.round(np.mean(running_losses), 4)
            epoch_corrects[phase] = (running_corrects/len(my_loader[phase]))
            
            cm = confusion_matrix(y_trues, y_preds, labels=[0, 1])
            src.helper.print_cm(cm, ['0', '1'])
            sens, spec, acc = src.helper.compute_stats(y_trues, y_preds )
            print('sens: {:.4f}'.format(sens))
            print('spec: {:.4f}'.format(spec))
            print('acc:  {:.4f}'.format(acc))
            print()
            
            
        print('\ Summary  train loss: {0} | val loss: {1} | train acc: {2:.4f} | val acc: {3:.4f}'.format(
            epoch_loss['train'], epoch_loss['val'], epoch_corrects['train'], epoch_corrects['val']))
        print('-' * 30)


#             epoch_acc = running_corrects.double() / dataset_sizes[phase]

#             writer.add_scaler()
#             loss_curve[phase].append(epoch_loss)
#             acc[phase].append(epoch_acc.cpu().numpy())


#             # deep copu the model
#             if phase == 'val' and epoch_acc > best_acc:
#                 best_acc = epoch_acc
#                 best_model_wts = copy.deepcopy(model.state_dict())
#                 best_epoch = epoch


                
                



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
               
            
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default=None)
    parser.add_argument('--prefix_name', type=str, required=True)

    parser.add_argument('--model', type=int, default=0)


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