import shutil
import os
import time
from datetime import datetime
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
# from torchsample.transforms import RandomRotate, RandomTranslate, RandomFlip, ToTensor, Compose, RandomAffine
from torchvision import transforms
import torch.nn.functional as F
from tensorboardX import SummaryWriter

# from src.dataloader import RCCDataset3D
import src.model

from sklearn import metrics
from sklearn.metrics import confusion_matrix



def train_model(model, train_loader, device, epoch, num_epochs, optimizer, writer, current_lr, log_every=100, weight = 1):
    _ = model.train()
    print('in train, weight = ', weight)
#     if torch.cuda.is_available():
#         model.cuda()
    model = model.to(device)
    
    y_preds = []
    y_trues = []
    losses = []
    y_argmax = []


    for i, (image, label, header) in enumerate(train_loader):
#         print('index = ', i)
        optimizer.zero_grad()
    
        image = image.to(device)
        label = label.to(device)
#         weight = weight.to(device)


        label = label[0]
#         weight = weight[0]

        prediction = model.forward(image.float())
#         loss = torch.nn.BCEWithLogitsLoss(weight=weight)(prediction, label)
        pos_weight = torch.FloatTensor([weight]).to(device)
        loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)(prediction, label)
#         loss = torch.nn.CrossEntropyLoss(weight=weight)(prediction, label)



        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        losses.append(loss_value)

        probas = torch.sigmoid(prediction)

        y_trues.append(int(label[0][1]))
        y_preds.append(probas[0][1].item())
        y_argmax.append(torch.argmax(probas).item())


        try:
            auc = metrics.roc_auc_score(y_trues, y_preds)
        except:
            auc = 0.5

        writer.add_scalar('Train/Loss', loss_value,
                          epoch * len(train_loader) + i)
        writer.add_scalar('Train/AUC', auc, epoch * len(train_loader) + i)

        if (i % log_every == 0) & (i > 0):
            print('''[Epoch: {0} / {1} |Single batch number : {2} / {3} ]| avg train loss {4} | train auc : {5} | lr : {6}'''.
                  format(
                      epoch + 1,
                      num_epochs,
                      i,
                      len(train_loader),
                      np.round(np.mean(losses), 4),
                      np.round(auc, 4),
                      current_lr
                  )
                  )
            
    cm = confusion_matrix(y_trues, y_argmax, labels=[0, 1])
    print_cm(cm, ['0', '1'])
    sens, spec, acc = computet_stats(y_argmax, y_trues)
    print('sens: {:.4f}'.format(sens))
    print('spec: {:.4f}'.format(spec))
    print('acc:  {:.4f}'.format(acc))
    print()


    writer.add_scalar('Train/AUC_epoch', auc, epoch + i)

    train_loss_epoch = np.round(np.mean(losses), 4)
    train_auc_epoch = np.round(auc, 4)
    return train_loss_epoch, train_auc_epoch


def evaluate_model(model, val_loader, device, epoch, num_epochs, writer, current_lr, log_every=20, weight=1):
    _ = model.eval()

    model = model.to(device)

    y_trues = []
    y_preds = []
    y_argmax = []
    losses = []

    for i, (image, label, header) in enumerate(val_loader):

        image = image.to(device)
        label = label.to(device)
#         weight = weight.to(device)


        label = label[0]
#         weight = weight[0]

        prediction = model.forward(image.float())

#         loss = torch.nn.BCEWithLogitsLoss(weight=weight)(prediction, label)
        pos_weight = torch.FloatTensor([weight]).to(device)
        loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)(prediction, label)
#         loss = torch.nn.CrossEntropyLoss(weight=weight)(prediction, label)



        loss_value = loss.item()
        losses.append(loss_value)

        probas = torch.sigmoid(prediction)

        y_trues.append(int(label[0][1]))
        y_preds.append(probas[0][1].item())
        y_argmax.append(torch.argmax(probas).item())


        try:
            auc = metrics.roc_auc_score(y_trues, y_preds)
        except:
            auc = 0.5

        writer.add_scalar('Val/Loss', loss_value, epoch * len(val_loader) + i)
        writer.add_scalar('Val/AUC', auc, epoch * len(val_loader) + i)

        if (i % log_every == 0) & (i > 0):
            print('''[Epoch: {0} / {1} |Single batch number : {2} / {3} ] | avg val loss {4} | val auc : {5} | lr : {6}'''.
                  format(
                      epoch + 1,
                      num_epochs,
                      i,
                      len(val_loader),
                      np.round(np.mean(losses), 4),
                      np.round(auc, 4),
                      current_lr
                  )
                  )

    cm = confusion_matrix(y_trues, y_argmax, labels=[0, 1])
    print_cm(cm, ['0', '1'])
    sens, spec, acc = computet_stats(y_argmax, y_trues)
    print('sens: {:.4f}'.format(sens))
    print('spec: {:.4f}'.format(spec))
    print('acc:  {:.4f}'.format(acc))
    print()

    
    writer.add_scalar('Val/AUC_epoch', auc, epoch + i)

    val_loss_epoch = np.round(np.mean(losses), 4)
    val_auc_epoch = np.round(auc, 4)
    return val_loss_epoch, val_auc_epoch

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
    
def computet_stats(pred_list, label_list):
    tp_idx = []
    fp_idx = []
    fn_idx = []
    tn_idx = []
    for i in range(len(label_list)):
    #     print(la[i])
        if label_list[i] == 1 and pred_list[i] == 1:
            tp_idx.append(i)
        elif label_list[i] == 0 and pred_list[i] == 0:
            tn_idx.append(i)
        elif label_list[i] == 1 and pred_list[i] == 0:
            fn_idx.append(i)
        elif label_list[i] == 0 and pred_list[i] == 1: 
            fp_idx.append(i)

    sens = len(tp_idx)/(len(tp_idx)+len(fn_idx))
    spec = len(tn_idx)/(len(tn_idx)+len(fp_idx))
    acc = (len(tn_idx) + len(tp_idx) )/(len(tn_idx) + len(tp_idx)+len(fn_idx) + len(fp_idx))
    return sens,spec,acc

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    (tn, fp, fn, tp) = cm.ravel()
    print('p/t     1     0')
    print('  1     {}    {}'.format(tp, fp))
    print('  0     {}    {}'.format(fn, tn))




