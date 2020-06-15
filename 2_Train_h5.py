# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, RandomSampler
from torchvision import datasets, models, transforms, utils
import matplotlib.pyplot as plt
import time
import os  
import sys
import copy
import pandas as pd
import visdom
import random
import matplotlib.image as mpimg 
import warnings
warnings.filterwarnings("ignore")
import nrrd
import shutil
from datetime import datetime
from tensorboardX import SummaryWriter


import src.dataloader
import src.train3d

# -

# ## Create Dataset and DataLoader

# +
import src.dataloader
import importlib
importlib.reload(src.dataloader)

augmentor =  transforms.Compose([
    transforms.Lambda(lambda x: x.repeat(3, 1, 1, 1).permute(3, 0, 1, 2)),
    src.dataloader.Resize(256), 
])
    

path = '/data/larson2/RCC_dl/new/clear_cell'
train_ds = src.dataloader.RCCDataset_h5(path, mode='train', transform=augmentor)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4, drop_last=False)

val_ds = src.dataloader.RCCDataset_h5(path, mode='val', transform=augmentor)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=True, num_workers=4, drop_last=False)



print(train_ds[0][0].shape)
print(val_ds[0][0].shape)



# + [markdown] heading_collapsed=true
# ### Preview Image

# + hidden=true
array = train_ds[0][0]
num_im = array.shape[0]
list_show = np.linspace(0, num_im-1, 8)

fig, axes = plt.subplots(ncols=8, nrows=1, figsize=(30, 4), sharex=True, sharey=True)
for index, i in enumerate(list_show):
    slice_num = int(i)
    im = array[slice_num, 0, :,:].squeeze()
#     im = array[:,:,slice_num].squeeze()


    axes[index].imshow(im, vmin=-160, vmax=240, cmap=plt.cm.gray)
    axes[index].set_title('slice: {}'.format(slice_num))
#     fig.suptitle('TITLE')
    fig.tight_layout()
    fig.subplots_adjust(top = 0.9, bottom=0.1)
#     plt.colorbar()
plt.show()
plt.hist(array.flatten(), 100)
plt.show()
# -

# ## Training

# ### Parameters

# +
device = torch.device("cpu")
prefix_name='test_cpu'
# task='abnormal'
# plane='sagittal'
augment=1
lr_scheduler='plateau'
gamma=0.8
epochs=25
lr=1e-4
flush_history=0
save_model=1
patience=25
log_every=25


log_root_folder = "/data/larson2/RCC_dl/logs/"
now = datetime.now()
logdir = log_root_folder + "test"
# logdir = log_root_folder + now.strftime("%Y%m%d-%H%M%S") + "/"
# os.makedirs(logdir)

model_root_dir = "/data/larson2/RCC_dl/models/"

writer = SummaryWriter(logdir)
         
# -

# ### Initialize 

# #### resnet

model = models.alexnet(pretrained=True)
# print(list(list(model.classifier.children())[1].parameters()))
mod = list(model.classifier.children())
mod.pop()
mod.append(torch.nn.Linear(4096, 2))
new_classifier = torch.nn.Sequential(*mod)
# print(list(list(new_classifier.children())[1].parameters()))
model.classifier = new_classifier
model.classifier
# m.classifier

model = models.alexnet(pretrained=True)
# print(list(list(model.classifier.children())[1].parameters()))
mod = list(model.classifier.children())
mod.pop()
mod.append(torch.nn.Linear(4096, 2))
new_classifier = torch.nn.Sequential(*mod)
# print(list(list(new_classifier.children())[1].parameters()))
model.classifier = new_classifier
model
# m.classifier

list(model.classifier.children()).pop()

# #### mrnet

# +
mrnet = src.model.MRNet()
mrnet = mrnet.to(device)
optimizer = optim.Adam(mrnet.parameters(), lr=lr, weight_decay=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=gamma)

best_val_loss = float('inf')
best_val_auc = float(0)

num_epochs = epochs
iteration_change_loss = 0
patience = patience
log_every = log_every
t_start_training = time.time()
# -

mrnet.classifier

# + [markdown] heading_collapsed=true
# ### Train

# + hidden=true
for epoch in range(epochs):
    current_lr = src.train3d.get_lr(optimizer)

    t_start = time.time()

    train_loss, train_auc = src.train3d.train_model(
        mrnet, train_loader, device, epoch, num_epochs, optimizer, writer, current_lr, log_every)
    val_loss, val_auc = src.train3d.evaluate_model(
        mrnet, val_loader, device, epoch, num_epochs, writer, current_lr)


    if lr_scheduler == 'plateau':
        scheduler.step(val_loss)
    elif lr_scheduler == 'step':
        scheduler.step()

    t_end = time.time()
    delta = t_end - t_start

    print("train loss : {0} | train auc {1} | val loss {2} | val auc {3} | elapsed time {4} s".format(
        train_loss, train_auc, val_loss, val_auc, delta))

    iteration_change_loss += 1
    print('-' * 30)

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        if bool(save_model):
            file_name = f'model_{prefix_name}_val_auc_{val_auc:0.4f}_train_auc_{train_auc:0.4f}_epoch_{epoch+1}.pth'
            for f in os.listdir(model_root_dir):
                if  (prefix_name in f):
                    os.remove(os.path.join(model_root_dir, f))
            torch.save(mrnet, os.path.join(model_root_dir, file_name))

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        iteration_change_loss = 0

    if iteration_change_loss == patience:
        print('Early stopping after {0} iterations without the decrease of the val loss'.
              format(iteration_change_loss))
        break

t_end_training = time.time()
print(f'training took {t_end_training - t_start_training} s')

# + hidden=true


# + hidden=true


# + hidden=true


# + hidden=true


# + hidden=true


# + hidden=true


# + hidden=true


# + hidden=true

