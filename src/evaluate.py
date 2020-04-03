from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from sklearn.metrics import confusion_matrix
import visdom

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def visualize_model(model, dataloader, device, output_path, num_images=8):
    was_training = model.training
    model.eval()
    images_so_far = 0
    #     fig = plt.figure()

    with torch.no_grad():

        for i, (inputs, labels, header) in enumerate(dataloader['val']):
            inputs = inputs.to(device, dtype=torch.float)
            labels = torch.from_numpy(np.asarray(labels, dtype='int64'))
            labels = labels.to(device)
            print('input.requires_grad =', inputs.requires_grad)
            # compute saliency map
            inputs.requires_grad = True
            torch.set_grad_enabled(True)
            outputs = model(inputs)

            _, preds = torch.max(outputs, 1)
            print(inputs.requires_grad)

            batch_size = inputs.size()[0]
            col = 2
            row = batch_size
            fig, ax = plt.subplots(row, col, figsize=(col * 2, row * 2))

            for j in range(inputs.size()[0]):
                images_so_far += 1

                inp = inputs.cpu().data[j]
                inp = inp.numpy().transpose((1, 2, 0))
                inp = rgb2gray(inp)
                ax[j, 0].imshow(inp, cmap=plt.cm.bone, vmin=-2, vmax=2)
                ax[j, 0].axis('off')
                ax[j, 0].set_title('predicted: {}, label:{}'.format(preds[j], labels[j]))

                # ----
                outputs[j, preds[j]].backward(retain_graph=True)
                T = inputs.grad
                T = T.cpu().data[j]
                T = np.abs(T)
                T = T.numpy().transpose((1, 2, 0))
                T = rgb2gray(T)
                #                 T = T[T[0::][:,0:]>0.3]
                ma = T.max()
                mi = T.min()

                ax[j, 1].imshow(T, plt.get_cmap('hot'))
                ax[j, 1].axis('off')
                ax[j, 1].set_title('predicted: {}, label:{}'.format(preds[j], labels[j]))

                inputs.grad = None
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, 'saliency{}.png'.format(i)))
            if i == 2:
                return
        model.train(mode=was_training)
