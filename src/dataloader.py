
from __future__ import print_function, division
import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import functional as F
# from torchvision.transforms.functional import .to_tensor(pic)
# [source]

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion

import nrrd

from skimage.transform import resize
# from skimage import transform, io

class RCCDataset(Dataset):
    """RCC Dataset"""
    
    def __init__(self, data_path, transform=None):
        """
        Args: 
            labels: patient index and and label, pandas dataframe
            image_dir: 
            
        """
        data = pd.read_csv(data_path, squeeze=True)
        data = data.tolist()
        self.list_img = data
        self.transform = transform
          

    
    def __len__(self): 
        return len(self.list_img)
    
    def __getitem__(self,index):
        filename = self.list_img[index]
        # name = filename.split('/')
        # name = name[-1]
        # name = name.split('.')
        # name = name[0]
        # name = name.split('_')
        # PatientID = name[1]
        # SliceID = name[2]
        
        readdata, header = nrrd.read(filename)
        
        # header['PatientID'] = PatientID
        # header['SliceID'] = SliceID
        
        label = header['label']
        label = int(label)
        sample = (readdata, label, header)
        if self.transform:
            readdata, label, header = self.transform(sample)
#         print(header)
        return (readdata,label, header)


class ToRGB(object):
    def __call__(self, sample):
        im, label, header = sample[0], sample[1], sample[2]
#         im = torch.from_numpy(im)
        im_RGB = np.stack((im,im,im), axis=-1)
        return (im_RGB, label, header)

class Rescale():
    def __call__(self, sample):
        im, label, header = sample[0], sample[1], sample[2]
        # Rescale from [-1024, 3071] to [0,1]
        img = (im + 1024) / 4096

        return (img, label, header)

class ToPIL(object):
    """
    Convert
    """

    def __call__(self, sample):
        im, label, header = sample[0], sample[1], sample[2]

        # Input : RGB array
        # convert from [0 1] tp [0 255]
        # im = im * 255
        # pic = Image.fromarray(np.uint8(im))
        pic = Image.fromarray(im)
        # print('after ToPIL, shape = ', pic.size)
        return (pic, label, header)

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        im, label, header = sample[0], sample[1], sample[2]

        tsfm = transforms.Resize(self.size)
        img = tsfm(im)

        return (img, label, header)


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        im, label, header = sample[0], sample[1], sample[2]

        tsfm = transforms.RandomCrop(self.size)
        img = tsfm(im)
        # print('after RandomCrop, shape = ', img.size)

        return (img, label, header)

class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        im, label, header = sample[0], sample[1], sample[2]

        tsfm = transforms.CenterCrop(self.size)
        img = tsfm(im)

        return (img, label, header)


class RandomResizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        im, label, header = sample[0], sample[1], sample[2]

        tsfm = transforms.RandomResizedCrop(self.size)
        img = tsfm(im)
        # print('after RandomResizedCrop, shape = ', img.size)

        return (img, label, header)



class ToTensor(object):
    def __call__(self, sample):
        im, label, header = sample[0], sample[1], sample[2]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img =transforms.functional.to_tensor(im)
        # print('after ToTensor, shape = ', img.size())

        return (img, label, header)


class Normalize(object):
    # for transfer learning from ImageNet dataset:

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):


        im, label, header = sample[0], sample[1], sample[2]

        im = F.normalize(im, self.mean, self.std)

        # im_normalized = np.zeros(im.shape)
        # im_normalized = torch.tensor(im_normalized)
        # for i in range(3):
        #     C = im[i,:,:]
        #     C = (C-mean[i]) / std[i]
        #     im_normalized[i,:,:] = C

        return (im, label, header)




#
#
# class Resize(object):
#     def __init__(self, output_size):
# #         assert isinstance(output_size, (int, tuple))
#         self.output_size = output_size
#
#     def __call__(self, sample):
#         im, label, header = sample[0], sample[1], sample[2]
#         h = self.output_size
#         w = self.output_size
# #         im_resize = transform.resize(im, (h,w))
#         im_resize = resize(im, (h,w),preserve_range=True)
#         sample = (im_resize, label, header)
#         return sample
#
#
# class RandomFlipLR():
#     def __call__(self, sample):
#         im, label, header = sample[0], sample[1], sample[2]
#         import random
#         p = random.uniform(0, 1)
#         if p > 0.5:
#             im = np.fliplr(im)
#         return (im, label, header)
#
# class RandomFlipUD():
#     def __call__(self, sample):
#         im, label, header = sample[0], sample[1], sample[2]
#         import random
#         p = random.uniform(0, 1)
#         if p > 0.5:
#             im = np.flipud(im)
#         return (im, label, header)
#
#
# class RandomCrop():
#     def __init__(self, output_size):
# #         assert isinstance(output_size, (int, tuple))
#         self.output_size = output_size
#
#     def __call__(self, sample):
#         im, label, header = sample[0], sample[1], sample[2]
# #         print('here', im.shape)
#         h,w = im.shape[0], im.shape[1]
#         new_h = self.output_size
#         new_w = self.output_size
#
#         top = np.random.randint(0, h - new_h)
#         left = np.random.randint(0, w - new_w)
# #         print('top:', top )
# #         print('left: ', left)
#         im2 = im[top:top+ new_h, left:left + new_w]
#
#         return (im2, label, header)
#
#
#
# class ZeroPad():
#     def __init__(self, output_size=224):
#         self.output_size = output_size
#
#     def __call__(self, sample):
#         im, label,header= sample[0], sample[1], sample[2]
#
#         def padwith0(vector, pad_width, iaxis, kwargs):
#             vector[:pad_width[0]] = 0.0
#             vector[-pad_width[1]:] = 0.0
#             return vector
#         pad_width = int((self.output_size - im.shape[0]) /2)
#         im_pad = np.pad(im, pad_width, padwith0)
#         return (im_pad, label, header)
#
#
# class RGB2Gray(object):
#     def __call__(self, sample):
#         im, label, header = sample[0], sample[1], sample[2]
#         im_gray = np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
#
#         return (im_gray, label, header)

    
def load_data(train_path, val_path, transforms, batch_size ):
    
    image_path = {'train': train_path, 'val': val_path}
    image_datasets = {x:RCCDataset(image_path[x], transform=transforms[x])
                     for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, 
                                 shuffle=True, num_workers=4)
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    return dataloaders, dataset_sizes