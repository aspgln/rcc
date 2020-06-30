
from __future__ import print_function, division
import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import functional as F
import random
import math
# from torchvision.transforms.functional import .to_tensor(pic)
# [source]
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
plt.ion
import nrrd
import h5py
from src.torchsample.transforms.tensor_transforms import RandomFlip

import src.torchsample.transforms.affine_transforms


# from skimage.transform import resize
# from skimage import transform, io


class RCCDataset_h5(Dataset):
    '''
    Dataset takes in two mandatory parameters, root_dir and mode, as well as two optiaonl parameter, transform and weights
    @root_dir: the input directory which contains imageing dataset and .csv files with pt_index and groundtruth label
    @mode: train or val or val, which directs to the according .csv file
    
    '''
    def __init__(self, root_dir, mode, transform=None, weights=None):
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        self.label_path = os.path.join(root_dir, mode+'.csv')
        self.records = pd.read_csv(self.label_path, header=None,  index_col=[0], names=['label']) # index = pt_index,  label = [0,1]
        self.index = self.records.index.tolist()
        self.labels = self.records['label'].tolist()
        self.paths = [os.path.join(root_dir, mode, x+'.hdf5') for x in self.index]
        
        if weights is None:
            pos = self.labels.count(1)
            neg = self.labels.count(0)
            if pos+neg != len(self.records):
                raise Exception('Error! pos + neg NOT equals to total!')
            self.weights = torch.FloatTensor([1, neg / pos])
        else:
            self.weights = torch.FloatTensor(weights)

    def __len__(self):
        return len(self.records)
    
    def __getitem__(self,index):
        with h5py.File(self.paths[index], mode='r', track_order=True) as f:
            # get label
            label = self.labels[index]
            if label == 1:
                label = torch.FloatTensor([1])
            elif label == 0:
                label = torch.FloatTensor([0])
                       
#             # get label, for BCELogitLoss
#             label = self.labels[index]
#             if label == 1:
#                 label = torch.FloatTensor([[0, 1]])
#             elif label == 0:
#                 label = torch.FloatTensor([[1, 0]])
            
            # get image array
            group = f['single_phase']
            array = group['data'][()]
            
            # convert to tensor
            array = torch.Tensor(array)
            if self.transform:
                array = self.transform(array)
            
            # header
            header = {}
            return (array, label, header)
        
        
class RCCDataset(Dataset):
    """RCC Dataset"""
    
    def __init__(self, list_img, transform=None):
        """
        Args: 
            labels: patient index and and label, pandas dataframe
            image_dir: 
            
        """
        self.list_img = list_img
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


    
class RCCDataset3D(Dataset):
    def __init__(self, root_dir, mode, transform=None, weights=None):
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        
        self.label_path = os.path.join(root_dir, mode+'_labels.csv')
        self.records = pd.read_csv(self.label_path, header=None,  index_col=[0], names=['label']) # all labels are here
        self.index = self.records.index.tolist()
        self.labels = self.records['label'].tolist()
        self.paths = [os.path.join(root_dir, mode, x+'.nrrd') for x in self.index]
        
        if weights is None:
            pos = self.labels.count(1)
            neg = self.labels.count(0)
            if pos+neg != len(self.records):
                raise Exception('Error! pos + neg NOT equals to total!')
            self.weights = torch.FloatTensor([1, neg / pos])
        else:
            self.weights = torch.FloatTensor(weights)

    def __len__(self): 
        return len(self.records)
    
    def __getitem__(self,index):
        array, header = nrrd.read(self.paths[index])


        label = self.labels[index]
        
        if label == 1:
            label = torch.FloatTensor([[0, 1]])
        elif label == 0:
            label = torch.FloatTensor([[1, 0]])


        if self.transform:
            array = self.transform(array)
        else:
            # to 3-channel
            array = np.stack((array,)*3, axis=0)
            array = np.moveaxis(array, -1, 0)
            
            # resize
#             array = torch.nn.functional.interpolate(array, 256)
            
            #rescale
            vmin = -150
            vmax=250
            array[array<vmin] = vmin
            array[array>vmax] = vmax
            array = (array+150) /400 *256
            
            # to 8-bit [0,255]
            array = np.uint8(array)
            
            array = torch.FloatTensor(array)
            # resize
            array = torch.nn.functional.interpolate(array, 256)
            
        
#         if label.item() == 1:
#             weight = np.array([self.weights[1]])
#             weight = torch.FloatTensor(weight)
#         else:
#             weight = np.array([self.weights[0]])
#             weight = torch.FloatTensor(weight)
    
        return (array, label,  header)

    
def get_paths(path, mode):
    paths=[]
    for x in os.scandir(path):
        for y in os.scandir(x.path):
            if  y.name == mode:
                for z in os.scandir(y.path):
                    paths.append(z.path)
    return paths

    
class RCCDataset3DTest(Dataset):
    def __init__(self, root_dir, mode, transform=None, weights=None):
        super().__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        
        self.paths = get_paths(root_dir, mode)
        self.pathology = [x.split('/')[-3] for x in self.paths]
        self.labels = [1 if x == 'papillary' else 0 for x in self.pathology]       
        self.records =  pd.DataFrame(list(zip(self.paths, self.pathology, self.labels)), columns=['paths', 'pathology', 'labels'])
        
#         if weights is None:
#             pos = self.labels.count(1)
#             neg = self.labels.count(0)
#             if pos+neg != len(self.records):
#                 raise Exception('Error! pos + neg NOT equals to total!')
#             self.weights = torch.FloatTensor([1, neg / pos])
#         else:
#             self.weights = torch.FloatTensor(weights)



    def __len__(self): 
        return len(self.paths)
    
    def __getitem__(self,index):
        array, header = nrrd.read(self.paths[index])

        label = self.labels[index]
        
        if label == 1:
            label = torch.FloatTensor([[0, 1]])
        elif label == 0:
            label = torch.FloatTensor([[1, 0]])

        if self.transform:
            array = self.transform(array)


        header['path'] = self.paths[index]
        return (array, label,  header)   
    
    
    
# class ToByte(object):
#     def __call__(self, array):
#         array = (array - array.min())
#         array = array * 255 / (array.max() - array.min())
#         array = array.type(torch.uint8)
#         return array

        
class Rescale(object):
    def __init__(self, vmin, vmax,zero_center=False, normalize=False):
        self.vmin = vmin
        self.vmax = vmax
        self.zero_center = zero_center
        self.normalize = normalize


    def __call__(self, array):
#         Rescale from [-1024, 3071] to [-vmin. vmax]

#         option 1: to [-160, 240]
        array[array<self.vmin] = self.vmin
        array[array>self.vmax] = self.vmax
        
        if self.zero_center == True:
            array = array - (self.vmax + self.vmin)/2
#         if self.normalize == True:
#             array = 
#         # option 2: zero-centered
#         array[array<self.vmin] = self.vmin
#         array[array>self.vmax] = self.vmax
#         array = (array - array.mean())
        
#         # option 3: to [0, 255]
#         array[array<self.vmin] = self.vmin
#         array[array>self.vmax] = self.vmax 
#         array = array - self.vmin
#         array = array * 255 / (self.vmax - self.vmin)

#         # option 4: to [0, 1]
#         array[array<self.vmin] = self.vmin
#         array[array>self.vmax] = self.vmax 
#         array = array - self.vmin
#         array = array / (self.vmax - self.vmin)


        # option 5
#         array = array/array.mean() + 3*array.std()
# #         vol[vol > 1] = 1
# #         vol[vol < 0] = 0 
        
        
        return array


class Normalize(object):
    def __call__(self, array):
        tsfm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        for i in range(array.shape[0]):
            temp = tsfm(array[i,:,:,:])
            if i == 0:
                array_tsfm = temp[None]
            else:
                array_tsfm = torch.cat((array_tsfm,temp[None]), dim=0)


        return array_tsfm

    
class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, array):
        
#         array = torch.FloatTensor(array)
        array = torch.nn.functional.interpolate(array, self.size)


        return array

 
 
class RandomRotate(object):
    '''
    input: s*c*h*w, floattensor
    output: s*c*h*w, FloatTensor
    '''
    def __init__(self, rotation_range):
        self.rotation_range = rotation_range

    def __call__(self, array):
        degree = random.uniform(-self.rotation_range, self.rotation_range)
        tsfm = src.torchsample.transforms.affine_transforms.Rotate(degree)

        for i in range(array.shape[0]):
            temp = tsfm(array[i,:,:,:])
            if i == 0:
                array_tsfm = temp[None]
            else:
                array_tsfm = torch.cat((array_tsfm,temp[None]), dim=0)

        return array_tsfm

    
class RandomHorizontalFlip(object):
    '''
    input: s*c*h*w, floattensor
    output: s*c*h*w, FloatTensor
    '''

    def __call__(self, array):
        
        p = random.random()
        
        if p > 0.5:
            tsfm = src.torchsample.transforms.tensor_transforms.RandomFlip(h=True, v=False, p=1)

            for i in range(array.shape[0]):
                temp = tsfm(array[i,:,:,:])
                if i == 0:
                    array_tsfm = temp[None]
                else:
                    array_tsfm = torch.cat((array_tsfm,temp[None]), dim=0)
            return array_tsfm
        
        else:
            return array


class Crop(object):
    '''
    input: s*c*h*w, floattensor
    output: s*c*h*w, FloatTensor
    crop a size*size area in the center
    '''
    def __init__(self, crop_size):
        self.crop_size = crop_size
        
    def __call__(self, array):
        tsfm = src.torchsample.transforms.tensor_transforms.SpecialCrop((self.crop_size, self.crop_size),crop_type=0)

        for i in range(array.shape[0]):
            temp = tsfm(array[i,:,:,:])
            if i == 0:
                array_tsfm = temp[None]
            else:
                array_tsfm = torch.cat((array_tsfm,temp[None]), dim=0)

        return array_tsfm

        
    
    
class RandomCenterCrop(object):
    '''
    input: s*c*h*w, floattensor
    output: s*c*h*w, FloatTensor
    crop a size*size area in the center
    '''
    def __init__(self, crop_size):
        self.crop_size = crop_size
        
    def __call__(self, array):
        actual_size = random.uniform(self.crop_size, array.size()[2])
        tsfm = src.torchsample.transforms.tensor_transforms.SpecialCrop((actual_size, actual_size),crop_type=0)

        for i in range(array.shape[0]):
            temp = tsfm(array[i,:,:,:])
            if i == 0:
                array_tsfm = temp[None]
            else:
                array_tsfm = torch.cat((array_tsfm,temp[None]), dim=0)

        return array_tsfm

#  ------------------------   
    
    
# class Rescale():
#     def __call__(self, sample):
#         im, label, header = sample[0], sample[1], sample[2]
#         # Rescale from [-1024, 3071] to [0,1]
#         img = (im + 1024) / 4096

#         return (img, label, header)

# class ToPIL(object):
#     """
#     Convert
#     """

#     def __call__(self, sample):
#         im, label, header = sample[0], sample[1], sample[2]

#         # Input : RGB array
#         # convert from [0 1] tp [0 255]
#         # im = im * 255
#         # pic = Image.fromarray(np.uint8(im))
#         pic = Image.fromarray(im)
#         # print('after ToPIL, shape = ', pic.size)
#         return (pic, label, header)

# class Resize(object):
#     def __init__(self, size):
#         self.size = size

#     def __call__(self, sample):
#         im, label, header = sample[0], sample[1], sample[2]

#         tsfm = transforms.Resize(self.size)
#         img = tsfm(im)

#         return (img, label, header)


# class RandomCrop(object):
#     def __init__(self, size):
#         self.size = size

#     def __call__(self, sample):
#         im, label, header = sample[0], sample[1], sample[2]

#         tsfm = transforms.RandomCrop(self.size)
#         img = tsfm(im)
#         # print('after RandomCrop, shape = ', img.size)

#         return (img, label, header)

# class CenterCrop(object):
#     def __init__(self, size):
#         self.size = size

#     def __call__(self, sample):
#         im, label, header = sample[0], sample[1], sample[2]

#         tsfm = transforms.CenterCrop(self.size)
#         img = tsfm(im)

#         return (img, label, header)


# class RandomResizedCrop(object):
#     def __init__(self, size):
#         self.size = size

#     def __call__(self, sample):
#         im, label, header = sample[0], sample[1], sample[2]

#         tsfm = transforms.RandomResizedCrop(self.size)
#         img = tsfm(im)
#         # print('after RandomResizedCrop, shape = ', img.size)

#         return (img, label, header)



# class ToTensor(object):
#     def __call__(self, sample):
#         im, label, header = sample[0], sample[1], sample[2]

#         # swap color axis because
#         # numpy image: H x W x C
#         # torch image: C X H X W
#         img =transforms.functional.to_tensor(im)
#         # print('after ToTensor, shape = ', img.size())

#         return (img, label, header)


# class Normalize(object):
#     # for transfer learning from ImageNet dataset:

#     def __init__(self, mean, std):
#         self.mean = mean
#         self.std = std

#     def __call__(self, sample):


#         im, label, header = sample[0], sample[1], sample[2]

#         im = F.normalize(im, self.mean, self.std)

#         # im_normalized = np.zeros(im.shape)
#         # im_normalized = torch.tensor(im_normalized)
#         # for i in range(3):
#         #     C = im[i,:,:]
#         #     C = (C-mean[i]) / std[i]
#         #     im_normalized[i,:,:] = C

#         return (im, label, header)





    
def load_data(train_path, val_path, transforms, batch_size ):
    
    image_path = {'train': train_path, 'val': val_path}
    image_datasets = {x:RCCDataset(image_path[x], transform=transforms[x])
                     for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, 
                                 shuffle=True, num_workers=4)
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    return dataloaders, dataset_sizes