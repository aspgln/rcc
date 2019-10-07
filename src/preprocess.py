from __future__ import print_function, division

import os
import numpy as np
import pandas as pd
# from utils.train import *


def create_csv(data_dir):
    def get_patient_path(data_path):
        list_patient_PATH = []
        for i in os.scandir(data_path):
            list_patient_PATH.append(i.path)
        return list_patient_PATH

    def cv_patient(patient_path, cv):
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=cv, shuffle=True)
        idx = []
        x = np.arange(len(patient_path))
        for train_idx, val_idx in kf.split(x):
            idx.append({'train': train_idx, 'val': val_idx})
        return idx

    def get_slice_path(patient_path, idx):
        slice_path = []
        for index, split in enumerate(idx):
            temp = {'train': [], 'val': []}
            for x in ['train', 'val']:
                for i in split[x]:
                    #             print(i)
                    folder = patient_path[i]
                    #             print(folder)
                    for file in os.scandir(folder):
                        temp[x].append(file.path)
                np.random.shuffle(temp[x])
            slice_path.append(temp)
        return slice_path

    patient_path = get_patient_path(data_dir)
    patient_path = patient_path
    cv = 5

    idx = cv_patient(patient_path, cv)
    slice_path = get_slice_path(patient_path, idx)
    for i in range(cv):
        print(len(slice_path[i]['train']) ,
              len(slice_path[i]['val']))
    # print(slice_path[0]['train'])

    df = pd.DataFrame(slice_path[0]['train'])
    df.to_csv('./path/train_path.csv', index=False, header=False)
    df = pd.DataFrame(slice_path[0]['val'])
    df.to_csv('./path/val_path.csv', index=False, header=False)
