from __future__ import print_function, division

import os
import numpy as np
import pandas as pd
import json
from pandas.io.json import json_normalize


import pydicom
import nrrd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shutil
from matplotlib import cm


# from utils.train import *



def load_log(PATH1, PATH2):
    '''
    load and combine AIR log from Sage (for MD.ai) and Kirti (AIR Retrieval)
    '''
    with open(PATH1) as f:
        #sage
        data1 = pd.read_csv(PATH1) 
        data1 = data1[['Orig Acc #','Orig Study UID',
                       'Anon Study UID',
                       'Image Req. Status',
                      'Anon Patient Name']]
    data1 = data1.dropna(subset=['Orig Acc #', 'Orig Study UID'])
    data1['Orig Acc #'] = data1['Orig Acc #'].astype('int64')
    
#     path2 = './metadata/kirti_all.csv'
    with open(PATH2) as f:
        #kirti
        data2 = pd.read_csv(PATH2)
        data2 = data2[['Orig Acc #',
                       'Orig Study UID',
                       'Anon Study UID',
                       'Image Req. Status'
                      ]]
    data2 = data2.dropna(subset=['Orig Acc #', 'Orig Study UID'])
    data2['Orig Acc #'] = data2['Orig Acc #'].astype('int64')

    data_log = pd.merge(data1, data2, how='inner', on=['Orig Study UID', 'Orig Acc #', 'Image Req. Status'])
    data_log = data_log[data_log['Image Req. Status']=='TRANSFERRED'] 

    return data_log


def load_json(PATH):
    '''load json from md.ai (annotations)'''
    
    with open(PATH) as f:
        js = json.load(f)
    json_data = json_normalize(js['datasets'][0]['annotations'])
    json_data_2 = json_data[['labelId','StudyInstanceUID','SeriesInstanceUID', 'SOPInstanceUID', 'data.vertices','data.height','data.width', 'data.x', 'data.y']]
    BBId =['L_y7KqAJ', # R_arterial
    'L_e7kVd1' , # L_arterial
    'L_4lgMO1' , # R_pv
    'L_q1Mrp7', # L_pv
    'L_e7D5Yl', # R_noncon
    'L_47EP91', # R_delay
    'L_PJR807' , # L_noncon
    'L_gljGOJ' , # L_delay
#     'L_k1zEyJ', # prone, series label
#     'L_97NAN1', # stacked, series label
           # NOTE: series label may overmap with image label, and confuses in find_series and find_slices
          ]
    
    data_annotation = []
    for i in range(len(BBId)):
    #     print(i)
    #     if i>7:
    #         break
        temp = json_data_2[json_data_2['labelId'].str.contains(BBId[i], na=False)]
        if i == 0:    
            data_annotation = temp
        else:
             data_annotation = pd.concat([data_annotation, temp], ignore_index=True)   
    #     print(data_annotation.shape)

    return data_annotation
 
def load_redcap(PATH):
    
    data_clinical = pd.read_csv(PATH)                             
    # data_clinical = data[['accession', 'biopsy_type', 'tumor_type','pathology', 'grade']]
    # drop empty record
    data_clinical = data_clinical.dropna(subset=['accession'])
    
    # create new df to seperate fields with 2 acc #
    new_df = data_clinical[['record_id', 'accession']]
    new_df = pd.DataFrame(new_df.accession.str.split(',').tolist(), index=new_df.record_id).stack()
    new_df = new_df.reset_index([0,'record_id'])
    new_df.columns = ['record_id', 'unique_acc']
    # drop acc# 11568327 for duplicate
    new_df = new_df.drop_duplicates(subset=['unique_acc'])

    data_clinical = pd.merge(data_clinical, new_df, left_on=['record_id'], 
                          right_on=['record_id'])
    data_clinical = data_clinical.drop(['accession'], axis=1)
    data_clinical['unique_acc'] = pd.to_numeric(data_clinical['unique_acc'])
    
    return data_clinical
    
def check_series_UID(SERIES_PATH, series_array):
    'check if a SERIES_PATH is in a list of series_array' 
    '''series_array is a dataframe'''
    fn = os.listdir(SERIES_PATH)[0]
    series_UID = pydicom.dcmread(os.path.join(SERIES_PATH, fn)).SeriesInstanceUID
    if series_UID in series_array.values:
        return True
    else:
        return False

def find_series(PATH, UIDs):
    folders = []
    for index, study_entry in enumerate(os.scandir(PATH)):
        for series_entry in os.scandir(study_entry.path):
            if check_series_UID(series_entry.path, UIDs['SeriesInstanceUID']):
                folders.append(series_entry.path)
#                 break
            else:
                pass
        print('found: ',len(folders))
    return folders

def find_slices(series_path, UIDs):

    files = os.listdir(series_path)
    slice_path = []
    for filename in files:
        ds = pydicom.dcmread(os.path.join(series_path, filename))
        if ds.SOPInstanceUID in UIDs['SOPInstanceUID'].values:
            slice_number = ds.InstanceNumber
            # pass in a tuple(slice_path, slice_number)
            tup = (os.path.join(series_path , filename), slice_number)
            slice_path.append(tup)
        else:
            pass
    return slice_path


def df_contains(df, column, string):
    df2 = df[df[column].str.contains(string)]
    return df2

def HU_rescale(im, slope, intercept):
    """
    Convert store value to HU value
    Linear correlation:
        Output Value = SV * RescaleSlope + RescaleIntercept
    """
    return im * slope + intercept



# def check_phase(labelId):
#     if labelId.values == 'L_e7D5Yl' or labelId.values == 'L_PJR807':
#         phase = 0 # non-contract phase
#     elif labelId.values == 'L_y7KqAJ' or labelId.values == 'L_e7kVd1':
#         phase = 1 # arterial phase
#     elif labelId.values ==  'L_q1Mrp7' or labelId.values == 'L_4lgMO1':
#         phase =2 # portal venous phase
#     elif labelId.values == 'L_gljGOJ' or labelId.values =='L_47EP91':
#         phase = 3 # delayed phase
#     else:
#         print('UNKNOWN PHASE!!!')
#     return phase


def write_nrrd(im_PATH, output_PATH, UIDs, df):
    
    
    def check_phase(labelId):
        if labelId.values == 'L_e7D5Yl' or labelId.values == 'L_PJR807':
            phase = 0 # non-contract phase
        elif labelId.values == 'L_y7KqAJ' or labelId.values == 'L_e7kVd1':
            phase = 1 # arterial phase
        elif labelId.values ==  'L_q1Mrp7' or labelId.values == 'L_4lgMO1':
            phase =2 # portal venous phase
        elif labelId.values == 'L_gljGOJ' or labelId.values =='L_47EP91':
            phase = 3 # delayed phase
        else:
            print('UNKNOWN PHASE!!!')
            print(labelId.values)
        return phase

               
    slices = []

    try:
        os.makedirs(output_PATH)
    except FileExistsError:
        pass

    counter = 0
    for index, studyEntry in enumerate(os.scandir(im_PATH)):
        # check if study UID exists
        if studyEntry.name in UIDs['StudyInstanceUID'].values:
    #         print(studyEntry.path)

            # create patient index as folder structure
            folderName = 'patient_' + str(counter).zfill(3)
            print(folderName)
            counter+=1
            try:
                os.makedirs(os.path.join(output_PATH, folderName))
            except FileExistsError as e:
    #             print(e)
                pass
            else: 
                pass

            for serieEntry in os.scandir(studyEntry.path):
                if serieEntry.name in UIDs['SeriesInstanceUID'].values:
                    phase_counter = {0:0, 1:0, 2:0, 3:0}
                    for sliceEntry in os.scandir(serieEntry.path):
                        # check if SOPUID exists
                        if os.path.splitext(sliceEntry.name)[0] in UIDs['SOPInstanceUID'].values:
                            # store all slice paths
                            slices.append(sliceEntry.path)  
                            # ds.SOPInstaceUID == os.path.splitext(sliceEntry.name)[0]
                            ds = pydicom.dcmread(sliceEntry.path)

                            # check again if UID matches
                            if ds.SOPInstanceUID == os.path.splitext(sliceEntry.name)[0]:
                                pass
                            else: 
                                print('Error here!! UID does not match!')

                            df2 = df[df['SOPInstanceUID'].str.contains(ds.SOPInstanceUID)]

                            meta = df[df['SOPInstanceUID'].str.contains(ds.SOPInstanceUID)]
                            phase = check_phase(meta.labelId)
                            phase_counter[phase] +=1
                            print('\t\tPhase = ', phase)
                            try:
                                # create folders by phase 
                                os.makedirs(os.path.join(output_PATH, folderName, str(phase)))
                            except FileExistsError as e:
                                # print('folder exists!')
    #                             print('\t', e)
                                pass
                            else: pass
                            slice_number = ds.InstanceNumber
    #                         print('\t', phase, ' ',slice_number)
                            
                            '''here write nrrd'''
                            
                            im = ds.pixel_array
                            im = HU_rescale(im, ds.RescaleSlope, ds.RescaleIntercept)

                            height = df2['data.height'].values
                            width = df2['data.width'].values
                            x = df2['data.x'].values
                            y = df2['data.y'].values
                            im_crop = get_fixed_bb(im, x, y, height, width, size=128)

#                             if slice_counter == 0:
                            accession = df2['unique_acc'].values[0]
                            pathology = df2['pathology'].values[0]
                            grade = int(df2['grade'].values)
                            aggressiveness = 1 if (grade > 2)  else 0
                            data = im_crop

#                             else:
#                                 data = np.dstack([data, im_crop])
                            
#                             slice_counter +=1

                            header = {'accession': accession,
                              'pathology': pathology,
                              'grade': grade,
                              'label': aggressiveness,
                              'SOPUID': ds.SOPInstanceUID,
                              'InstanceNumber': slice_number,
                              'WindowCenter': ds.WindowCenter,
                              'WindowWidth': ds.WindowWidth,
                              'RescaleIntercept': ds.RescaleIntercept,
                              'RescaleSlope': ds.RescaleSlope,
                              'PatientID': ds.PatientID
                              }
                            
                    nrrd.write(os.path.join(output_PATH, folderName, str(phase), folderName + '_' + str(phase) + '.nrrd'), data, header)



                    print('\tPhase ', phase,  ': slice found = ', phase_counter)


    #         break
    #     print(index)
        if index > 11:
            break
    
    return









# def write_nrrd(slices, output_PATH, df):
#     try:
#         os.makedirs(output_PATH)
#     except FileExistsError:
#         # directory already exists
#         #         print('folder exists!')
#         pass
#     labels = []

#     for index, sl in enumerate(slices, 0):
#         folder_name = 'patient_' + str(index).zfill(3)
#         try:
#             os.makedirs(os.path.join(output_PATH, folder_name))
#         except FileExistsError:
#             pass
#             #             print('folder exists!')

#             pass

#         data = np.zeros([128, 128, len(slices[index])])
#         for i in range(len(slices[index])):
#             slice_path = slices[index][i][0]
#             slice_number = slices[index][i][1]
#             ds = pydicom.dcmread(slice_path)

#             im = ds.pixel_array
#             # rescale to HU range
#             im = HU_rescale(im, ds.RescaleSlope, ds.RescaleIntercept)

#             df2 = df[df['SOPInstanceUID'].str.contains(ds.SOPInstanceUID)]

#             vertices = df2['data.vertices'].values[0]
#             coord = closed_polygon(vertices)

#             im_crop = get_fixed_bb(im, coord)

#             data = im_crop

#             if i == 0:
#                 accession = df2['accession'].values[0]
#                 pathology = df2['pathology'].values[0]
#                 grade = int(df2['grade'].values)
#                 aggressiveness = 1 if (grade > 2)  else 0
#                 labels.append((folder_name, aggressiveness))

#             header = {'accession': accession,
#                       'pathology': pathology,
#                       'grade': grade,
#                       'label': aggressiveness,
#                       'SOPUID': ds.SOPInstanceUID,
#                       'InstanceNumber': slice_number,
#                       'WindowCenter': ds.WindowCenter,
#                       'WindowWidth': ds.WindowWidth,
#                       'RescaleIntercept': ds.RescaleIntercept,
#                       'RescaleSlope': ds.RescaleSlope,
#                       'PatientID': ds.PatientID
#                       }

#             nrrd.write(os.path.join(output_PATH, folder_name, folder_name + '_' + str(i).zfill(2) + '.nrrd'), data,
#                        header)

#         print('write ', folder_name)

#     # print('Center = {}, Width = {}'.format(ds.WindowCenter, ds.WindowWidth))
#     #         print('Intercept = {}, Slope = {}'.format(ds.RescaleIntercept, ds.RescaleSlope))
#     #         print('Min = {}, Max = {}'.format(np.min(data), np.max(data)))
#     #         print()
#     return labels


def organize_by_class(src, des, labelNames):
    """
    :param src: source directory
    :param des: destination directory
    :param labelNames: a list of class labels
    :return:

    Note: will throw exception if target path already exist
    dirs_exist_ok ignores the exception, but only in python 3.8+

    """
    for i, exam in enumerate(os.scandir(src)):
        _, header = nrrd.read(os.path.join(exam.path,
                                            os.listdir(exam.path)[0]))
        source_path = exam.path

        if header['label'] in labelNames:
            target_path = os.path.join(des, header['label'], exam.name)
            shutil.copytree(source_path, target_path)
        else:
            raise Exception('Unknown class name')


"""
functions for visualization
"""


def hist_HU(im, HU_range=(-1024, 3071)):
    """
    Plot a histogram with HU range [-1024, 3071]
    Water = 0 HU
    Air = -1000 HU
    Bone = +400 HU
    Soft Tissue = 40+/-400 HU

    See HU_rescale
    """

    vals = im.flatten()
    # plot histogram with 256 bins
    #     HU_range = (360, 440)
    b, bins, patches = plt.hist(vals, 256, HU_range)
    plt.xlim(HU_range)
    plt.show()




# def get_bounding_box(coord, margin=0):
#     coord = pd.DataFrame(coord)
#     xmin = coord[0].min() - margin
#     xmax = coord[0].max() + margin
#     ymin = coord[1].min() - margin
#     ymax = coord[1].max() + margin
#     height = ymax - ymin
#     width = xmax - xmin
#     rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2,
#                              edgecolor='r', facecolor='none')
#     return rect




# def get_mask(coord, margin=0):
#     coord = pd.DataFrame(coord)
#     xmin = np.rint(coord[0].min()).astype(int) - margin
#     ymin = np.rint(coord[1].min()).astype(int) - margin
#     xmax = np.rint(coord[0].max()).astype(int) + margin
#     ymax = np.rint(coord[1].max()).astype(int) + margin
#     height = ymax - ymin
#     width = xmax - xmin
#     rect = patches.Rectangle((xmin, ymin), width, height,
#                                         linewidth=2, edgecolor='r', facecolor='none')
#     mask = np.zeros([512, 512])
#     mask[ymin:ymax, xmin:xmax] = 1

#     return mask


# def show_mask(im, coord, margin):
#     mask = get_mask(coord, margin)

#     figure, ax = plt.subplots(figsize=(5, 5))
#     plt.imshow(mask)
#     #     ax.imshow(im, cmap=plt.cm.bone, clim=(0, 2000))
#     plt.plot(coord[0], coord[1], 'b')
#     plt.xlim(0, 512)
#     plt.ylim(0, 512)




def get_tight_bb(im, x, y, height, width):
    coord = []
    for i in range(int(height)):
        coord.append((x, y+i))
        coord.append((x+int(width-1), y+i))
    for j in range(1,int(width)-1):
        coord.append((x+j, y))
        coord.append((x+j, y+int(height-1)))
    return coord
        
def get_fixed_bb(im, x, y, height, width, size=128):
    
    xc = x + width/2
    xc = xc[0]
    yc = y + height/2
    yc = yc[0]
    
    x0 = np.rint(xc-size/2).astype(int)
    x1 = np.rint(xc+size/2).astype(int)
    y0 = np.rint(yc-size/2).astype(int)
    y1 = np.rint(yc+size/2).astype(int)
    
    im2 = im[y0:y1, x0:x1]
    return im2




def get_cropped_bb(im, coord, margin=15):
    mask = get_mask(coord, margin)
    coord = np.argwhere(mask)
    x0, y0 = coord.min(axis=0)
    x1, y1 = coord.max(axis=0) + 1
    im2 = im[x0:x1, y0:y1]
    return im2


# def show_cropped_bb(im, coord, margin):
#     mask = get_mask(coord, margin)
#     im_crop = get_cropped_bb(im, coord, margin)
#     x, y = coord.min(axis=0)
#     figure, ax = plt.subplots(figsize=(5, 5))
#     ax.imshow(im_crop, cmap=plt.cm.bone)
#     plt.plot(coord[0] - x + margin, coord[1] - y + margin, 'b')
#     plt.xlim(0, im_crop.shape[1] - 1)
#     plt.ylim(0, im_crop.shape[0] - 1)
#     plt.show()

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
    cv = 2

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
