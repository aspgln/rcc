from __future__ import print_function, division

#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from pandas.io.json import json_normalize




def show_batch(sample_batched):
    """show image with label for a batch of samples"""
    image_batch, lable_batch = sample_batched[0], sample_batched[1]

    batch_size = len(image_batch)
    im_size = image_batch.size(2)

    col = 4
    row = int((batch_size) / col)

    fig, ax = plt.subplots(nrows=row, ncols=col)
    for i in range(row):
        for j in range(col):
            print('batch no ', i*4+j)
            #             im = ax[i,j].imshow(image_batch[i*4+j, 1, :, :], cmap=plt.cm.bone, vmin = -360, vmax = 440)

            if row > 1:
                # show RGB
                im = ax[i, j].imshow((np.transpose(image_batch[i * 4 + j, :, :, :])+2)/4)


                ax[i, j].axis('off')
                ax[i, j].set_title(lable_batch[i * 4 + j].numpy())
            if row == 1:
                im = ax[j].imshow((np.transpose(image_batch[i * 4 + j, :, :, :])+2)/4)
                ax[j].axis('off')
                ax[j].set_title(lable_batch[i * 4 + j].numpy())
    cb_ax = fig.add_axes([0.96, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(im, cax=cb_ax)
    # cbar = fig.colorbar(im)

    plt.show()
    return


def show_batch2(sample_batched, vmin=-360, vmax=440):
    """show image with label for a batch of samples"""
    image_batch, lable_batch = sample_batched[0], sample_batched[1]
    batch_size = len(image_batch)
    im_size = image_batch.size(2)

    col = 4
    row = int((batch_size) / col)
    # fig,ax = plt.subplots(nrows=row)
    for i in range (row):
        fig, ax = plt.subplots()

        im1 = image_batch[i*4+0, 0, :, :]
        im2 = image_batch[i*4+1, 0, :, :]
        im3 = image_batch[i*4+2, 0, :, :]
        im4 = image_batch[i*4+3, 0, :, :]

        im_batch = np.hstack((im1,im2,im3,im4))
        im = ax.imshow(im_batch, cmap=plt.cm.bone, vmin=vmin, vmax=vmax)

        cb_ax = fig.add_axes([0.96, 0.1, 0.02, 0.5])
        cbar = fig.colorbar(im, cax=cb_ax)
        # cbar = fig.colorbar(im)
        plt.show()
    # plt.show()
    return

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    (tn, fp, fn, tp) = cm.ravel()
    print('p/t     1     0')
    print('  1    {}    {}'.format(tp, fp))
    print('  0     {}    {}'.format(fn, tn))


def evaluate(pred_list, label_list):
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


'''
START FROM HERE: 
'''

def load_log(PATH1, PATH2):
    '''
    load and combine AIR log from Sage (for MD.ai) and Kirti (AIR Retrieval)
    '''
    with open(PATH1) as f:
        #sage
        data1 = pd.read_csv(PATH1) 
#         data1 = data1[['Orig Acc #','Orig Study UID',
#                        'Anon Study UID',
#                        'Image Req. Status',
#                       'Anon Patient Name']]
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
    json_data = json_data[['labelId','StudyInstanceUID','SeriesInstanceUID', 'SOPInstanceUID', 'data.vertices','data.height','data.width', 'data.x', 'data.y', 'height', 'width']]
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
        temp = json_data[json_data['labelId'].str.contains(BBId[i], na=False)]
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


def show_grade_distribution(df, pathology_type):
    
    df = df[df['pathology']== pathology_type]
    print(pathology_type)

    patientStats = df[['Anon Patient Name', 'grade', 'pathology', 'size']].drop_duplicates()
    
    patient_grade_is_null = patientStats[patientStats['grade'].isnull()]
    patientStats = patientStats[patientStats['grade'].notnull()]
    print('Number of patients with grades: ', patientStats.shape[0])
    
    patientStats = patientStats['grade'].value_counts().sort_index()
    print()
    print('By patients:')
    for grade in range(1,7):
        if grade in patientStats:
            if grade == 5:
                print('\tUnspecified high grade: {:.0f}'.format(patientStats[grade]))
            elif grade == 6:
                print('\tUnspecified low grade: {:.0f}'.format(patientStats[grade]))

            else:
                print('\tGrade', int(grade), ':', patientStats[grade]) 
        else: 
            patientStats = patientStats.add(pd.Series([0], index=[grade]), fill_value=0)
    print()     
    low = patientStats[1.0]+patientStats[2.0]+patientStats[6.0]
    high = patientStats[3.0]+patientStats[4.0]+patientStats[5.0]
    print('\tLow grade: {:.0f}/{:.0f}, {:.1%}'.format(low, low+high, low/(low+high)))    
    print('\tHigh grade: {:.0f}/{:.0f}, {:.1%}'.format(high, low+high, high/(low+high))) 

    
    print()
    sliceStats = df['grade'].value_counts().sort_index()
    print('By slices:')
    for grade in range(1,7):
        if grade in sliceStats:
            if grade == 5:
                print('\tUnspecified high grade: {:.0f}'.format(sliceStats[grade]))
            elif grade == 6:
                print('\tUnspecified low grade: {:.0f}'.format(sliceStats[grade]))
            else:
                print('\tGrade', int(grade), ':', sliceStats[grade]) 
        else: 
            sliceStats = sliceStats.add(pd.Series([0], index=[grade]), fill_value=0)
    print()     
    low = sliceStats[1.0]+sliceStats[2.0]+sliceStats[6.0]
    high = sliceStats[3.0]+sliceStats[4.0]+sliceStats[5.0]
    print('\tLow grade: {:.0f}/{:.0f}, {:.1%}'.format(low, low+high, low/(low+high)))    
    print('\tHigh grade: {:.0f}/{:.0f}, {:.1%}'.format(high, low+high, high/(low+high))) 
    
    
    return 




def show_pathology_stats(dataframe, pathology_type):
    df_pathology  = dataframe[dataframe['pathology']== pathology_type]

    print(pathology_type)
    print('Number of patients annoated on MD.ai: ', np.unique(df_pathology['Anon Patient Name']).shape)
    print('Number of studies annoated on MD.ai: ', np.unique(df_pathology['StudyInstanceUID']).shape)
    print('Number of series annoated on MD.ai: ', np.unique(df_pathology['SeriesInstanceUID']).shape)
    print('Number of slices annoated on MD.ai: ', np.unique(df_pathology['SOPInstanceUID']).shape)


    df = df_pathology[['Anon Patient Name','StudyInstanceUID','SeriesInstanceUID', \
                        'SOPInstanceUID', 'size']]

    df = df.drop_duplicates(subset=['StudyInstanceUID'])
    print()

    def df_difference(df1, df2):
        df3 = pd.concat([df1,df2]).drop_duplicates(keep=False)
        return df3

    # df_multi: studies where multiple tumors on records (filtered by size)
    # df_duplicated: studies where 2 StudyInsatnceUID correponeds to 1 Patient Name
    # df_unique: neither of the above two conditions
    df_multi = df[df['size'].str.contains(',')]
    df_temp = df_difference(df, df_multi)
    df_unique = df_temp.drop_duplicates(subset='Anon Patient Name', keep=False)
    df_duplicated = df_difference(df_temp, df_unique)

    df_pathology_full= pd.merge(df_pathology, df_unique, how='inner', on=['Anon Patient Name','StudyInstanceUID','size'])
    df_pathology_full = df_pathology_full.drop(['SeriesInstanceUID_y', 'SOPInstanceUID_y'], axis=1)
    df_pathology_full = df_pathology_full.rename(columns={'SeriesInstanceUID_x': 'SeriesInstanceUID', 'SOPInstanceUID_x': 'SOPInstanceUID'})  # new method


    print('total studies: {}'.format(len(df)))
    print('multiple tumor stueis: {}'.format(len(df_multi)))
    print('duplicated studies: {}'.format(len(df_duplicated)))
    print()
    print('valid studies: {}'.format(len(df_unique)))
    print('number of slices: {}'.format(len(df_pathology_full)))

    return


def show_number_of_lesions(dataframe, pathology_type):
    
    df = dataframe[dataframe['pathology']== pathology_type]
    print(pathology_type)
    print('Number of patients : ', np.unique(df['Anon Patient Name']).shape)
    print('Number of studies : ', np.unique(df['unique_acc']).shape)

    patientStats = df[['unique_acc', 'grade', 'pathology', 'size']].drop_duplicates()


    df_lesions = patientStats['size']
    l_list = []
    for x in df_lesions:
        temp = x.split(',')
        for i in range(len(temp)):
            l_list.append(temp[i])
    print('Number of lesions: {}'.format(len(l_list)))




'''
select image and hdf5 processing 
'''

def check_phase(labelId):
    if labelId == 'L_e7D5Yl' or labelId == 'L_PJR807':
        phase = 0 # non-contract phase
    elif labelId == 'L_y7KqAJ' or labelId == 'L_e7kVd1':
        phase = 1 # arterial phase
    elif labelId ==  'L_q1Mrp7' or labelId == 'L_4lgMO1':
        phase =2 # portal venous phase
    elif labelId == 'L_gljGOJ' or labelId =='L_47EP91':
        phase = 3 # delayed phase
    else:
        print('UNKNOWN PHASE!!!')
    return phase
    

    
# def create_data_sstruct()





















