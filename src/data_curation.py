from __future__ import print_function, division

#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import h5py
import os
import pydicom
from pandas.io.json import json_normalize
import shutil
import sys

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
    
    data_log = data_log[['Orig Patient Name', 'Anon Patient Name', 'Orig MRN', 'Anon MRN', 'Orig Acc #', 'Anon Acc #', 'Orig Study UID', 'Anon Study UID_x']]
    data_log = data_log.rename(columns={'Orig MRN':'MRN_log','Anon MRN':'Anon MRN_log', 'Orig Acc #':'Acc_log', 'Anon Acc #':'Anon Acc_log', 
                                       'Orig Study UID':'Orig StudyUID_log',  'Anon Study UID_x':'Anon StudyUID_log'})
    return data_log


def load_json(PATH):
    '''load json from md.ai (annotations)'''
    
    with open(PATH) as f:
        js = json.load(f)
    json_data = json_normalize(js['datasets'][0]['annotations'])
    json_data = json_data[['labelId','StudyInstanceUID','SeriesInstanceUID', 'SOPInstanceUID','data.height','data.width', 'data.x', 'data.y', 'height', 'width', 'annotationNumber']]
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
    data_clinical = data_clinical.dropna(subset=['mrn'])
    data_clinical = data_clinical[['record_id', 'mrn', 'accession', 'size', 'pathology', 'grade','laterality', 'stacked']]

    
#     # create new df to seperate fields with 2 acc #
#     new_df = data_clinical[['record_id', 'accession']]
#     new_df = pd.DataFrame(new_df.accession.str.split(',').tolist(), index=new_df.record_id).stack()
#     new_df = new_df.reset_index([0,'record_id'])
#     new_df.columns = ['record_id', 'unique_acc']
#     # drop acc# 11568327 for duplicate
#     new_df = new_df.drop_duplicates(subset=['unique_acc'])

#     data_clinical = pd.merge(data_clinical, new_df, left_on=['record_id'], 
#                           right_on=['record_id'])
#     data_clinical = data_clinical.drop(['accession'], axis=1)
#     data_clinical['accession'] = pd.to_numeric(data_clinical['accession'])


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


def df_difference(df1, df2):
    df3 = pd.concat([df1,df2]).drop_duplicates(keep=False)
    return df3

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
    print('Number of studies : ', np.unique(df['accession']).shape)

    patientStats = df[['accession', 'grade', 'pathology', 'size']].drop_duplicates()


    df_lesions = patientStats['size']
    l_list = []
    for x in df_lesions:
        temp = x.split(',')
        for i in range(len(temp)):
            l_list.append(temp[i])
    print('Number of lesions: {}'.format(len(l_list)))


    
def read_metadata():
    path1 =  './metadata/sage_all.csv'
    path2 = './metadata/kirti_all.csv'
    data_log = load_log(path1, path2)

    data_clinical = load_redcap('./metadata/RedCap.csv')

    data_annotation = load_json('./metadata/mdai.json')


    merge_1 = pd.merge(data_log, data_clinical, left_on='MRN_log', 
                       right_on='mrn')
    # merge between StudyInstanceUID(md.ai) and AIR(from sage)
    merge_2 = pd.merge(data_annotation, merge_1, left_on='StudyInstanceUID', 
                       right_on='Anon StudyUID_log')
    merge_2 = merge_2[['mrn','MRN_log','Anon MRN_log','record_id', 'accession', 'Acc_log', 'Anon Acc_log','Orig Patient Name', 'Anon Patient Name', 
                       'pathology', 'grade','size',
                       'Orig StudyUID_log','Anon StudyUID_log','StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID',
                       'height', 'width', 'labelId',  'data.height', 'data.width', 'data.x', 'data.y', 'laterality', 'stacked']]

    # merge with data excluded cysts
    data_exclude_cysts = pd.read_csv('./metadata/all_patients/excluded_cyctic.csv')
    # data_exclude_cysts
    data_exclude_cysts.index.name='pt_index'
    data_exclude_cysts = data_exclude_cysts.reset_index()


    data_merge = pd.merge( data_exclude_cysts,merge_2, how='inner', left_on='MRN', right_on='mrn')

    print('Number of studies downloaed from AIR: ', data_log.shape)
    print('Number of studies annoated on MD.ai: ', np.unique(data_annotation['SeriesInstanceUID']).shape)
    print('Number of studies on Redcap : ', data_clinical.shape)
    print('Number of studies from merge AIR and Redcap : ', merge_1.shape) 
    print('Number of studies from merge AIR, Redcap and MD.ai : ', np.unique(merge_2['accession']).shape)
    print('Number of studies from exclude_cysts : ', np.unique(data_merge['accession']).shape)
    print('Number of patients : ', np.unique(data_merge['Anon Patient Name']).shape)


    return data_merge

    


'''
select image and hdf5 processing 
'''

def check_phase(labelId):
    if labelId == 'L_e7D5Yl' or labelId == 'L_PJR807':
        phase = 'noncon' # non-contract phase
    elif labelId == 'L_y7KqAJ' or labelId == 'L_e7kVd1':
        phase = 'arterial' # arterial phase
    elif labelId ==  'L_q1Mrp7' or labelId == 'L_4lgMO1':
        phase = 'portal' # portal venous phase
    elif labelId == 'L_gljGOJ' or labelId =='L_47EP91':
        phase = 'delayed' # delayed phase
    else:
        print('UNKNOWN PHASE!!!')
    return phase
    
    
    

        
def HU_rescale(im, slope, intercept):
    """
    Convert store value to HU value
    Linear correlation:
        Output Value = SV * RescaleSlope + RescaleIntercept
    """
    return im * slope + intercept


def get_fixed_bb(im, x, y, height, width, size=128):
    xc = x + width/2
    yc = y + height/2
    
    x0 = np.rint(xc-size/2).astype(int)
    x1 = np.rint(xc+size/2).astype(int)
    y0 = np.rint(yc-size/2).astype(int)
    y1 = np.rint(yc+size/2).astype(int)
    
    im2 = im[y0:y1, x0:x1]
    return im2        
        


        
def crop_image(im, row):
    '''
    get the cropped bounding box
    @im: original image
    @row: one row of df in an iteration
    '''

    height = row['data.height']
    width = row['data.width']
    x = row['data.x']
    y = row['data.y']
    im_crop = get_fixed_bb(im, x, y, height, width, size=128)
    return im_crop

def find_max_phase(df):
    this_series = df['labelId'].value_counts()
    max_label = this_series.keys()[0]
    n_max = this_series.values[0]
    phase = check_phase(max_label)
#     print(this_series)
    try: 
        pass
    except this_series.values[1] > 1:
        print('Second common case is greater than 1!')
        return
    return (max_label, phase, n_max)


def add_attrs(h5_object, df):

    attrs_list = ['MRN', 'Pathology', 'grade', 'Tumor Type', 'pt_index', 'record_id']
    for attr in attrs_list:
        h5_object.attrs[attr] =  df[attr].drop_duplicates().squeeze()
    
#     (max_label, phase, n_max) = find_max_phase(df)
#     h5_object.attrs['annotation_phase'] = phase
#     h5_object.attrs['annotation_label'] = max_label
    
    
#     attrs_list = ['StudyInstanceUID','SeriesInstanceUID',  'SOPInstanceUID','InstanceNumber','WindowCenter', 'WindowWidth','RescaleIntercept', 'RescaleSlope','PatientID','PatientName']
#     for attr in attrs_list:
#         h5_object.attrs[attr] = np.empty([n_max,], dtype=object)
    
        

def create_h5_object(input_path, path, df_this_patient, dim=128):
    # create an hdf5 object for each patient
    # initialization
    with h5py.File(path, mode='w', track_order=True) as f:
        # create single phase image dset
        group_single = f.create_group('single_phase')
        dset = group_single.create_dataset('data', (dim,dim,0), maxshape=(dim,dim,None), dtype='f')
        # add series level attributes
        add_attrs(group_single, df_this_patient)

        # create each dicom header as an individual dset
        (max_label, phase, n_max) = find_max_phase(df_this_patient)
        group_single.attrs['max_label'] = max_label
        group_single.attrs['phase'] = phase
        group_single.attrs['n_max'] = n_max
        
        dicom_names = ['StudyInstanceUID','SeriesInstanceUID',  'SOPInstanceUID','InstanceNumber','WindowCenter', 'WindowWidth','RescaleIntercept', 'RescaleSlope','PatientID','PatientName']
        dt = h5py.string_dtype(encoding='utf-8')        
        for x in dicom_names:
            group_single.create_dataset(x, (n_max,), dtype=dt) 
            
        
#         # create multi-phase dset
#         group_multi = f.create_group('multi_phase')
#         for ph in ['noncon', 'arterial', 'portal', 'delayed']:
# #             add_group_attrs(group_multi, df_this_patient)
#             dset = group_multi.create_dataset(ph, (dim,dim,0), maxshape=(dim,dim,None), dtype='f')
    
    with h5py.File(path, mode='a', track_order=True) as f:
        df_valid = df_this_patient[df_this_patient['labelId'] == max_label]
        for index, (_, row) in enumerate(df_valid.iterrows()):
            input_slice_path = os.path.join(input_path, row['StudyInstanceUID'], 
                                   row['SeriesInstanceUID'], row['SOPInstanceUID']+'.dcm')
            
            if os.path.exists(input_slice_path) == False:
                print('DICOM does not exist, check UIDs.')
                raise FileExistsError
            else:
                pass
            
            # read data pydicom
#             if index == 0:
#                 print(row['pt_index'])
            ds = pydicom.dcmread(input_slice_path)
            im = ds.pixel_array
            im = HU_rescale(im, ds.RescaleSlope, ds.RescaleIntercept)
            im_crop = crop_image(im, row)
            
            data = f['single_phase/data']
            data.resize((dim,dim,data.shape[2]+1))
            data[:,:,-1] = im_crop

#             if index == 10:
#                 print('here')
    
            for name in dicom_names: 
                try: 
                    dicom_name_dset = f['single_phase/{}'.format(name)]
                    if type(ds.data_element(name).value) ==  pydicom.multival.MultiValue:
                        if index == 0:
                            print('\tAt pt_{}, {} is a MultiValue = {}, take {}'.format(row['pt_index'], name, ds.data_element(name).value, ds.data_element(name).value[0]))
                        dicom_name_dset[index] = ds.data_element(name).value[0]
                    else: 
                        dicom_name_dset[index] = ds.data_element(name).value
                except KeyError:
                    print('\tAt pt_{}, {} key ks not found!!'.format(row['pt_index'], name))
                    continue
#         for key, value in f['single_phase/data'].attrs.items():
#             print(key, value)

    return 



def read_h5_object(path):
    with h5py.File(path, mode='r', track_order=True) as f:
        for p in ['noncon', 'arterial', 'portal', 'delayed']:
            dset = f['multi_phase/{}'.format(p)]
            print('\t', p, dset.shape[2])

            
            
def print_hdf_names(path):
    def print_names(name):
        print(name)
    with h5py.File(path, 'r', track_order=True) as f:
        f.visit(print_names)   

def display_hdf5_single_phase(path):
    '''
    show hdf5 file for single phase
    input: path
    '''
    with h5py.File(path, mode='r', track_order=True) as f:
        dset = f['single_phase/data']
        print('shape:', dset.shape)
        phase = f['single_phase'].attrs['phase']
        
    #     im = dset[:,:,0]
        num = dset.shape[2]
        if num == 0:
            pass
        elif num == 1:
            list_to_display = np.linspace(0, num-1, num)
            fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(2,2), sharex=True, sharey=True)
            for index, i in enumerate(list_to_display):
                slice_num = int(i)
        #         print(type(slice_num))
                im = dset[:,:,slice_num]
                axes.imshow(im, vmin=-360, vmax=440, cmap=plt.cm.bone)
                axes.set_title('slice: {}'.format(slice_num))
                fig.suptitle(p, fontsize=16, y=0.95)
                fig.tight_layout()
                fig.subplots_adjust(top = 0.7, bottom=0.1)
        elif num > 1:
            list_to_display = np.linspace(0, num-1, 10)
            fig, axes = plt.subplots(ncols=10, nrows=1, figsize=(20,2), sharex=True, sharey=True)
            for index, i in enumerate(list_to_display):
                slice_num = int(i)
        #         print(type(slice_num))
                im = dset[:,:,slice_num]

                axes[index].imshow(im, vmin=-360, vmax=440, cmap=plt.cm.bone)
                axes[index].set_title('slice: {}'.format(slice_num))
                fig.suptitle(phase, fontsize=16, y=0.90)
                fig.tight_layout()
                fig.subplots_adjust(top = 0.7, bottom=0.1)


                
def create_h5(input_path, output_path, df):
    '''
    A wrapper function for converting DICOM to hdf5
    @input_path: DICOM location
    @output_path: hdf5 location
    @df: data_merge
    '''
    
    # set index to mrn
    mrn_list = df['MRN'].drop_duplicates().tolist()
    total = len(mrn_list)

    dup_list = []
    for i, mrn in enumerate(mrn_list):
        print(i)

        df_this_patient = df[df['MRN']==mrn]

        pt_index = 'pt' + str(df_this_patient['pt_index'].iloc[0]).zfill(3)
        pt_path = os.path.join(output_path, pt_index+'.hdf5')


#         this block is to find duplicate patients
#         if df_this_patient['size'].drop_duplicates().values[0].__len__() > 3 and df_this_patient['Pathology'].drop_duplicates().values[0] == 'clear cell':
#             dup_list.append(df_this_patient['pt_index'].iloc[0])
#     #         break
#     #         print()
#             print(pt_index)
#             return


        this = create_h5_object(input_path, pt_path, df_this_patient )
    #     if i > 19:
    #         break
    #     j = i / total
    #     sys.stdout.write("\r[%-20s] %d/%d" % ('='*int(20*j), i, total))
    #     sys.stdout.flush()
    #     break
        print('-'*30)

    return
    
    

def display_hdf5(path):
    '''
    show hdf5 file for multi-phase
    input: path
    '''
    with h5py.File(path, mode='r', track_order=True) as f:
        for p in ['noncon', 'arterial', 'portal', 'delayed']:
            dset = f['{}/data'.format(p)]
            print(p, dset.shape)

            dset =f['{}/data'.format(p)]
        #     im = dset[:,:,0]
            num = dset.shape[2]
            if num == 0:
                pass
            elif num == 1:
                list_to_display = np.linspace(0, num-1, num)
                fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(2,2), sharex=True, sharey=True)
                for index, i in enumerate(list_to_display):
                    slice_num = int(i)
            #         print(type(slice_num))
                    im = dset[:,:,slice_num]
                    axes.imshow(im, vmin=-360, vmax=440, cmap=plt.cm.bone)
                    axes.set_title('slice: {}'.format(slice_num))
                    fig.suptitle(p, fontsize=16, y=0.95)
                    fig.tight_layout()
                    fig.subplots_adjust(top = 0.7, bottom=0.1)
            elif num > 1:
                list_to_display = np.linspace(0, num-1, 10)
                fig, axes = plt.subplots(ncols=10, nrows=1, figsize=(20,2), sharex=True, sharey=True)
                for index, i in enumerate(list_to_display):
                    slice_num = int(i)
            #         print(type(slice_num))
                    im = dset[:,:,slice_num]

                    axes[index].imshow(im, vmin=-360, vmax=440, cmap=plt.cm.bone)
                    axes[index].set_title('slice: {}'.format(slice_num))
                    fig.suptitle(p, fontsize=16, y=0.90)
                    fig.tight_layout()
                    fig.subplots_adjust(top = 0.7, bottom=0.1)


def drop_invalid_index(pt_index, input_path):
    '''
    Drop invalid cases
    @pt_index: a list-like object contains pt_index. Valid formats include python list [], numpy array, a single or multi-index pandas object 
        example: array([1,2,3,4,5])
        example: df.index
        
    @input_path: directory of hdf files
    
    @ouput: same type of pt_index
    '''
    
    # DROP annotation on noncon or delayed phases
    print('before shape = {}'.format(pt_index.shape))
    
    for index in pt_index: 
        file_path = os.path.join(input_path, 'pt{}.hdf5'.format(str(index).zfill(3)))
        with h5py.File(file_path, mode='r', track_order=True) as f:
            slice_max = 0
            
            phase = f['single_phase'].attrs['phase']
            
#             for multi-phase dset
#             for p in ['noncon', 'arterial', 'portal', 'delayed']:
#                 dset = f['{}/data'.format(p)]

#                 if dset.shape[2]>slice_max:
#                     slice_max = dset.shape[2]
#                     phase_max = p

            if phase != 'arterial' and phase != 'portal':
                print('\tDROPPED: {}, has {}'.format('pt{}.hdf5'.format(str(index).zfill(3)), phase))
                # delete this index from list
                pt_index = pt_index[pt_index!=index]
    print('after shape = {}'.format(pt_index.shape))
    print()
    return pt_index


def create_labels(root_dir):
    '''
    Create (pt_index, label) pairs. This is used for dataloader.
    Train/val/test folder under root_dir
    Will create train/val/test.csv under the root_dir
    '''
    
    for mode in ['train', 'val', 'test']:
        # labels is a list of tuple, storing (pt_index, label)
        labels = []
        for pt in os.scandir(os.path.join(root_dir, mode)):
            with h5py.File(pt.path, mode='r', track_order=True) as f:
                grade = f['single_phase'].attrs['grade']
                if grade == 1 or grade == 2 or grade == 6:
                    label = 0
                elif grade == 3 or grade == 4 or grade ==5:
                    label = 1
#                 print(pt.name.split('.')[0], label, grade)
                labels.append((pt.name.split('.')[0], label))

        idx, values = zip(*labels)
        s = pd.Series(values, idx)
        s.to_csv(os.path.join(root_dir, '{}.csv'.format(mode)), header=False)




# def create_h5_one_phase(index_list, input_path, output_path):
# #    '''
# #    Find the correct phase, either arterial/portal, and save as new hdf5 file
# #    @index_list: a list-list object, such as pd.Series or list
# #    '''
#     for i in index_list:
#         in_ = os.path.join(input_path, 'pt{}.hdf5'.format(str(i).zfill(3)))    
#         shutil.copy(in_, output_path)
#         out_ = os.path.join(output_path, 'pt{}.hdf5'.format(str(i).zfill(3)))    

#         with h5py.File(out_, mode='a', track_order=True) as f:
#             del f['noncon']
#             del f['delayed']

#             dset_arterial = f['arterial/data']
#             dset_portal = f['portal/data']

#             if dset_arterial.shape[2] > dset_portal.shape[2]:
#                 del f['portal']
#             elif dset_arterial.shape[2] < dset_portal.shape[2]:
#                 del f['arterial']
#             elif arterial_len == portal_len:
#                 print('EQUAL with ', arterial_len, ', check index', i)
#     print('finished.')















