from __future__ import print_function, division

import os
import numpy as np
import pandas as pd
import json
import sys
from pandas.io.json import json_normalize


import pydicom
import nrrd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import shutil
# from matplotlib import cm

import src.preprocess


def find_CECT(inputPath, df, outputPath):
    '''
    find series that are being annotated.
    all images from this serie are copied.
    
    @inputPath: directory to all retrived exams
    @df: a pandas dataframe contains all metadata (from redcap and md.ai)
    @outputPath: write path
    '''
    
    studyCounter = 0
    for index, studyEntry in enumerate(os.scandir(inputPath)):
        if studyEntry.name in df['StudyInstanceUID'].values:
            # create folder patient000, patinet001, patient002, etc.
            folderName = 'study_' + str(studyCounter).zfill(3)
            df2 = df[df['StudyInstanceUID'].str.contains(studyEntry.name)]
            print('Study: ', folderName)
            studyCounter +=1 
            
            seriesCounter = 0 
            for seriesEntry in os.scandir(studyEntry.path):
                if seriesEntry.name in df['SeriesInstanceUID'].values:
                    # stores series as patient000/0, patinet000/1
                    # need to check again, incase two phases combined in one serie.
                    seriesName = str(seriesCounter)
                    
#                     df3 = df[df['StudyInstanceUID'].str.contains(seriesEntry.name)]
                    print('\t' + seriesName + '  '+ seriesEntry.name)
                    writePath = os.path.join(outputPath, folderName, seriesName) 
                    try:
                        os.makedirs(writePath)
                    except FileExistsError:
                        pass
                    
                    for sopEntry in os.scandir(seriesEntry.path):
                        shutil.copy(sopEntry.path, os.path.join(writePath, sopEntry.name))
                        
                    print('\t Completed!') 
                    seriesCounter += 1
                    
        if index > 20:
            break
            
    return
    

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
    

def find_all(name, path):
    result = []
    for root, dirs, files in os.walk(path):
        if name in files:
            result.append(os.path.join(root, name))
    return result
    
def seperate_phase(inputPath, outputPath, df):
    # split CECT by phase
    # non-contrast: /0
    # arterial: /1
    # pv: /2
    # delayed: /3
    
    total = len(os.listdir(inputPath))
    
    # find total number of patients
#     df_pt = df[['Anon Patient Name','StudyInstanceUID','SeriesInstanceUID', \
#                         'SOPInstanceUID' ]]
#     df_pt = df_pt.drop_duplicates(subset=['StudyInstanceUID'])
#     df_unique = df_pt.drop_duplicates(subset='Anon Patient Name', keep=False)
#     df_duplicated = pd.concat([df_pt,df_unique]).drop_duplicates(keep=False)
#     df_multiple
    
#     print('total %d patients:' % (len(df_pt)))
#     print('unique %d patients:' % (len(df_unique)))
#     print('duplicated %d patients:' % (len(df_duplicated)))
#     print()
    
    
    

    df_pt = df[['Anon Patient Name','StudyInstanceUID','SeriesInstanceUID', \
                        'SOPInstanceUID', 'size']]

    df_pt = df_pt.drop_duplicates(subset=['StudyInstanceUID'])
    def df_difference(df1, df2):
        df3 = pd.concat([df1,df2]).drop_duplicates(keep=False)
        return df3
    df_multi = df_pt[df_pt['size'].str.contains(',')]
    df_temp = df_difference(df_pt, df_multi)
    df_unique = df_temp.drop_duplicates(subset='Anon Patient Name', keep=False)
    df_duplicated = df_difference(df_temp, df_unique)


    print('total patients: {}'.format(len(df_pt)))
    print('multiple tumor patients: {}'.format(len(df_multi)))
    print('duplicated patients: {}'.format(len(df_duplicated)))
    print('value patients: {}'.format(len(df_unique)))

    print()

    
    
    
    
    patient_names = df_unique['Anon Patient Name'].values # array of patient name
    # create a dictionary {patient_name: 0 or 1 (appeared or not)}
    ptdic = {patient_names[i]: 0 for i in range(len(patient_names))}
                
    
    total = len(ptdic)
    ptCounter = 0
    for key, value in ptdic.items():
        if value != 0:
            print('DUPLICATED pt!!!')
        
        # make directory: pt000/
        folderName = 'pt' + str(ptCounter).zfill(3)
        try:
            os.makedirs(os.path.join(outputPath, folderName))
#             print("make " + str(ptCounter))
        except FileExistsError as e:
#             print(e)
            pass
        

        # look up by patient name, return all annotated slices of this patient
        df2 = df[df['Anon Patient Name'].str.contains(key.split('^')[1])]

        
        for i in range(len(df2)):
            sliceLoc = os.path.join(df2.iloc[i]['StudyInstanceUID'], df2.iloc[i]['SeriesInstanceUID'], df2.iloc[i]['SOPInstanceUID'])
            srcPath = os.path.join(inputPath, sliceLoc+'.dcm')
            if os.path.exists(srcPath) == False:
                raise FileExistsError
            else:
#                 print('True')
                pass
            
            phase = check_phase(df2.iloc[i]['labelId'])
            writePath = os.path.join(outputPath, folderName, str(phase))
            
            try:
                os.makedirs(writePath)

            except FileExistsError: 
#                 print('Directory exists.')
                pass
            shutil.copy(srcPath, writePath)

            
#             print('phase = ', phase)
#             print('seris = ', df2.iloc[i]['SeriesInstanceUID'])
#             print('write path = ', writePath)
#             print()
        
  
        
        ptCounter+=1
        
        j = ptCounter/ total
#         sys.stdout.write("\r[%-20s] %d%%" % ('='*int(20*j), 100*j))
        sys.stdout.write("\r[%-20s] %d/%d" % ('='*int(20*j), ptCounter, total))

        sys.stdout.flush()
        
        
#         if ptCounter > 100:
#             return   
        
        
    if ptCounter == total:  
        print()
        print('processed', ptCounter, 'patients, ')
    else:
        print('ERROR! Number does not match!!')
        
        
        
        
        
    
#     # use RedCap data to see if a patient has two accession number
#     patient_list = ['unique_pt_names', 'appeared']  
#     for i in directory: 
#         ds = i.path
#         if ds.patient_name in patient_list and 'appeard' == 0:
#             appeared == 1
#             mkdir('patient000')
            
#         elif ds.patient_name in patient_list and 'appeared' == 1:
#             print('multiple studies for thsis patient')
    

    
def select_phase(inputPath, outputPath):
    # select one-phase, using the one has the largest annotations.
    
    total = len(os.listdir(inputPath))
    
    for index, ptEntry in enumerate(os.scandir(inputPath)):
        folderName = 'pt' + str(index).zfill(3)
        writePath = os.path.join(outputPath, folderName)

        try:
            os.makedirs(writePath)
        except FileExistsError: 
#                 print('Directory exists.') 
                pass
                
        maxFiles = 0
        for phaseEntry in os.scandir(ptEntry.path):
            cpt = sum([len(files) for r, d, files in os.walk(phaseEntry.path)])
            if cpt >= maxFiles:
                maxFiles = cpt
                copyPath = phaseEntry.path
                phase = phaseEntry.name
#             print('\tphase ', phaseEntry.name, ' = ', cpt)
#         print('\tmax files = ', maxFiles, '@ ', copyPath)


        for index2, sliceEntry in enumerate(os.scandir(copyPath)):
            filename = str(index2)
            shutil.copy(sliceEntry.path, writePath)
        if maxFiles == (index2 +1):
#             print('\t', maxFiles, ' files, phase', phase, 'are copied!')
            pass
        else:
            print('NUMBERS NOT EQUAL!!!')

            print('maxFiles = ', maxFiles)
            print('index2 = ', index2)
        
        if phase == 0 or  phase == 3:
            print('Error! Phase', phase, 'is selected!')
            break
        
#         if index > 20:
#             break    

        
        j = (index+1) / total
#         sys.stdout.write("\r[%-20s] %d%%" % ('='*int(20*j), 100*j))
        sys.stdout.write("\r[%-20s] %d/%d" % ('='*int(20*j), (index+1), total))


        sys.stdout.flush()
    
    if (index+1) == total:  
        print()
        print('processed', index+1, 'patients, ')
    else:
        print('ERROR! Number does not match!!')




    
# def qc_check()    
    
    
    
def print_info(df):
    for index, row in df.iterrows():
        print('\t\tindex: {}, '.format(index))
        print('\t\tSOPID: {}, tumor size: {}, '.format(row['SOPInstanceUID'], row['size']))
        print('\t\tpatient name: {}, acc #: {}'.format(row['Anon Patient Name'], row['unique_acc']))    
        print('\t\trecord id: {}'.format(row['record_id']))    


    return 
    
    
def write_nrrd(inputPath, outputPath, df):
    count = 0
    
    # calculate total number of slices need to be processed 
    total = sum([len(files) for root, dir, files in os.walk(inputPath)])


    for index, ptEntry in enumerate(os.scandir(inputPath)):


        for index2, sliceEntry in enumerate(os.scandir(ptEntry.path)):

            ds = pydicom.dcmread(sliceEntry.path)

            im = ds.pixel_array
            im = src.preprocess.HU_rescale(im, ds.RescaleSlope, ds.RescaleIntercept)



            sopUID = sliceEntry.name[:-4] # exclude '.dcm'
            metadata =  df[df['SOPInstanceUID'].str.contains(sopUID)]
            
            # skip in two annotations found in one slice
            if len(metadata) > 1:
                print()
                print('\tDuplicated SOPUID: ')
                print('\t{}'.format(ptEntry.name))
                print_info(metadata)
                print()
                count += 1
                break 


            try:
                height = metadata['data.height'].values
                width = metadata['data.width'].values
                x = metadata['data.x'].values
                y = metadata['data.y'].values
                im_crop = src.preprocess.get_fixed_bb(im, x, y, height, width, size=128)


                accession = metadata['unique_acc'].values[0]
                pathology = metadata['pathology'].values[0]
                grade = int(metadata['grade'].values[0])
                aggressiveness = 1 if (grade > 2)  else 0   
                phase = check_phase(metadata['labelId'].values[0])
            
                header = {'accession': accession,
                          'pathology': pathology,
                          'grade': grade,
                          'label': aggressiveness,
                          'StudyUID': ds.StudyInstanceUID,
                          'SeriesUID': ds.SeriesInstanceUID,
                          'SOPUID': ds.SOPInstanceUID,
                          'InstanceNumber': ds.InstanceNumber,
                          'WindowCenter': ds.WindowCenter,
                          'WindowWidth': ds.WindowWidth,
                          'RescaleIntercept': ds.RescaleIntercept,
                          'RescaleSlope': ds.RescaleSlope,
                          'PatientID': ds.PatientID,
                          'PatientName':ds.PatientName,
                          'Phase': phase,
                  }
                
                try:
                    os.makedirs(os.path.join(outputPath, ptEntry.name))
                except FileExistsError:
                    pass
                # only write nrrd here
                nrrd.write(os.path.join(outputPath, ptEntry.name, str(index2)+'.nrrd'), im_crop, header)
            
                
                
            except AttributeError as err:
                print()
                print('\tpatient name = {}'.format(ptEntry.name))

                print('\t', err) 
                print_info(metadata)
                print()
                count +=1 
                break
                

            count += 1
        
        j = count / total
#         sys.stdout.write("\r[%-20s] %d%%" % ('='*int(20*j), 100*j))
        sys.stdout.write("\r[%-20s] %d/%d" % ('='*int(20*j), count, total))


        sys.stdout.flush()
    
    if count == total:  
        print()
        print('Converted ', index, 'patients, ', count, 'slices.')
    else:
        print('ERROR! Number does not match!!')
    








