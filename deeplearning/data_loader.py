# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 17:14:38 2024

@author: pky0507
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def get_data_list(root='/dataset/IPMN_Classification/', t = 1, center = None):
    image_list = []
    label_list = []
    center_names = [['nyu'], ['CAD', 'MCF'], ['northwestern', 'NU'], ['AHN', 'ahn'], ['mca'], ['IU'], ['EMC']]
    
    df = pd.read_excel(os.path.join(root, 'IPMN_labels_t'+str(t)+'.xlsx'), usecols=[0, 3])
    df_cleaned = df.dropna(subset=[df.columns[1]]) # remove NaN
    names = df_cleaned.iloc[:, 0].values
    labels = df_cleaned.iloc[:, 1].to_numpy(dtype=np.int64)//2 # we treat no/low-risk as 0 and high-risk as 1
    if center == None:
        center = np.arange(len(center_names))
    elif isinstance(center, int):
        center = [center]
    center_name = []
    for i in center:
        center_name += center_names[i]
    for i in range(len(names)):
        name = names[i].replace('.nii.gz', '')
        for c in center_name:
            if c in name:
                image_list.append(os.path.join(root, 't'+str(t)+'_clean_ROI', name+'.nii.gz'))
                label_list.append(labels[i])
                break
    return image_list, label_list

def get_fold(image:list, label:list, n_splits = 4, fold = 0):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
    skf.get_n_splits(image, label)
    for i, (train_index, test_index) in enumerate(skf.split(image, label)):
        if i == fold:
            train_image = [image[j] for j in train_index]
            train_label = [label[j] for j in train_index]
            test_image = [image[j] for j in test_index]
            test_label = [label[j] for j in test_index]
            return train_image, train_label, test_image, test_label