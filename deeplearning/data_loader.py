import os
import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import monai
import pandas as pd
import json
from monai.data import DataLoader, ImageDataset,decollate_batch
def get_tr_vl_ts_list(dataset_dtl,fold=0):

    with open(dataset_dtl, 'r') as f:
        fold_data = json.load(f)
    test_list = []
    for name in fold_data['test_files']:
        test_list.append(name.split('.nii.gz')[0])
    # test_list = [n.lower() for n in test_list]

    train_list =[]
    for name in fold_data['cross_validation'][fold]['train_files']:
        train_list.append(name.split('.nii.gz')[0])
    # train_list = [n.lower() for n in train_list]


    val_list=[]
    for name in fold_data['cross_validation'][fold]['validation_files']:
        val_list.append(name.split('.nii.gz')[0])
    # val_list = [n.lower() for n in val_list]
    
    return train_list,val_list,test_list


def get_data_list(split, images_path,labels_path,fold):
    image_list = []
    label_list = []
    try:
        df = pd.read_csv(labels_path)
    except Exception as e1:
        try:
            df = pd.read_excel(labels_path)
        except Exception as e2:
            print(f"An error occurred while reading label {labels_path}: {e2}")
    df_cleaned = df.dropna(subset=[df.columns[1]]) # remove NaN

    train_list,val_list, test_list = get_tr_vl_ts_list(split,int(fold))
    df_train = df_cleaned[df_cleaned['name'].isin(train_list)]
    df_val = df_cleaned[df_cleaned['name'].isin(val_list)]
    df_test = df_cleaned[df_cleaned['name'].isin(test_list)]

    df_train['path'] = df_train['name'].apply(lambda x: os.path.join(images_path, x+'.nii.gz'))
    df_test['path'] = df_test['name'].apply(lambda x: os.path.join(images_path, x+'.nii.gz'))
    df_val['path'] = df_val['name'].apply(lambda x: os.path.join(images_path, x+'.nii.gz'))

    return df_train,df_val,df_test

# image_list, label_list = get_data_list()

# train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96)), RandRotate90()])
# test_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96))])
# train_ds = ImageDataset(image_files=image_list, labels=label_list, transform=train_transforms)