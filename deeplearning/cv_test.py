# %%
import pandas as pd
from sklearn.metrics import accuracy_score,roc_auc_score,classification_report,recall_score,precision_score

import numpy as np
import json
from monai.networks.nets import DenseNet121
import torch
from monai.data import DataLoader, ImageDataset
from monai.transforms import RandRotate90, Resize, EnsureChannelFirst, Compose, ScaleIntensity,RandAxisFlip
import os
from tqdm import tqdm
from sklearn.preprocessing import label_binarize


# %%
def get_results(acc_list,auc_list,recall_list,precision_list):    
    acc_list = np.array(acc_list)
    auc_list = np.array(auc_list)
    recall_list = np.array(recall_list)
    precision_list = np.array(precision_list)
    print(f'Recall, Average:{recall_list.mean():.4f}, Std:{recall_list.std():.4f}')
    print(f'precision, Average:{precision_list.mean():.4f}, Std:{precision_list.std():.4f}')
    print(f'Accuracy, Average:{acc_list.mean():.4f}, Std:{acc_list.std():.4f}')
    print(f'AUC, Average:{auc_list.mean():.4f}, Std:{auc_list.std():.4f}')
    
    
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

# %%
def get_dl_probabilities(dl_model, test_dataloader):
    prediction_list = []
    y_list = []
    with torch.no_grad(): 
        progress_bar = tqdm(test_dataloader, desc="Testing")
        
    for X, y in progress_bar:
        y_list.extend(y)
        X = X.to('cuda')
        pred = dl_model(X)
        prediction_list.append(torch.nn.functional.softmax(pred, dim=-1).cpu().detach().numpy().reshape(2))
        # prediction_list.append(torch.nn.functional.softmax(pred, dim=-1).cpu().numpy())
        
    return np.array(prediction_list)


train_test_info = 'Train_Test_4'

dataset_dtl_path = f'/home/pyq6817/IPMN-Radiomics-Plus-Deeplearning/{train_test_info}.json'
# deep learning input
input_path = '/data/Ziliang/IPMN_cysts_20240909/deeplearning/ROI'
label_path = '/home/pyq6817/IPMN-Radiomics-Plus-Deeplearning/labels.csv'
dl_model_dir = '/data/Ziliang/IPMN_cysts_20240909/DenseNet121_weights'


# Get features
feature_path = '/home/pyq6817/IPMN-Radiomics-Plus-Deeplearning/radiomics/'
data = pd.read_csv(os.path.join(feature_path,'Data/2D_t2/all.csv'))
data["Label"] = data['Label'].apply(lambda x: x-1)
data["Name"] = data['Name'].str.replace(r'^IU_', 'IUC_', regex=True)

train_list,val_list,test_list = get_tr_vl_ts_list(dataset_dtl=dataset_dtl_path,fold=0)
# train_data = data[data['Name'].isin(train_list)]
test_data = data[data['Name'].isin(test_list)] # This is for validation
# val_data = data[data['Name'].isin(val_list)]
train_val_data = data[data['Name'].isin(val_list+train_list)]

# Use train and val data to build scalar



# select 
x_test_raw = test_data.drop(columns=['Center','Name','Label'])
y_test = test_data[["Label"]]
# x_test_scaled = pd.DataFrame(standard_scaler.transform(x_test_raw),columns=x_test_raw.columns)
# x_test = x_test_scaled[feature_list]
test_data['path'] =test_data['Name'].apply(lambda x: os.path.join(input_path, x+'.nii.gz'))
test_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96))])
# %%
models = []
test_acc_list = []
test_auc_list = []
test_recall_list = []
test_precision_list = []
val_acc_list = []
val_auc_list = []
val_recall_list = []
val_precision_list = []

# ------------------------

for fold in range(5):
    train_list,val_list,test_list = get_tr_vl_ts_list(dataset_dtl=dataset_dtl_path,fold=fold)

    train_data = data[data['Name'].isin(train_list)]
    val_data = data[data['Name'].isin(val_list)]

    x_train_raw = train_data.drop(columns=['Center','Name','Label'])
    # y_train = train_data[["Label"]]
    # x_train_scaled = pd.DataFrame(standard_scaler.transform(x_train_raw),columns=x_train_raw.columns)
    # x_train = x_train_scaled[feature_list]
    train_data['path'] = train_data['Name'].apply(lambda x: os.path.join(input_path, x+'.nii.gz'))

    x_val_raw = val_data.drop(columns=['Center','Name','Label'])
    y_val = val_data[["Label"]]
    # x_val_scaled = pd.DataFrame(standard_scaler.transform(x_val_raw),columns=x_val_raw.columns)
    # x_val = x_val_scaled[feature_list]
    val_data['path'] = val_data['Name'].apply(lambda x: os.path.join(input_path, x+'.nii.gz'))

    densenet = DenseNet121(
            spatial_dims=3,  # 3D input
            in_channels=1,   # Typically for grayscale (e.g., MRI/CT scans), change to 3 for RGB
            out_channels=2   # Adjust for binary or multi-class segmentation/classification
        )
    
    densenet.load_state_dict(torch.load(os.path.join(dl_model_dir,f'model_loss_{train_test_info}_fold{fold}.pth'), map_location='cpu', weights_only=True))
    densenet.to('cuda')
    
    
    test_ds = ImageDataset(image_files=test_data['path'].to_list(), labels=test_data['Label'].to_list(), transform=test_transforms)
    val_ds = ImageDataset(image_files=val_data['path'].to_list(), labels=val_data['Label'].to_list(), transform=test_transforms)
    
    
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False,num_workers=1)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False,num_workers=1)
    
    
    # ------------------validation-----------
    dl_prob = get_dl_probabilities(densenet,val_dataloader)
    y_prob = dl_prob
    y_pred = np.argmax(dl_prob, axis=1)
    val_accuracy = accuracy_score(y_val, y_pred)
    val_acc_list.append(val_accuracy)
    val_auc = roc_auc_score(y_val, y_prob[:,1],average='macro')
    val_auc_list.append(val_auc)
    
    dl_prob = get_dl_probabilities(densenet,test_dataloader)
    y_prob = dl_prob
    y_pred = np.argmax(dl_prob, axis=1)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_acc_list.append(test_accuracy)
    test_auc = roc_auc_score(y_test, y_prob[:,1],average='macro')
    test_auc_list.append(test_auc)
    print(f"Test {fold} - ACC: {test_accuracy:.4f}, AUC: {test_auc:.4f}")
    print("-" * 40)
print('Validation set')
get_results(val_acc_list,val_auc_list,val_recall_list,val_precision_list)
print("Test set")
get_results(test_acc_list,test_auc_list,test_recall_list,test_precision_list)
print('================================================')




