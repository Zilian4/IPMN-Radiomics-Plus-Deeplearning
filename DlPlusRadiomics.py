# %%
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,roc_auc_score,classification_report,recall_score,precision_score,roc_curve
from sklearn.svm import SVC
import numpy as np
import fusion_model
import matplotlib.pyplot as plt
import json
from monai.networks.nets import DenseNet121
import torch
from monai.data import DataLoader, ImageDataset
from monai.transforms import RandRotate90, Resize, EnsureChannelFirst, Compose, ScaleIntensity,RandAxisFlip
import os
from tqdm import tqdm
from joblib import load
import pickle
from sklearn.linear_model import LogisticRegression
from fusion_model import FusionModel
from sklearn.metrics import confusion_matrix

def sensitivity_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)  
    specificity = tn / (tn + fp)  
    return sensitivity, specificity

def get_results(acc_list,auc_list,recall_list,specifity_list):    
    acc_list = np.array(acc_list)
    auc_list = np.array(auc_list)
    recall_list = np.array(recall_list)
    specifity_list = np.array(specifity_list)
    print(f'Sensitivity, Average: {recall_list.mean():.4f}±{recall_list.std():.4f}')
    print(f'Specificity, Average: {specifity_list.mean():.4f}±{specifity_list.std():.4f}')
    print(f'Accuracy, Average: {acc_list.mean():.4f}±{acc_list.std():.4f}')
    print(f'AUC, Average: { auc_list.mean():.4f}±{auc_list.std():.4f}')

def get_data_list(dataset_dtl, images_path,labels_path,fold):

    image_list = []
    label_list = []
    df = pd.read_csv(labels_path)
    df_cleaned = df.dropna(subset=[df.columns[1]]) # remove NaN

    train_list,val_list, test_list = get_tr_vl_ts_list(dataset_dtl,fold)
    df_train = df_cleaned[df_cleaned['name'].isin(train_list)]
    df_val = df_cleaned[df_cleaned['name'].isin(val_list)]
    df_test = df_cleaned[df_cleaned['name'].isin(test_list)]

    df_train['path'] = df_train['name'].apply(lambda x: os.path.join(images_path, x+'.nii.gz'))
    df_test['path'] = df_test['name'].apply(lambda x: os.path.join(images_path, x+'.nii.gz'))
    df_val['path'] = df_val['name'].apply(lambda x: os.path.join(images_path, x+'.nii.gz'))
    
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
    with torch.no_grad(): 
        progress_bar = tqdm(test_dataloader, desc="Testing")
    for X, y in progress_bar:
        X = X.to('cuda')
        pred = dl_model(X)
        prediction_list.append(torch.nn.functional.softmax(pred, dim=-1).cpu().detach().numpy().reshape(2))
    return np.array(prediction_list)

def get_rf_probabilities(model, radiomics_features_normed):
    probs = model.predict_proba(radiomics_features_normed)  # Output probabilities for each class
    return probs

# %%
def train_fusion_model(dl_model, rf_model, image_dataloader, radiomics_data,labels):
    dl_probs = get_densenet_probabilities(dl_model, image_dataloader)
    rf_probs = get_rf_probabilities(rf_model, radiomics_data)
    fusion_features = np.hstack([dl_probs, rf_probs])
    # fusion_features = np.hstack([densenet_probs[:,1].reshape(len(labels),1), rf_probs[:,1].reshape(len(labels),1)])
    print(fusion_features.shape)
    model = LogisticRegression()
    model.fit(fusion_features, labels)
    return model

def predict_with_fusion_model(densenet_model, rf_model, fusion_model,image_dataloader, radiomics_data):
    densenet_probs = get_dl_probabilities(densenet_model, image_dataloader)
    rf_probs = get_rf_probabilities(rf_model, radiomics_data)
    # Combine probabilities for the fusion model  
    fusion_features = np.hstack([densenet_probs, rf_probs])
    # fusion_features = np.hstack([densenet_probs[:,1].reshape(len(radiomics_data),1), rf_probs[:,1].reshape(len(radiomics_data),1)])
    fusion_predictoin = fusion_model.predict(fusion_features)
    fusion_proba = fusion_model.predict_proba(fusion_features)
    return fusion_predictoin,fusion_proba

# %%
train_test_info = 'Train_Test_1'
<<<<<<< HEAD
feature_dim =  '3D'
=======
>>>>>>> 0d7189f2d5bb3f437989ebeeb2e811fd8b186bdc

dataset_dtl_path = f'/home/pyq6817/IPMN-Radiomics-Plus-Deeplearning/{train_test_info}.json'

# deep learning input
input_path = '/data/Ziliang/IPMN_cysts_20240909/deeplearning/ROI'
label_path = '/home/pyq6817/IPMN-Radiomics-Plus-Deeplearning/labels.csv'
feature_path = '/home/pyq6817/IPMN-Radiomics-Plus-Deeplearning/radiomics'
dl_model_dir = '/data/Ziliang/IPMN_cysts_20240909/DenseNet121_weights'
radiomcis_model_dir = f'/home/pyq6817/IPMN-Radiomics-Plus-Deeplearning/radiomics/trained_models/{feature_dim}'



if feature_dim=='2D':
    if train_test_info =='Train_Test_4':
        # Test 4
        feature_list = [
        'skewness-Laws R5S5',
        'Collage_kurt_MaximalCorrelationCoefficient_1_nb_4_ws_5',
        'Collage_skew_InformationMeasureOfCorrelation2_1_nb_8_ws_3',
        'Collage_kurt_Contrast_1_nb_8_ws_7',
        'Collage_median_MaximalCorrelationCoefficient_1_nb_32_ws_3',
        'Collage_skew_Correlation_1_nb_16_ws_5',
        'Collage_skew_MaximalCorrelationCoefficient_1_nb_4_ws_3',
        'Collage_skew_SumVariance_1_nb_8_ws_5',
        'median-Laws E5L5',
        'Collage_skew_Entropy_1_nb_16_ws_7',
        'skewness-Laws S5E5',
        'skewness-Laws W5L5',
        'Collage_skew_DifferenceEntropy_1_nb_4_ws_3',
        'Collage_var_Contrast_1_nb_8_ws_7',
        'Collage_kurt_SumEntropy_1_nb_16_ws_5',
        ]
    elif train_test_info =='Train_Test_3':
        # test3
        feature_list = ['Collage_skew_InformationMeasureOfCorrelation2_1_nb_8_ws_3',
        'skewness-Laws S5E5',
        'skewness-Laws R5S5',
        'Collage_skew_InformationMeasureOfCorrelation2_1_nb_16_ws_5',
        'Collage_skew_Correlation_1_nb_8_ws_5',
        'median-Laws E5L5',
        'median-Laws W5S5',
        'Collage_skew_MaximalCorrelationCoefficient_1_nb_4_ws_3']
    elif train_test_info =='Train_Test_2':
        # test2
        feature_list = ['Collage_kurt_MaximalCorrelationCoefficient_1_nb_4_ws_5',
        'median-Laws E5L5',
        'Collage_skew_InformationMeasureOfCorrelation2_1_nb_8_ws_3',
        'Collage_var_InformationMeasureOfCorrelation2_1_nb_16_ws_3',
        'skewness-Laws R5S5',
        'Collage_skew_MaximalCorrelationCoefficient_1_nb_4_ws_3',
        'skewness-Laws S5E5',
        'median-Laws W5S5',
        'Collage_kurt_MaximalCorrelationCoefficient_1_nb_4_ws_7',
        'Collage_median_SumEntropy_1_nb_8_ws_3',
        'Collage_kurt_SumEntropy_1_nb_16_ws_5',
        'Collage_skew_SumVariance_1_nb_4_ws_7']
    elif train_test_info =='Train_Test_1':
        feature_list = ['Collage_skew_InformationMeasureOfCorrelation2_1_nb_8_ws_3',
        'Collage_var_InformationMeasureOfCorrelation2_1_nb_16_ws_3',
        'Collage_kurt_Contrast_1_nb_16_ws_7',
        'Collage_skew_MaximalCorrelationCoefficient_1_nb_4_ws_3',
        'Collage_kurt_SumAverage_1_nb_16_ws_3',
        'Collage_kurt_InformationMeasureOfCorrelation2_1_nb_16_ws_3',
        'Collage_kurt_MaximalCorrelationCoefficient_1_nb_4_ws_5',
        'median-Laws E5L5',
        'Collage_kurt_SumEntropy_1_nb_16_ws_3',
        'skewness-Laws S5E5',
        'kurtosis-Haralick correlation ws=7 n=4',
        'Collage_kurt_InformationMeasureOfCorrelation2_1_nb_64_ws_5',
        'Collage_median_DifferenceVariance_1_nb_64_ws_7',
        'skewness-Laws E5S5',
        'skewness-Gradient sobelxy']
elif feature_dim=='3D':
    if train_test_info =='Train_Test_1':
        feature_list = [
        'Collage_var_InformationMeasureOfCorrelation2_1_nb_8_ws_5',
        'Collage_var_SumEntropy_1_nb_32_ws_7',
        'skewness-Haralick info1 ws=5 n=4',
        'skewness-Laws  S5L5E5',
        'median-Laws  E5E5E5',
        'Collage_skew_DifferenceEntropy_1_nb_32_ws_7',
        'median-Laws  R5E5S5',
        'median-Laws  W5E5S5',
        'Collage_kurt_InformationMeasureOfCorrelation2_1_nb_4_ws_5',
        'median-Laws  E5L5S5',
                ]
    elif train_test_info =='Train_Test_2':
        # test3
        feature_list = ['Collage_var_InformationMeasureOfCorrelation2_1_nb_16_ws_3',
        'median-Laws  E5L5S5',
        'median-Laws  E5E5E5',
        'skewness-Laws  R5R5L5',
        'median-Laws  L3S3L3',
        'skewness-Laws  S3E3S3',
        'Collage_kurt_Correlation_2_nb_64_ws_5',
        'Collage_kurt_SumAverage_1_nb_32_ws_7',
        'skewness-Laws  S5L5E5',
        'skewness-Laws  S3E3E3',
        'skewness-Laws  W5L5S5',
        'kurtosis-Laws  L5R5L5',]
    elif train_test_info =='Train_Test_3':
        # test2
        feature_list = ['skewness-Laws  S5W5R5',
        'skewness-Laws  S5L5E5',
        'median-Laws  L3S3L3',
        'median-Laws  S3E3E3',
        'Collage_skew_MaximalCorrelationCoefficient_1_nb_4_ws_5',
        'median-Laws  R5E5S5',
        'skewness-Laws  W5L5S5',
        'Collage_kurt_SumAverage_1_nb_32_ws_7',
        'Collage_var_InformationMeasureOfCorrelation2_1_nb_16_ws_3',
        'median-Laws  E5L5S5',
        'Collage_kurt_InformationMeasureOfCorrelation2_1_nb_4_ws_5',
        'Collage_kurt_InformationMeasureOfCorrelation2_2_nb_16_ws_7',
        'Collage_kurt_InformationMeasureOfCorrelation2_1_nb_16_ws_3',
        'median-Laws  E5E5E5',
        'median-Laws  W5E5S5',]
    elif train_test_info =='Train_Test_4':
        feature_list = ['skewness-Laws  E5E5R5',
        'skewness-Laws  E3S3E3',
        'skewness-Laws  R5E5E5',
        'skewness-Haralick correlation ws=5 n=4',
        'skewness-Laws  E3E3E3',
        'Collage_kurt_SumAverage_1_nb_16_ws_7',
        'Collage_kurt_MaximalCorrelationCoefficient_1_nb_4_ws_7',
        'skewness-Haralick info1 ws=5 n=64',
        'Collage_kurt_InformationMeasureOfCorrelation2_2_nb_16_ws_7',
        'skewness-Laws  R5R5L5',
        'Collage_var_SumEntropy_1_nb_32_ws_7',
        'median-Laws  R5E5L5']

# %%
# Get features
data = pd.read_csv(os.path.join(feature_path,f'Data/{feature_dim.upper()}_t2/all.csv'))
data["Label"] = data['Label'].apply(lambda x: x-1)
data["Name"] = data['Name'].str.replace(r'^IU_', 'IUC_', regex=True)

train_list,val_list,test_list = get_tr_vl_ts_list(dataset_dtl=dataset_dtl_path,fold=0)
# train_data = data[data['Name'].isin(train_list)]
test_data = data[data['Name'].isin(test_list)] # This is for validation
val_data = data[data['Name'].isin(val_list)]
train_val_data = data[data['Name'].isin(val_list+train_list)]

# Use train and val data to build scalar
standard_scaler = StandardScaler()

train_val_data.drop(columns=['Center','Name','Label'])
x_train_val_raw = train_val_data.drop(columns=['Center','Name','Label'])
y_train_val = train_val_data['Label']
x_train_val_scaled = pd.DataFrame(standard_scaler.fit_transform(x_train_val_raw),columns=x_train_val_raw.columns)
x_train_val = x_train_val_scaled[feature_list]

val_data.drop(columns=['Center','Name','Label'])
val_data_raw = val_data.drop(columns=['Center','Name','Label'])
y_val = val_data['Label']
x_val_scaled = pd.DataFrame(standard_scaler.fit_transform(x_train_val_raw),columns=x_train_val_raw.columns)
x_val = x_val_scaled[feature_list]

# select 
x_test_raw = test_data.drop(columns=['Center','Name','Label'])
y_test = test_data[["Label"]]
x_test_scaled = pd.DataFrame(standard_scaler.transform(x_test_raw),columns=x_test_raw.columns)
x_test = x_test_scaled[feature_list]

test_data['path'] =test_data['Name'].apply(lambda x: os.path.join(input_path, x+'.nii.gz'))
prediction_record_test = test_data[['Name','Label']]
prediction_record_val = val_data[['Name','Label']]
# print(prediction_record_val.shape)

# %% [markdown]
# Deep learning settings

# %%
test_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96))])

# %%
models = []
test_acc_list = []
test_auc_list = []
test_sensitivity_list = []
test_specificity_list = []


val_acc_list = []
val_auc_list = []
val_sensitivity_list = []
val_specificity_list = []
# ------------------------



roc_list = []

for fold in range(5):
    radiomics_model = load(os.path.join(radiomcis_model_dir,f'{train_test_info}/{train_test_info}_Fold{fold+1}.joblib'))
    train_list,val_list,_ = get_tr_vl_ts_list(dataset_dtl=dataset_dtl_path,fold=fold)

    train_data = data[data['Name'].isin(train_list)]
    val_data = data[data['Name'].isin(val_list)]

    x_train_raw = train_data.drop(columns=['Center','Name','Label'])
    # y_train = train_data[["Label"]]
    x_train_scaled = pd.DataFrame(standard_scaler.transform(x_train_raw),columns=x_train_raw.columns)
    x_train = x_train_scaled[feature_list]
    train_data['path'] = train_data['Name'].apply(lambda x: os.path.join(input_path, x+'.nii.gz'))

    x_val_raw = val_data.drop(columns=['Center','Name','Label'])
    y_val = val_data[["Label"]]
    x_val_scaled = pd.DataFrame(standard_scaler.transform(x_val_raw),columns=x_val_raw.columns)
    x_val = x_val_scaled[feature_list]
    val_data['path'] = val_data['Name'].apply(lambda x: os.path.join(input_path, x+'.nii.gz'))

    densenet = DenseNet121(
            spatial_dims=3,  # 3D input
            in_channels=1,   # Typically for grayscale (e.g., MRI/CT scans), change to 3 for RGB
            out_channels=2   # Adjust for binary or multi-class segmentation/classification
        )
    
    densenet.load_state_dict(torch.load(os.path.join(dl_model_dir,f'model_auc_{train_test_info}_fold{fold}.pth'), map_location='cpu', weights_only=True))
    densenet.to('cuda')
    
    train_ds = ImageDataset(image_files=train_data['path'].to_list(), labels=train_data['Label'].to_list(), transform=test_transforms)
    test_ds = ImageDataset(image_files=test_data['path'].to_list(), labels=test_data['Label'].to_list(), transform=test_transforms)
    val_ds = ImageDataset(image_files=val_data['path'].to_list(), labels=val_data['Label'].to_list(), transform=test_transforms)
    
    
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False,num_workers=1)
    train_dataloader = DataLoader(train_ds, batch_size=1, shuffle=False,num_workers=1)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False,num_workers=1)
    
    # ----------Probability of training set---------
    dl_prob = get_dl_probabilities(densenet,train_dataloader)
    radiomics_prob = get_rf_probabilities(radiomics_model,x_train)
    probs = np.hstack([radiomics_prob, dl_prob])
    
    print('5 CV get best t and k.....')    
    # Get the best t and k in the CV set
    fm = FusionModel()
    param_grid = {
    'k': np.linspace(0, 1, 50),  # 50 values from 0 to 1
    't': np.linspace(0, 1, 50)  # 50 values from 0 to 1
    }
    grid_search = GridSearchCV(estimator=fm, param_grid=param_grid, scoring='roc_auc')
    grid_search.fit(probs,train_data['Label'])
    fm = FusionModel(**grid_search.best_params_)
    
    # ------------------validation-----------
    dl_prob = get_dl_probabilities(densenet,val_dataloader)
    radiomics_prob = get_rf_probabilities(radiomics_model,x_val)
    probs = np.hstack([radiomics_prob, dl_prob])
    y_val_prob = fm.predict_proba(probs)
      
    val_auc = roc_auc_score(y_val, y_val_prob[:, 1])
    val_auc_list.append(val_auc)
    fm.fit_threshold(y_val, probs)
    print('Fusion weights: k: {}, t: {}'.format(fm.k,fm.t),'\n')
    print('Best threshold: ',fm.threshold)
    print("youden_statistic: ",fm.best_Youden_statistic)
    y_val_pred = fm.predict(probs)
    val_sensitivity, val_specificity = sensitivity_specificity(y_val, y_val_pred)
    # print("Sensitivity: ",val_sensitivity)
    # print("Specificity: ",val_specificity)
    val_sensitivity_list.append(val_sensitivity)
    val_specificity_list.append(val_specificity)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_acc_list.append(val_accuracy)
    
    # fusion_model = train_fusion_model(dl_model=densenet, rf_model=radiomics_model, 
    #                    image_dataloader=train_dataloader, radiomics_data=x_train,
    #                    labels=train_data['Label'])
    
    

    # y_pred,y_prob = predict_with_fusion_model(densenet_model = densenet, rf_model = radiomics_model, fusion_model = fusion_model,
    #                              image_dataloader = test_dataloader, radiomics_data = x_test)

    # prediction_record_val[f'Fold{fold}_prediction'] = y_val_pred
    # ------------------test-----------
    dl_prob = get_dl_probabilities(densenet,test_dataloader)
    radiomics_prob = get_rf_probabilities(radiomics_model,x_test)
    probs = np.hstack([radiomics_prob, dl_prob])
    y_prob = fm.predict_proba(probs)
    y_pred = fm.predict(probs) 
<<<<<<< HEAD
    prediction_record_test[f'Fold{fold}_prediction'] = y_pred
    
    # print("Deeplearning proba:\n",dl_prob)
    # print("Radiomics proba:\n",radiomics_prob)
    # print('Fusion weights: k: {}, t: {}'.format(fm.k,fm.t),'\n')
    # print("fusion proba:\n",y_prob)
=======
    
    
>>>>>>> 0d7189f2d5bb3f437989ebeeb2e811fd8b186bdc
    test_accuracy = accuracy_score(y_test, y_pred)
    test_acc_list.append(test_accuracy)
    test_auc = roc_auc_score(y_test, y_prob[:, 1])
    test_auc_list.append(test_auc)
<<<<<<< HEAD
    test_specificity, test_sensitivity = sensitivity_specificity(y_test, y_pred)
    test_sensitivity_list.append(test_sensitivity)
    test_specificity_list.append(test_specificity)
=======
>>>>>>> 0d7189f2d5bb3f437989ebeeb2e811fd8b186bdc
    # ---------ROC curve----------
    fpr, tpr, _ =roc_curve(y_test, y_prob[:, 1])
    roc_list.append([fpr, tpr, test_auc])
    
    print(f"Test {fold} - ACC: {test_accuracy:.4f}, AUC: {test_auc:.4f}")
    print("-" * 40)
    
<<<<<<< HEAD
    print(f"Val  {fold} - ACC: {val_accuracy:.4f}, AUC: {val_auc:.4f}")
    print("-" * 40)

# prediction_record_test.to_csv(f'/home/pyq6817/IPMN-Radiomics-Plus-Deeplearning/pred_record_test/{train_test_info}_{feature_dim}.csv',index=False)
# prediction_record_val.to_csv(f'/home/pyq6817/IPMN-Radiomics-Plus-Deeplearning/pred_record_val/{train_test_info}_{feature_dim}.csv',index=False)
=======
>>>>>>> 0d7189f2d5bb3f437989ebeeb2e811fd8b186bdc
print('Validation set')
get_results(val_acc_list,val_auc_list,val_sensitivity_list,val_specificity_list)
print("Test set")
get_results(test_acc_list,test_auc_list,test_sensitivity_list,test_specificity_list)
print('================================================')

<<<<<<< HEAD
# with open(f"/home/pyq6817/IPMN-Radiomics-Plus-Deeplearning/ROC_info/ROC_curve_data_{train_test_info}_{feature_dim}.pkl", "wb") as file:
#     pickle.dump(roc_list, file)
# print("ROC info saved 'data.pkl'")

=======
with open(f"ROC_curve_data_{train_test_info}.pkl", "wb") as file:
    pickle.dump(roc_list, file)
print("Data saved in 'data.pkl'")
>>>>>>> 0d7189f2d5bb3f437989ebeeb2e811fd8b186bdc
