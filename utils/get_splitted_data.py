import json
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