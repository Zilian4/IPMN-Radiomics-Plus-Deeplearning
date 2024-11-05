import numpy as np
def get_results(acc_list,auc_list,recall_list,precision_list):    
    acc_list = np.array(acc_list)
    auc_list = np.array(auc_list)
    recall_list = np.array(recall_list)
    precision_list = np.array(precision_list)
    print(f'Recall, Average:{recall_list.mean():.4f}, Std:{recall_list.std():.4f}')
    print(f'precision, Average:{precision_list.mean():.4f}, Std:{precision_list.std():.4f}')
    print(f'Accuracy, Average:{acc_list.mean():.4f}, Std:{acc_list.std():.4f}')
    print(f'AUC, Average:{auc_list.mean():.4f}, Std:{auc_list.std():.4f}')