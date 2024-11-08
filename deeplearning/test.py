import os
import argparse
import torch
import torch.nn as nn
from data_loader import get_data_list, get_fold
from model import get_model
from sklearn.metrics import roc_auc_score, roc_curve
from monai.data import DataLoader, ImageDataset
from monai.transforms import Resize, EnsureChannelFirst, Compose, ScaleIntensity
from train import test_fn
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PanSeg Training.")
    parser.add_argument("--data-path", default="/dataset/IPMN_Classification/", type=str, help="dataset path")
    parser.add_argument("--output-dir", default="./saved", type=str, help="path to save outputs")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("-b", "--batch-size", default=32, type=int, help="batch size")
    parser.add_argument("-j", "--workers", default=0, type=int, metavar="N", help="number of data loading workers")
    parser.add_argument("--resume", default="model_auc.pth", type=str, help="path of checkpoint")
    parser.add_argument("--t", default=1, type=int, help="modality (must be 1 or 2)")
    parser.add_argument("-f", "--fold", default=0, type=int, help="fold id for cross validation  (must be 0 to 3)")
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, 't'+str(args.t), 'fold'+str(args.fold))
    
    device = torch.device(args.device)
    n_center = 7
    image_lists = []
    label_lists = []
    test_ds = []
    test_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96))])
    for c in range(n_center):
        image_list, label_list = get_data_list(root=args.data_path, t = args.t, center=c)
        _, _, test_image, test_label = get_fold(image_list, label_list, fold = args.fold)
        print(f"Center {c+1} has {len(test_image)} testing images")
        test_ds.append(ImageDataset(image_files=test_image, labels=test_label, transform=test_transforms))
    test_dataloader = []
    for c in range(n_center):
        test_dataloader.append(DataLoader(test_ds[c], batch_size=args.batch_size, shuffle=False, num_workers=args.workers))
    n_test_dataloader = sum([len(test_dataloader[i]) for i in range(n_center)])
    n_test_ds = sum([len(test_ds[i]) for i in range(n_center)])

    model = get_model(out_channels = 2)
    model.load_state_dict(torch.load(os.path.join(args.output_dir, args.resume), map_location='cpu', weights_only=True))
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    
    log = {'test_loss':[[] for i in range(n_center+1)], 'test_acc':[[] for i in range(n_center+1)], 'test_auc':[[] for i in range(n_center+1)]}    
    y_all = []
    pred_all = []
    plt.figure(figsize=(8,6))
    for c in range(n_center):
        epoch_log, epoch_y = test_fn(test_dataloader[c], model, loss_fn, device)
        for metric in ['loss', 'acc', 'auc']:
            log['test_'+metric][c].append(epoch_log[metric])
        y_all.extend(epoch_y['true'])
        pred_all.extend(epoch_y['pred'])
        fpr, tpr, thresholds = roc_curve(epoch_y['true'], epoch_y['pred'])
        plt.plot(fpr, tpr, lw=2, label=f'Center {c+1}')
    
    log['test_loss'][-1].append(sum([log['test_loss'][i][-1]*len(test_dataloader[i]) for i in range(n_center)])/n_test_dataloader)
    log['test_acc'][-1].append(sum([log['test_acc'][i][-1]*len(test_ds[i]) for i in range(n_center)])/n_test_ds)
    log['test_auc'][-1].append(roc_auc_score(y_all, pred_all))
    fpr, tpr, thresholds = roc_curve(y_all, pred_all)
    plt.plot(fpr, tpr, lw=4, label='Global')
    for c in range(n_center):
        print(f"Center {c+1} test loss {log['test_loss'][c][-1]:.4f} acc {log['test_acc'][c][-1]:.4f} auc {log['test_auc'][c][-1]:.4f}")
    print(f"Global test loss {log['test_loss'][-1][-1]:.4f} acc {log['test_acc'][-1][-1]:.4f} auc {log['test_auc'][-1][-1]:.4f}")
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.pdf", format="pdf", bbox_inches='tight')
    #plt.show()