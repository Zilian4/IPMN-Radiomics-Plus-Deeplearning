import os
import argparse
import json
from tqdm import tqdm
import torch
import torch.nn as nn
from data_loader import get_data_list, get_fold
from model import get_model
from seed import seed_everything
from sklearn.metrics import roc_auc_score
from monai.data import DataLoader, ImageDataset
from monai.transforms import RandRotate90, Resize, EnsureChannelFirst, Compose, ScaleIntensity, RandAxisFlip

def train_fn(dataloader, model, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    total_correct = 0
    batch_count = 0
    sample_count = 0
    progress_bar = tqdm(dataloader, desc="Training")
    for batch, (X, y) in enumerate(progress_bar):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        correct = (pred.argmax(1) == y).type(torch.float).sum().item()
        total_correct += correct
        batch_count += 1
        sample_count += len(X)
        progress_bar.set_postfix(lr=optimizer.param_groups[0]['lr'], loss=f"{loss.item():.4f}", loss_avg = f"{total_loss/batch_count:.4f}", acc=f"{correct/len(X):.4f}", acc_avg = f"{total_correct/sample_count:.4f}")   
    return {'loss': total_loss/batch_count, 'acc': total_correct/sample_count}

def test_fn(dataloader, model, loss_fn, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    batch_count = 0
    sample_count = 0
    y_all = []
    pred_all = []
    with torch.no_grad(): 
        progress_bar = tqdm(dataloader, desc="Testing")
        for X, y in progress_bar:
            y_all.extend(y)
            X, y = X.to(device), y.to(device)
            pred = model(X)
            pred_all.extend(torch.nn.functional.softmax(pred, dim=-1)[:, 1].cpu().numpy())
            loss = loss_fn(pred, y)
            total_loss += loss.item()
            correct = (pred.argmax(1) == y).type(torch.float).sum().item()
            total_correct += correct
            batch_count += 1
            sample_count += len(X)
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", loss_avg = f"{total_loss/batch_count:.4f}", acc=f"{correct/len(X):.4f}", acc_avg = f"{total_correct/sample_count:.4f}")
    auc_score = roc_auc_score(y_all, pred_all)
    return {'loss': total_loss/batch_count, 'acc': total_correct/sample_count, 'auc': auc_score}, {'true': y_all, 'pred': pred_all}
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PanSeg Training.")
    parser.add_argument("--data-path", default="/dataset/IPMN_Classification/", type=str, help="dataset path")
    parser.add_argument("--output-dir", default="./saved", type=str, help="path to save outputs")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("-b", "--batch-size", default=16, type=int, help="batch size")
    parser.add_argument("-j", "--workers", default=0, type=int, metavar="N", help="number of data loading workers")
    parser.add_argument("--epochs", default=100, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--lr", default=1e-3, type=float, help="initial learning rate")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--t", default=1, type=int, help="modality (must be 1 or 2)")
    parser.add_argument("-f", "--fold", default=0, type=int, help="fold id for cross validation  (must be 0 to 3)")
    parser.add_argument("-s", "--seed", default=None, type=int, metavar="N", help="Seed")
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, 't'+str(args.t), 'fold'+str(args.fold))

    if args.seed:
        seed_everything(args.seed)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    device = torch.device(args.device)
    image_lists = []
    label_lists = []
    train_ds = []
    test_ds = []
    train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96)), RandAxisFlip(), RandRotate90()])
    test_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96))])
    image_list, label_list = get_data_list(root=args.data_path, t = args.t)
    train_image, train_label, test_image, test_label = get_fold(image_list, label_list, fold = args.fold)

    train_ds.append(ImageDataset(image_files=train_image, labels=train_label, transform=train_transforms))
    test_ds.append(ImageDataset(image_files=test_image, labels=test_label, transform=test_transforms))
    train_dataloader = DataLoader(torch.utils.data.ConcatDataset(train_ds), batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_dataloader = []
    test_dataloader.append(DataLoader(test_ds[c], batch_size=args.batch_size, shuffle=False, num_workers=args.workers))
    n_test_dataloader = sum([len(test_dataloader[i]) for i in range(n_center)])
    n_test_ds = sum([len(test_ds[i]) for i in range(n_center)])
    
    model = get_model(out_channels = 2)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    
    log = {'train_loss':[], 'train_acc':[], 'test_loss':[[] for i in range(n_center+1)], 'test_acc':[[] for i in range(n_center+1)], 'test_auc':[[] for i in range(n_center+1)]}    
    best_ind = {'loss': 0, 'acc': 0, 'auc': 0}
    for epoch in range(args.epochs):
        epoch_log = train_fn(train_dataloader, model, loss_fn, optimizer, device)
        for metric in ['loss', 'acc']:
            log['train_'+metric].append(epoch_log[metric])
        scheduler.step()
        y_all = []
        pred_all = []
        for c in range(n_center):
            epoch_log, epoch_y = test_fn(test_dataloader[c], model, loss_fn, device)
            for metric in ['loss', 'acc', 'auc']:
                log['test_'+metric][c].append(epoch_log[metric])
            y_all.extend(epoch_y['true'])
            pred_all.extend(epoch_y['pred'])
        
        log['test_loss'][-1].append(sum([log['test_loss'][i][-1]*len(test_dataloader[i]) for i in range(n_center)])/n_test_dataloader)
        log['test_acc'][-1].append(sum([log['test_acc'][i][-1]*len(test_ds[i]) for i in range(n_center)])/n_test_ds)
        log['test_auc'][-1].append(roc_auc_score(y_all, pred_all))
        print(f"Epoch {epoch+1} train loss {log['train_loss'][-1]:.4f} acc {log['train_acc'][-1]:.4f}")
        for c in range(n_center):
            print(f"Center {c+1} test loss {log['test_loss'][c][-1]:.4f} acc {log['test_acc'][c][-1]:.4f} auc {log['test_auc'][c][-1]:.4f}")
        print(f"Global test loss {log['test_loss'][-1][-1]:.4f} acc {log['test_acc'][-1][-1]:.4f} auc {log['test_auc'][-1][-1]:.4f}")
        
        torch.save(model.state_dict(), os.path.join(args.output_dir, "checkpoint.pth"))
        with open(os.path.join(args.output_dir, "log.json"), 'w') as f:
            json.dump(log, f)
        if log['test_loss'][-1][-1] <= min(log['test_loss'][-1]):    
            best_ind['loss'] = epoch
            torch.save(model.state_dict(), os.path.join(args.output_dir, "model_loss.pth"))
        for metric, metric2 in (['acc', 'auc'], ['auc', 'acc']): # save model when metric improves or both metric and metric2 are maximized. The second condition avoids one best again but another worse
            if epoch == 0 or log['test_'+metric][-1][-1] > max(log['test_'+metric][-1][:-1]) or log['test_'+metric][-1][-1] == max(log['test_'+metric][-1][:-1]) and log['test_'+metric2][-1][-1] >= max(log['test_'+metric2][-1][:-1]):
                best_ind[metric] = epoch
                torch.save(model.state_dict(), os.path.join(args.output_dir, 'model_'+metric+'.pth'))
    
    for metric in ['acc', 'auc']:
        print(f"{metric} best model reached at epoch {best_ind[metric]+1}")
        for c in range(n_center):
            print(f"Center {c+1} test loss {log['test_loss'][c][best_ind[metric]]:.4f} acc {log['test_acc'][c][best_ind[metric]]:.4f} auc {log['test_auc'][c][best_ind[metric]]:.4f}")
        print(f"Global test loss {log['test_loss'][-1][best_ind[metric]]:.4f} acc {log['test_acc'][-1][best_ind[metric]]:.4f} auc {log['test_auc'][-1][best_ind[metric]]:.4f}")