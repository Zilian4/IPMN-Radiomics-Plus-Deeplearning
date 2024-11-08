import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from data_loader import get_data_list
from model import get_model
from monai.data import DataLoader, ImageDataset
from monai.transforms import RandRotate90, Resize, EnsureChannelFirst, Compose, ScaleIntensity,RandAxisFlip
from train import test

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PanSeg Training.")
    parser.add_argument("--data-path", default="/dataset/IPMN_Classification/", type=str, help="dataset path")
    parser.add_argument("--output-dir", default="./saved", type=str, help="path to save outputs")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("-b", "--batch-size", default=32, type=int, help="batch size")
    parser.add_argument("-j", "--workers", default=0, type=int, metavar="N", help="number of data loading workers")
    parser.add_argument("--split-ratio", default=0.8, type=float, help="training ratio")
    parser.add_argument("--split-seed", default=0, type=float, help="split seed")
    parser.add_argument("--resume", default="model_auc.pth", type=str, help="path of checkpoint")
    parser.add_argument("--t", default=1, type=int, help="modality (must be 1 or 2)")
    args = parser.parse_args()
        
    device = torch.device(args.device)
    image_list, label_list = get_data_list(root=args.data_path, t = args.t)
    split = int(np.floor(len(image_list) * args.split_ratio))
    indices = np.random.default_rng(seed=args.split_seed).permutation(len(image_list))
    train_idx, test_idx = list(indices[:split]), list(indices[split:])
    train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96)), RandAxisFlip(prob=0.5), RandRotate90()])
    test_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((96, 96, 96))])
    train_ds = ImageDataset(image_files=[image_list[i] for i in train_idx], labels=[label_list[i] for i in train_idx], transform=train_transforms)
    test_ds = ImageDataset(image_files=[image_list[i] for i in test_idx], labels=[label_list[i] for i in test_idx], transform=test_transforms)
    train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_dataloader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    model = get_model()
    model.load_state_dict(torch.load(os.path.join(args.output_dir, args.resume), map_location='cpu', weights_only=True))
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    loss, acc, auc = test(test_dataloader, model, loss_fn, device)
    print(f"Test loss {loss:.4f} test acc {acc:.4f}  test auc {auc:.4f}")
