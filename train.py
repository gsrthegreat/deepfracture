"""
train.py — optional fine-tuning script for FractureAI.

Usage:
    python train.py --model densenet121 --data_dir ./dataset --epochs 20

Dataset structure expected:
    dataset/
      train/
        Fractured/
        Normal/
      val/
        Fractured/
        Normal/

Saves best weights to weights/<model>.pth
"""

import os
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets
from sklearn.metrics import roc_auc_score

from model import _BUILDERS
from preprocess import _NORMALIZE

import torchvision.transforms as T
import cv2
from PIL import Image


# ── Transforms ────────────────────────────────────────────────────────────────
class CLAHETransform:
    def __call__(self, img):
        import cv2, numpy as np
        clahe  = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
        arr    = np.array(img)
        lab    = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        arr    = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(arr)


class UnsharpTransform:
    def __call__(self, img):
        import cv2, numpy as np
        arr      = np.array(img)
        blurred  = cv2.GaussianBlur(arr, (0, 0), 1.0)
        sharp    = cv2.addWeighted(arr, 1.8, blurred, -0.8, 0)
        return Image.fromarray(np.clip(sharp, 0, 255).astype(np.uint8))


def get_transforms(train: bool):
    base = [T.Resize((256, 256)), CLAHETransform(), UnsharpTransform()]
    if train:
        aug = [
            T.RandomCrop(224),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.2, contrast=0.4),
        ]
    else:
        aug = [T.CenterCrop(224)]
    return T.Compose(base + aug + [T.ToTensor(), _NORMALIZE])


# ── Helpers ───────────────────────────────────────────────────────────────────
def make_sampler(dataset):
    targets = np.array(dataset.targets)
    counts  = np.bincount(targets)
    weights = 1.0 / counts[targets]
    return WeightedRandomSampler(torch.DoubleTensor(weights), len(weights), replacement=True)


def get_pos_idx(classes):
    for i, c in enumerate(classes):
        if 'frac' in c.lower():
            return i
    return 0


def find_threshold(labels, probs, pos_idx, min_spec=0.60):
    bin_labels = (np.array(labels) == pos_idx).astype(int)
    best_t, best_j = 0.45, -1.0
    for t in np.linspace(0.30, 0.75, 46):
        preds = (np.array(probs) >= t).astype(int)
        tp = ((preds == 1) & (bin_labels == 1)).sum()
        fn = ((preds == 0) & (bin_labels == 1)).sum()
        tn = ((preds == 0) & (bin_labels == 0)).sum()
        fp = ((preds == 1) & (bin_labels == 0)).sum()
        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
        if spec < min_spec:
            continue
        j = sens + spec - 1.0
        if j > best_j:
            best_j, best_t = j, float(t)
    return best_t


# ── Main ──────────────────────────────────────────────────────────────────────
def train(model_key, data_dir, epochs, batch_size, lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"),
                                    transform=get_transforms(True))
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, "val"),
                                    transform=get_transforms(False))

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=make_sampler(train_ds), num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=0)

    classes = train_ds.classes
    pos_idx = get_pos_idx(classes)
    print(f"Classes: {classes} | Fractured idx: {pos_idx}")

    counts  = np.bincount(train_ds.targets)
    cw      = torch.FloatTensor(len(counts) / (len(counts) * counts)).to(device)
    cw[pos_idx] *= 1.5          # extra sensitivity boost
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)

    model, _ = _BUILDERS[model_key](num_classes=2)
    model.to(device)

    best_auc, best_wts = 0.0, copy.deepcopy(model.state_dict())
    os.makedirs("weights", exist_ok=True)

    # Phase 1: head only
    for param in model.parameters():
        param.requires_grad = False
    head_params = [p for n, p in model.named_parameters()
                   if any(k in n for k in ('classifier', 'fc'))]
    for p in head_params:
        p.requires_grad = True

    optimizer = optim.AdamW(head_params, lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(3, epochs // 4))

    phase1 = max(3, epochs // 4)
    phase2 = epochs - phase1

    print(f"\n=== Phase 1: head warmup ({phase1} epochs) ===")
    for epoch in range(phase1):
        _run_epoch(model, train_loader, criterion, optimizer, device, "train")
        auc = _run_epoch(model, val_loader, criterion, None, device, "val",
                         pos_idx=pos_idx)
        scheduler.step()
        print(f"  Epoch {epoch+1}/{phase1} | val AUC: {auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_wts = copy.deepcopy(model.state_dict())
            torch.save(best_wts, f"weights/{model_key}.pth")

    # Phase 2: full fine-tune
    for param in model.parameters():
        param.requires_grad = True
    optimizer = optim.AdamW(model.parameters(), lr=lr * 0.1, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase2)

    print(f"\n=== Phase 2: full fine-tune ({phase2} epochs) ===")
    for epoch in range(phase2):
        _run_epoch(model, train_loader, criterion, optimizer, device, "train")
        auc = _run_epoch(model, val_loader, criterion, None, device, "val",
                         pos_idx=pos_idx)
        scheduler.step()
        print(f"  Epoch {epoch+1}/{phase2} | val AUC: {auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_wts = copy.deepcopy(model.state_dict())
            torch.save(best_wts, f"weights/{model_key}.pth")

    print(f"\nBest val AUC: {best_auc:.4f} — weights saved to weights/{model_key}.pth")

    # Calibrate threshold
    model.load_state_dict(best_wts)
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for x, y in val_loader:
            p = torch.softmax(model(x.to(device)), dim=1)[:, pos_idx]
            all_probs.extend(p.cpu().numpy())
            all_labels.extend(y.numpy())
    thresh = find_threshold(all_labels, all_probs, pos_idx)
    with open(f"weights/{model_key}_threshold.txt", "w") as f:
        f.write(f"{thresh:.4f}\n")
    print(f"Saved threshold: {thresh:.4f}")


def _run_epoch(model, loader, criterion, optimizer, device, phase, pos_idx=0):
    is_train = phase == "train"
    model.train() if is_train else model.eval()
    all_labels, all_probs = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(is_train):
            out  = model(x)
            loss = criterion(out, y)
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        probs = torch.softmax(out.detach(), dim=1)[:, pos_idx]
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
    if not is_train:
        try:
            return roc_auc_score(
                (np.array(all_labels) == pos_idx).astype(int), all_probs)
        except Exception:
            return 0.5
    return 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="densenet121",
                        choices=list(_BUILDERS.keys()))
    parser.add_argument("--data_dir",   default="./dataset")
    parser.add_argument("--epochs",     type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr",         type=float, default=1e-4)
    args = parser.parse_args()
    train(args.model, args.data_dir, args.epochs, args.batch_size, args.lr)
