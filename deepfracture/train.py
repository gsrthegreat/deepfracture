import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
import copy
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import numpy as np
import cv2

from model import DeepFractureModel, BaselineCNN, DenseFractureModel
from utils import get_transforms


def get_positive_index(classes):
    """
    ImageFolder assigns class indices alphabetically. We must locate the
    'Fractured' class robustly instead of assuming index 1, otherwise the
    ROC/AUC and sensitivity computations are inverted (a serious bug that
    artificially lowers reported accuracy for the positive class).
    """
    for i, c in enumerate(classes):
        if 'frac' in c.lower():
            return i
    # Fallback: assume the second class is positive
    return 1


def compute_class_weights(dataset, sensitivity_boost=1.5):
    """
    Computes inverse frequency class weights to handle class imbalance.
    The positive (Fractured) class receives an extra boost to reduce missed
    hairline / occult fractures during training.
    """
    targets = dataset.targets
    class_counts = np.bincount(targets)
    total_samples = len(targets)

    class_weights = total_samples / (len(class_counts) * class_counts)
    pos_idx = get_positive_index(dataset.classes)
    class_weights[pos_idx] *= sensitivity_boost
    return torch.FloatTensor(class_weights)


def build_weighted_sampler(dataset):
    """
    Oversample fractured images and upweight real/reference radiographs so
    transfer-learning models generalise to real X-rays (not just synthetic).
    """
    targets = np.array(dataset.targets)
    class_counts = np.bincount(targets)
    sample_weights = 1.0 / class_counts[targets]

    for i, path in enumerate(dataset.samples):
        fname = os.path.basename(path[0]).lower()
        if fname.startswith(("real_", "ref")):
            sample_weights[i] *= 5.0
        elif fname.startswith("syn_frac_hair"):
            sample_weights[i] *= 1.5

    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(targets),
        replacement=True,
    )


def find_optimal_threshold(all_labels, all_probs, pos_idx, min_specificity=0.65):
    """
    Pick a threshold via Youden's J (sensitivity + specificity - 1) with a
    specificity floor to limit false positives on normal radiographs.
    """
    bin_labels = (np.array(all_labels) == pos_idx).astype(int)
    fracture_probs = np.array(all_probs)

    best_thresh, best_score, best_sens = 0.55, -1.0, 0.0
    for thresh in np.linspace(0.35, 0.75, 41):
        preds = (fracture_probs >= thresh).astype(int)
        tp = np.sum((preds == 1) & (bin_labels == 1))
        fn = np.sum((preds == 0) & (bin_labels == 1))
        tn = np.sum((preds == 0) & (bin_labels == 0))
        fp = np.sum((preds == 1) & (bin_labels == 0))
        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
        if spec < min_specificity:
            continue
        youden = sens + spec - 1.0
        if youden > best_score:
            best_score, best_thresh, best_sens = youden, float(thresh), sens

    return best_thresh, best_sens


def save_inference_threshold(threshold, model_name):
    path = f'threshold_{model_name}.txt'
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f'{threshold:.4f}\n')
    print(f'Saved sensitivity-tuned threshold ({threshold:.4f}) to {path}')


def save_misclassified(inputs, labels, preds, dataset_classes, model_name):
    """
    Error Analysis: Save misclassified samples to disk for visual inspection.
    """
    error_dir = f'error_analysis_{model_name}'
    os.makedirs(error_dir, exist_ok=True)

    for i in range(len(preds)):
        if preds[i] != labels[i]:
            img = inputs[i].cpu().numpy().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)
            img = (img * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            true_label = dataset_classes[labels[i]]
            pred_label = dataset_classes[preds[i]]

            filename = os.path.join(
                error_dir,
                f'true_{true_label}_pred_{pred_label}_{np.random.randint(10000)}.jpg'
            )
            cv2.imwrite(filename, img)


def collect_validation_scores(model, val_loader, device, pos_idx):
    """Gather fracture probabilities and labels for validation metrics."""
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_probs.extend(probs[:, pos_idx].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_labels), np.array(all_probs)


def run_validation_metrics(model, val_loader, device, pos_idx, threshold=0.5):
    """AUC + sensitivity at the chosen threshold for checkpointing."""
    all_labels, all_probs = collect_validation_scores(model, val_loader, device, pos_idx)
    bin_labels = (all_labels == pos_idx).astype(int)
    try:
        fpr, tpr, _ = roc_curve(bin_labels, all_probs)
        val_auc = auc(fpr, tpr)
    except Exception:
        val_auc = 0.0

    preds = (all_probs >= threshold).astype(int)
    tp = np.sum((preds == 1) & (bin_labels == 1))
    fn = np.sum((preds == 0) & (bin_labels == 1))
    sensitivity = tp / (tp + fn + 1e-8)
    composite = 0.65 * val_auc + 0.35 * sensitivity
    return val_auc, sensitivity, composite


def evaluate_model(model, val_loader, device, classes, model_name, threshold=0.5):
    print(f"\n--- Final Model Evaluation ({model_name}) ---")
    pos_idx = get_positive_index(classes)
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            fracture_probs = probs[:, pos_idx]
            preds = (fracture_probs >= threshold).long()
            preds = torch.where(preds == 1, pos_idx,
                                1 - pos_idx if len(classes) == 2 else preds)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(fracture_probs.cpu().numpy())

            save_misclassified(inputs, labels, preds.cpu().numpy(), classes, model_name)

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))

    cm = confusion_matrix(all_labels, all_preds)
    if cm.shape == (2, 2):
        tp = cm[pos_idx, pos_idx]
        fn = cm[pos_idx, :].sum() - tp
        fp = cm[:, pos_idx].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        print(f"Sensitivity (Recall): {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix ({model_name})')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name}.png')
    plt.close()

    if len(classes) == 2:
        # Binary labels aligned to positive class
        bin_labels = (np.array(all_labels) == pos_idx).astype(int)
        fpr, tpr, _ = roc_curve(bin_labels, all_probs)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic ({model_name})')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f'roc_curve_{model_name}.png')
        plt.close()
        print(f"Final Validation AUC: {roc_auc:.4f}")


def _train_phase(model, train_loader, val_loader, train_dataset, val_dataset,
                 criterion, optimizer, scheduler, device, num_epochs,
                 pos_idx, best_score, best_model_wts, model_type, phase_name,
                 decision_threshold=0.5):
    """Runs one training phase and returns the updated best composite score/weights."""
    for epoch in range(num_epochs):
        print(f'[{phase_name}] Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader, dataset = train_loader, train_dataset
            else:
                model.eval()
                dataloader, dataset = val_loader, val_dataset

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataset)
            epoch_acc = running_corrects.double() / len(dataset)
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val':
                val_auc, sensitivity, composite = run_validation_metrics(
                    model, val_loader, device, pos_idx, decision_threshold)
                print(f'Validation AUC: {val_auc:.4f} | Sensitivity: {sensitivity:.4f}')
                if scheduler is not None:
                    scheduler.step(composite)
                if composite > best_score:
                    best_score = composite
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, f'best_{model_type}.pth')
                    print(f'  -> New best model saved (composite={composite:.4f})')
        print()

    return best_score, best_model_wts


def train_model(data_dir, model_type='resnet', num_epochs=10, batch_size=32,
                learning_rate=0.001):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print(f"Error: Format data in MURA structure under {data_dir}.")
        return

    train_dataset = datasets.ImageFolder(train_dir, transform=get_transforms(is_train=True))
    val_dataset = datasets.ImageFolder(val_dir, transform=get_transforms(is_train=False))

    train_sampler = build_weighted_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    classes = train_dataset.classes
    pos_idx = get_positive_index(classes)
    print(f"Classes: {classes} | Positive (Fractured) index: {pos_idx}")

    # Class imbalance handling
    class_weights = compute_class_weights(train_dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.08)
    print(f"Computed Class Weights: {class_weights.cpu().numpy()}")

    best_score = 0.0
    decision_threshold = 0.5

    if model_type == 'baseline':
        model = BaselineCNN(num_classes=len(classes)).to(device)
        best_model_wts = copy.deepcopy(model.state_dict())
        # weight_decay regularises; ReduceLROnPlateau adapts the LR
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                         factor=0.5, patience=2)
        best_score, best_model_wts = _train_phase(
            model, train_loader, val_loader, train_dataset, val_dataset,
            criterion, optimizer, scheduler, device, num_epochs, pos_idx,
            best_score, best_model_wts, model_type, 'Train', decision_threshold)
    else:
        # Transfer-learning models use two-phase fine-tuning
        if model_type == 'densenet':
            model = DenseFractureModel(num_classes=len(classes), freeze_layers=True).to(device)
        else:
            model = DeepFractureModel(num_classes=len(classes), freeze_layers=True).to(device)
        best_model_wts = copy.deepcopy(model.state_dict())

        phase1_epochs = max(3, num_epochs // 4)
        phase2_epochs = max(3, num_epochs // 3)
        phase3_epochs = max(2, num_epochs - phase1_epochs - phase2_epochs)

        # ---- Phase 1: warm-up head only ----
        model.set_finetune_mode(phase=1)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                        factor=0.5, patience=1)
        best_score, best_model_wts = _train_phase(
            model, train_loader, val_loader, train_dataset, val_dataset,
            criterion, optimizer, scheduler, device, phase1_epochs, pos_idx,
            best_score, best_model_wts, model_type, 'Phase1-Warmup', decision_threshold)

        # ---- Phase 2: fine-tune deepest blocks ----
        model.set_finetune_mode(phase=2)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=1e-4, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                        factor=0.5, patience=2)
        best_score, best_model_wts = _train_phase(
            model, train_loader, val_loader, train_dataset, val_dataset,
            criterion, optimizer, scheduler, device, phase2_epochs, pos_idx,
            best_score, best_model_wts, model_type, 'Phase2-Finetune', decision_threshold)

        # ---- Phase 3: mid-level features for hairline fractures ----
        model.set_finetune_mode(phase=3)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=5e-5, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                        factor=0.5, patience=2)
        best_score, best_model_wts = _train_phase(
            model, train_loader, val_loader, train_dataset, val_dataset,
            criterion, optimizer, scheduler, device, phase3_epochs, pos_idx,
            best_score, best_model_wts, model_type, 'Phase3-Hairline', decision_threshold)

    print(f'Training complete. Best validation composite score: {best_score:.4f}')
    model.load_state_dict(best_model_wts)

    all_labels, all_probs = collect_validation_scores(model, val_loader, device, pos_idx)
    decision_threshold, tuned_sens = find_optimal_threshold(all_labels, all_probs, pos_idx)
    save_inference_threshold(decision_threshold, model_type)
    print(f'Sensitivity-tuned threshold: {decision_threshold:.4f} (sensitivity={tuned_sens:.4f})')

    evaluate_model(model, val_loader, device, classes, model_type, decision_threshold)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet',
                        choices=['baseline', 'resnet', 'densenet'])
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    train_model(data_dir=args.data_dir, model_type=args.model,
                num_epochs=args.epochs, batch_size=args.batch_size)
