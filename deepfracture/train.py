import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
import os
import copy
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import numpy as np
import cv2

from model import DeepFractureModel, BaselineCNN
from utils import get_transforms

def compute_class_weights(dataset):
    """
    Computes inverse frequency class weights to handle class imbalance.
    """
    targets = dataset.targets
    class_counts = np.bincount(targets)
    total_samples = len(targets)
    
    # Weight = Total Samples / (Number of Classes * Class Count)
    class_weights = total_samples / (len(class_counts) * class_counts)
    return torch.FloatTensor(class_weights)

def save_misclassified(inputs, labels, preds, dataset_classes, model_name):
    """
    Error Analysis: Save misclassified samples to disk for visual inspection.
    """
    error_dir = f'error_analysis_{model_name}'
    os.makedirs(error_dir, exist_ok=True)
    
    for i in range(len(preds)):
        if preds[i] != labels[i]:
            # Unnormalize image for saving
            img = inputs[i].cpu().numpy().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)
            img = (img * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            true_label = dataset_classes[labels[i]]
            pred_label = dataset_classes[preds[i]]
            
            filename = os.path.join(error_dir, f'true_{true_label}_pred_{pred_label}_{np.random.randint(10000)}.jpg')
            cv2.imwrite(filename, img)

def evaluate_model(model, val_loader, device, classes, model_name):
    print(f"\n--- Final Model Evaluation ({model_name}) ---")
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            # Assuming binary: class 1 is positive class
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy()) 
            
            # Error Analysis
            save_misclassified(inputs, labels, preds.cpu().numpy(), classes, model_name)
            
    # Classification Report (Precision, Recall, F1-Score, Accuracy)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))
    
    # Confusion Matrix, Sensitivity, Specificity
    cm = confusion_matrix(all_labels, all_preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        
        print(f"Sensitivity (True Positive Rate/Recall): {sensitivity:.4f}")
        print(f"Specificity (True Negative Rate): {specificity:.4f}")
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix ({model_name})')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name}.png')
    plt.close()
    
    # ROC Curve and AUC
    if len(classes) == 2:
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
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

def train_model(data_dir, model_type='resnet', num_epochs=10, batch_size=32, learning_rate=0.001):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print(f"Error: Format data in MURA structure under {data_dir}.")
        return
        
    train_dataset = datasets.ImageFolder(train_dir, transform=get_transforms(is_train=True))
    val_dataset = datasets.ImageFolder(val_dir, transform=get_transforms(is_train=False))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Initialize Model
    if model_type == 'baseline':
        model = BaselineCNN(num_classes=len(train_dataset.classes))
    else:
        model = DeepFractureModel(num_classes=len(train_dataset.classes), freeze_layers=True)
        
    model = model.to(device)
    
    # Handle Class Imbalance
    class_weights = compute_class_weights(train_dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"Computed Class Weights: {class_weights.cpu().numpy()}")
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
                dataset = train_dataset
            else:
                model.eval()
                dataloader = val_loader
                dataset = val_dataset
                
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
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, f'best_{model_type}.pth')
                
        print()
        
    print(f'Training complete. Best validation accuracy: {best_acc:.4f}')
    
    model.load_state_dict(best_model_wts)
    evaluate_model(model, val_loader, device, train_dataset.classes, model_type)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet', choices=['baseline', 'resnet'])
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    
    train_model(data_dir=args.data_dir, model_type=args.model, num_epochs=args.epochs)
