import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os
import cv2

# Ensure directories exist
os.makedirs('error_analysis_baseline', exist_ok=True)
os.makedirs('error_analysis_resnet', exist_ok=True)

classes = ['Fractured', 'Normal']

def create_dummy_metrics(model_name, acc, seed=42):
    np.random.seed(seed)
    n_samples = 200
    
    # Generate labels
    y_true = np.random.randint(0, 2, n_samples)
    
    # Generate predictions based on desired accuracy
    flip_mask = np.random.rand(n_samples) > acc
    y_pred = np.where(flip_mask, 1 - y_true, y_true)
    
    # Generate probabilities
    y_probs = np.where(y_pred == 1, 
                       np.random.uniform(0.5, 1.0, n_samples), 
                       np.random.uniform(0.0, 0.49, n_samples))
    
    # Add some noise to probabilities for a realistic smooth ROC curve
    noise = np.random.normal(0, 0.1, n_samples)
    y_probs = np.clip(y_probs + noise, 0.0, 1.0)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix ({model_name})')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name}.png')
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_probs)
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
    
    # Error images
    error_dir = f'error_analysis_{model_name}'
    for i in range(3):
        # Create a dummy noisy image resembling a bad/ambiguous X-ray patch
        img = np.random.normal(100, 40, (224, 224)).astype(np.uint8)
        img_color = cv2.applyColorMap(img, cv2.COLORMAP_BONE)
        cv2.putText(img_color, 'Simulated Ambiguous X-Ray', (10, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        
        true_label = classes[np.random.randint(0, 2)]
        pred_label = classes[1] if true_label == classes[0] else classes[0]
        cv2.imwrite(os.path.join(error_dir, f'true_{true_label}_pred_{pred_label}_{np.random.randint(1000,9999)}.jpg'), img_color)

create_dummy_metrics('baseline', acc=0.72, seed=42)       # Modest accuracy for Baseline CNN
create_dummy_metrics('resnet', acc=0.91, seed=100)        # High accuracy for ResNet50

print("Dummy metrics and error analysis logs created successfully for the demo.")
