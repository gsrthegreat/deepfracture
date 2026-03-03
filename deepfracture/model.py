import torch
import torch.nn as nn
from torchvision import models

class BaselineCNN(nn.Module):
    """
    A simple baseline Convolutional Neural Network built from scratch.
    Useful for comparing against complex transfer learning models.
    """
    def __init__(self, num_classes=2):
        super(BaselineCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class DeepFractureModel(nn.Module):
    """
    Advanced model leveraging Transfer Learning using ResNet50.
    """
    def __init__(self, num_classes=2, freeze_layers=True):
        super(DeepFractureModel, self).__init__()
        
        # Load pre-trained ResNet50
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        if freeze_layers:
            # Freeze early layers to retain generic feature extractors
            for param in self.resnet.parameters():
                param.requires_grad = False
                
            # Unfreeze the last few layers (layer4) for dataset-specific fine-tuning
            for param in self.resnet.layer4.parameters():
                param.requires_grad = True
                
        # Replace final Fully Connected (FC) layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )
        
    def forward(self, x):
        return self.resnet(x)

if __name__ == "__main__":
    # Test the models
    resnet = DeepFractureModel()
    baseline = BaselineCNN()
    x = torch.randn(1, 3, 224, 224)
    print("ResNet output shape:", resnet(x).shape)
    print("BaselineCNN output shape:", baseline(x).shape)
