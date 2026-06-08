import torch
import torch.nn as nn
from torchvision import models


class BaselineCNN(nn.Module):
    """
    A simple baseline Convolutional Neural Network built from scratch.
    Useful for comparing against complex transfer learning models.

    Improvements for accuracy:
    - BatchNorm after every conv block for training stability.
    - AdaptiveAvgPool makes the network input-size agnostic and reduces
      the size of the classifier (fewer parameters -> less overfitting).
    """
    def __init__(self, num_classes=2):
        super(BaselineCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # AdaptiveAvgPool keeps the Grad-CAM target meaningful and bounds FC size.
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class DeepFractureModel(nn.Module):
    """
    Advanced model leveraging Transfer Learning using ResNet50.

    Improvements for accuracy:
    - Two-phase fine-tuning support via set_finetune_mode().
    - A richer classifier head (BN + Dropout) for better generalisation.
    - inplace ReLU disabled for Grad-CAM gradient compatibility.
    """
    def __init__(self, num_classes=2, freeze_layers=True):
        super(DeepFractureModel, self).__init__()

        # Load pre-trained ResNet50
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Disable inplace ReLU everywhere for Grad-CAM compatibility
        for m in self.resnet.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False

        # Replace final Fully Connected (FC) layer with a stronger head
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_ftrs, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        if freeze_layers:
            self.set_finetune_mode(phase=1)

    def set_finetune_mode(self, phase=1):
        """
        Phase 1 (warmup): freeze the entire backbone, train only the new head.
        Phase 2 (fine-tune): additionally unfreeze layer4 for domain adaptation.
        """
        # Freeze everything first
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Always train the classifier head
        for param in self.resnet.fc.parameters():
            param.requires_grad = True

        if phase >= 2:
            for param in self.resnet.layer4.parameters():
                param.requires_grad = True
        if phase >= 3:
            # Mid-level features help detect thin cortical lines (hairline fractures)
            for param in self.resnet.layer3.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.resnet(x)


class DenseFractureModel(nn.Module):
    """
    DenseNet169 transfer-learning model (the architecture described in the
    project report). Generally gives the best accuracy on MURA.
    """
    def __init__(self, num_classes=2, freeze_layers=True):
        super(DenseFractureModel, self).__init__()
        self.backbone = models.densenet169(weights=models.DenseNet169_Weights.DEFAULT)

        # Disable inplace ReLU for Grad-CAM compatibility
        for m in self.backbone.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = False

        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=False),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

        if freeze_layers:
            self.set_finetune_mode(phase=1)

    def set_finetune_mode(self, phase=1):
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True
        if phase >= 2:
            for name, param in self.backbone.named_parameters():
                if 'denseblock4' in name or 'norm5' in name:
                    param.requires_grad = True
        if phase >= 3:
            for name, param in self.backbone.named_parameters():
                if 'denseblock3' in name or 'denseblock4' in name or 'norm5' in name:
                    param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)


if __name__ == "__main__":
    # Test the models
    resnet = DeepFractureModel()
    baseline = BaselineCNN()
    dense = DenseFractureModel()
    x = torch.randn(2, 3, 224, 224)
    print("ResNet output shape:", resnet(x).shape)
    print("BaselineCNN output shape:", baseline(x).shape)
    print("DenseNet output shape:", dense(x).shape)
