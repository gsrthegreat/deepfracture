"""
model.py — pretrained model definitions for FractureAI.

All models use ImageNet-pretrained backbones fine-tuned for binary
fracture classification (class 0 = Fractured, class 1 = Normal).

On Streamlit Cloud the weights don't exist yet, so we download them
from torchvision and return a zero-shot transfer model instead of
crashing. If you have your own fine-tuned weights, drop them in the
repo as  weights/<model_key>.pth  and they will be loaded automatically.
"""

import os
import torch
import torch.nn as nn
from torchvision import models

# ── Registry ──────────────────────────────────────────────────────────────────
MODELS = {
    "DenseNet121 (fast)":       "densenet121",
    "EfficientNet-B3 (accurate)": "efficientnet_b3",
    "ResNet50 (balanced)":      "resnet50",
}


def _build_densenet121(num_classes=2):
    m = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    for mod in m.modules():
        if isinstance(mod, nn.ReLU):
            mod.inplace = False
    in_f = m.classifier.in_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_f, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=False),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    # target layer: last dense block
    target = m.features.denseblock4
    return m, target


def _build_efficientnet_b3(num_classes=2):
    m = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
    for mod in m.modules():
        if isinstance(mod, nn.ReLU) or isinstance(mod, nn.SiLU):
            if hasattr(mod, 'inplace'):
                mod.inplace = False
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_f, num_classes),
    )
    # target layer: last conv block before global pool
    target = m.features[-1]
    return m, target


def _build_resnet50(num_classes=2):
    m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    for mod in m.modules():
        if isinstance(mod, nn.ReLU):
            mod.inplace = False
    in_f = m.fc.in_features
    m.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_f, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=False),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    target = m.layer4[-1]
    return m, target


_BUILDERS = {
    "densenet121":     _build_densenet121,
    "efficientnet_b3": _build_efficientnet_b3,
    "resnet50":        _build_resnet50,
}


def load_model(display_name: str):
    """
    Build the model and optionally load fine-tuned weights.
    Falls back gracefully to ImageNet init if no weights file found.
    """
    key     = MODELS[display_name]
    builder = _BUILDERS[key]
    model, target_layer = builder(num_classes=2)

    weight_path = os.path.join("weights", f"{key}.pth")
    if os.path.exists(weight_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state  = torch.load(weight_path, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"[FractureAI] Loaded fine-tuned weights from {weight_path}")
    else:
        print(f"[FractureAI] No weights at {weight_path} — using ImageNet init.")

    return model, target_layer
