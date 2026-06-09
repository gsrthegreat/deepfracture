"""
preprocess.py — image preprocessing for FractureAI.

Pipeline (mirrors ImageNet validation preprocessing + medical enhancements):
  1. Resize to 256×256
  2. CLAHE in LAB space (enhances low-contrast fracture lines)
  3. Unsharp masking (sharpens cortical edges)
  4. CenterCrop to 224×224
  5. ToTensor + ImageNet normalisation

Returns both the normalised tensor (for the model) and the
preprocessed uint8 RGB array (for Grad-CAM overlay).
"""

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms


def _clahe(img_np: np.ndarray) -> np.ndarray:
    """CLAHE on the L channel of LAB — preserves colour, boosts local contrast."""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _unsharp(img_np: np.ndarray, sigma: float = 1.0, amount: float = 0.8) -> np.ndarray:
    """Unsharp mask to accentuate thin fracture lines."""
    blurred   = cv2.GaussianBlur(img_np, (0, 0), sigma)
    sharpened = cv2.addWeighted(img_np, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


def preprocess(pil_image: Image.Image):
    """
    Args:
        pil_image: PIL RGB image (any size).

    Returns:
        tensor      : torch.FloatTensor of shape (3, 224, 224), normalised.
        model_view  : np.ndarray uint8 (224, 224, 3) — the image the model sees,
                      used for Grad-CAM overlay at model resolution.
    """
    # 1. Resize
    img = pil_image.convert("RGB").resize((256, 256), Image.BILINEAR)
    img_np = np.array(img)

    # 2. CLAHE
    img_np = _clahe(img_np)

    # 3. Unsharp mask
    img_np = _unsharp(img_np)

    # 4. CenterCrop to 224
    h, w   = img_np.shape[:2]
    top    = (h - 224) // 2
    left   = (w - 224) // 2
    img_np = img_np[top:top + 224, left:left + 224]

    model_view = img_np.copy()  # uint8 RGB for overlay

    # 5. ToTensor + normalise
    tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
    tensor = _NORMALIZE(tensor)

    return tensor, model_view
