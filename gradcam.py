"""
gradcam.py — reliable Grad-CAM implementation for FractureAI.

Key design decisions:
- Uses register_full_backward_hook (not .grad) for reliable gradient capture
  on non-leaf tensors (DenseNet, EfficientNet outputs are non-leaf).
- Always runs with torch.enable_grad() regardless of outer context.
- Handles DenseNet's tuple outputs (denseblock4 returns a tensor directly).
- Percentile-based background suppression tuned for medical imaging.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2


class GradCAM:
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self._fmaps       = None
        self._grads       = None

    def _fwd_hook(self, module, inp, out):
        # DenseNet denseblock output is a plain tensor
        self._fmaps = out if isinstance(out, torch.Tensor) else out[0]

    def _bwd_hook(self, module, grad_in, grad_out):
        self._grads = grad_out[0]

    def generate(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        """
        Returns a float32 numpy array of shape (224, 224) in range [0, 1].
        Red = high activation (model focused here).
        """
        self.model.eval()
        self._fmaps = None
        self._grads = None

        h1 = self.target_layer.register_forward_hook(self._fwd_hook)
        h2 = self.target_layer.register_full_backward_hook(self._bwd_hook)

        try:
            inp = input_tensor.detach().requires_grad_(True)

            with torch.enable_grad():
                logits = self.model(inp)
                self.model.zero_grad(set_to_none=True)
                logits[0, target_class].backward()

            if self._fmaps is None or self._grads is None:
                raise RuntimeError("Grad-CAM hooks did not fire — check target layer.")

            fmaps = self._fmaps.detach()   # (1, C, H, W)
            grads = self._grads.detach()   # (1, C, H, W)

            # Global-average-pool the gradients → importance weights per channel
            weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
            cam = (weights * fmaps).sum(dim=1).squeeze(0)   # (H, W)
            cam = F.relu(cam).cpu().numpy()

            return _postprocess_cam(cam)

        finally:
            h1.remove()
            h2.remove()


def _postprocess_cam(cam: np.ndarray) -> np.ndarray:
    """
    Resize to 224×224, smooth, suppress background, normalise to [0,1].
    Tuned for medical X-rays where fracture activations are often subtle.
    """
    cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_CUBIC)

    # Light smoothing — preserves thin fracture lines
    cam = cv2.GaussianBlur(cam, (0, 0), sigmaX=1.5)

    # Suppress bottom 55th percentile (background) — lower than typical 70-75
    # because fracture activations in X-rays are often low-magnitude
    floor = np.percentile(cam, 55)
    cam   = np.clip(cam - floor, 0, None)

    # Normalise
    cam -= cam.min()
    denom = cam.max()
    if denom > 1e-8:
        cam /= denom

    return cam.astype(np.float32)


def render_heatmap(rgb_img: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Blend a Jet colourmap heatmap onto an RGB uint8 image.
    alpha controls original image visibility (higher = more visible original).
    """
    h, w = rgb_img.shape[:2]
    cam_resized  = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
    cam_uint8    = np.uint8(255 * cam_resized)
    colormap     = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    colormap_rgb = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)

    base = rgb_img.copy().astype(np.float32)
    heat = colormap_rgb.astype(np.float32)
    blended = (alpha * base + (1 - alpha) * heat).clip(0, 255).astype(np.uint8)
    return blended
