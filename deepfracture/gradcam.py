import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt


def get_gradcam_target_layer(model, model_key):
    """Return a spatially detailed layer for sharper Grad-CAM localisation."""
    if model_key == 'densenet':
        return model.backbone.features.denseblock3
    if model_key == 'resnet':
        return model.resnet.layer3[-1].conv3
    return model.features[8]


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None

    def _forward_hook(self, module, input, output):
        self.feature_maps = output
        if isinstance(output, torch.Tensor) and torch.is_grad_enabled():
            output.retain_grad()

    def generate_heatmap(self, input_image, target_class=None):
        self.model.eval()
        self.feature_maps = None

        handles = [self.target_layer.register_forward_hook(self._forward_hook)]

        try:
            input_image = input_image.detach().requires_grad_(True)

            with torch.enable_grad():
                output = self.model(input_image)

                if target_class is None:
                    target_class = int(output.argmax(dim=1).item())

                self.model.zero_grad(set_to_none=True)
                score = output[0, target_class]
                score.backward(retain_graph=False)

            gradients = self.feature_maps.grad if self.feature_maps is not None else None

            if gradients is None or self.feature_maps is None:
                raise RuntimeError(
                    "Grad-CAM could not capture gradients. "
                    "Check that the target layer produces spatial feature maps."
                )

            feature_maps = self.feature_maps.detach()
            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * feature_maps, dim=1).squeeze(0)

            cam = F.relu(cam)
            cam = cam.cpu().numpy()
            cam = self._refine_cam(cam)
            return cam, target_class
        finally:
            for handle in handles:
                handle.remove()

    @staticmethod
    def _refine_cam(cam):
        """Upsample, smooth, and suppress background activations."""
        cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_CUBIC)
        cam = cv2.GaussianBlur(cam, (0, 0), sigmaX=3)
        floor = np.percentile(cam, 72)
        cam = np.clip(cam - floor, 0, None)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam


def overlay_heatmap_on_rgb(rgb_img, heatmap, save_path=None, alpha=0.45):
    """Overlay heatmap on an RGB uint8 image that matches the model input size."""
    if isinstance(rgb_img, np.ndarray):
        base = rgb_img.copy()
    else:
        base = np.array(rgb_img.convert('RGB'))

    heatmap_resized = cv2.resize(heatmap, (base.shape[1], base.shape[0]),
                                  interpolation=cv2.INTER_CUBIC)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colormap = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)

    overlayed = cv2.addWeighted(base, alpha, colormap, 1 - alpha, 0)

    if save_path:
        plt.imsave(save_path, overlayed)

    return overlayed


def map_cam_to_original(cam_224, orig_w, orig_h):
    """Map a 224x224 CAM back through the validation resize/center-crop pipeline."""
    cam_256 = np.zeros((256, 256), dtype=np.float32)
    offset = (256 - 224) // 2
    cam_256[offset:offset + 224, offset:offset + 224] = cam_224
    return cv2.resize(cam_256, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)


def overlay_heatmap(original_img_path, heatmap, save_path=None):
    """Backward-compatible wrapper; prefers aligned overlay via overlay_heatmap_on_rgb."""
    raw_img = cv2.imread(original_img_path)
    if raw_img is None:
        raise ValueError(f"Could not read image at {original_img_path}")
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    cam_orig = map_cam_to_original(heatmap, raw_img.shape[1], raw_img.shape[0])
    return overlay_heatmap_on_rgb(raw_img, cam_orig, save_path=save_path)
