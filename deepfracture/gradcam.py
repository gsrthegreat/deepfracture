import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        """
        Initialize Grad-CAM with PyTorch model and target layer.
        """
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_feature_maps)
        # Use full backward hook for PyTorch >= 2.0 compatibility
        self.target_layer.register_full_backward_hook(self.save_gradients)
        
    def save_feature_maps(self, module, input, output):
        self.feature_maps = output.detach()
        
    def save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate_heatmap(self, input_image, target_class=None):
        """
        Generate Grad-CAM heatmap for the given input image.
        """
        # Ensure model is in eval mode
        self.model.eval()
        
        # Forward pass
        output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        # Backward pass
        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()
        
        # Compute the guided gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.feature_maps, dim=1).squeeze(0)
        
        # ReLU passed to CAM
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        
        # Normalize
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        
        return cam, target_class

def overlay_heatmap(original_img_path, heatmap, save_path=None):
    """
    Overlay the heatmap on the original image.
    """
    # Read original image
    raw_img = cv2.imread(original_img_path)
    if raw_img is None:
        raise ValueError(f"Could not read image at {original_img_path}")
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    
    # Resize heatmap to match original image size
    heatmap_resized = cv2.resize(heatmap, (raw_img.shape[1], raw_img.shape[0]))
    
    # Convert heatmap to uint8
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    
    # Apply colormap
    colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colormap = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
    
    # Overlay heatmap on image
    alpha = 0.5
    overlayed_img = cv2.addWeighted(raw_img, alpha, colormap, 1 - alpha, 0)
    
    if save_path:
        # Convert back to BGR to save via cv2, or use plt.imsave which expects RGB
        plt.imsave(save_path, overlayed_img)
        
    return overlayed_img
