import torch
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
from skimage.feature import graycomatrix, graycoprops

class CLAHETransform:
    """
    Custom PyTorch Transform applying Contrast Limited Adaptive Histogram Equalization (CLAHE).
    Enhances local contrast, crucial for revealing subtle bone fractures in radiographs.
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        # Instantiate temporarily to avoid multiprocessing pickling errors
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        
        # Convert PIL to numpy
        img_np = np.array(img)
        # Apply CLAHE to L channel in LAB color space if RGB, or directly if grayscale
        if len(img_np.shape) == 3:
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            img_np = clahe.apply(img_np)
        return Image.fromarray(img_np)

def get_transforms(is_train=True):
    """
    Returns image transformations for training and validation.
    Includes advanced augmentations for robustness.
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            CLAHETransform(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            CLAHETransform(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

def extract_image_features(image_path):
    """
    Extract advanced features for Risk Stratification:
    - Intensity distribution (mean, std)
    - Edge density (Canny)
    - Texture features (GLCM: contrast, homogeneity)
    """
    # Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        pil_img = Image.open(image_path).convert('L')
        img = np.array(pil_img)
        
    # 1. Intensity features
    mean_intensity = np.mean(img)
    std_dev = np.std(img)
    
    # 2. Edge density
    edges = cv2.Canny(img, 100, 200)
    edge_density = np.sum(edges > 0) / (img.shape[0] * img.shape[1])
    
    # 3. GLCM Texture features
    # Resize to speed up GLCM computation
    img_small = cv2.resize(img, (256, 256))
    glcm = graycomatrix(img_small, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    
    return mean_intensity, std_dev, edge_density, contrast, homogeneity
