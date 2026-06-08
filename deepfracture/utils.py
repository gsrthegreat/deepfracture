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
    def __init__(self, clip_limit=3.0, tile_grid_size=(4, 4)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        img_np = np.array(img)
        if len(img_np.shape) == 3:
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            img_np = clahe.apply(img_np)
        return Image.fromarray(img_np)


class UnsharpMaskTransform:
    """
    Classical unsharp masking to accentuate thin fracture lines after CLAHE.
    """
    def __init__(self, radius=1.0, amount=0.8):
        self.radius = radius
        self.amount = amount

    def __call__(self, img):
        img_np = np.array(img)
        blurred = cv2.GaussianBlur(img_np, (0, 0), self.radius)
        sharpened = cv2.addWeighted(img_np, 1.0 + self.amount, blurred, -self.amount, 0)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        return Image.fromarray(sharpened)


def prepare_model_input(pil_image):
    """
    Build the exact tensor the classifier sees plus a 224x224 RGB view for Grad-CAM overlay.
    Pipeline must exactly mirror get_transforms(is_train=False).
    """
    resize = transforms.Resize((256, 256))
    clahe = CLAHETransform()
    unsharp = UnsharpMaskTransform()
    crop = transforms.CenterCrop(224)
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    img = resize(pil_image.convert('RGB'))
    img = clahe(img)
    img = unsharp(img)
    model_view = crop(img)
    tensor = normalize(to_tensor(model_view))
    return tensor, np.array(model_view)


def compute_localized_edge_score(image_path):
    """
    Fractures produce a focal high-edge line; joint surfaces spread edges diffusely.
    Returns (max_cell_density, mean_density, localization_ratio).
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.array(Image.open(image_path).convert('L'))
    img = cv2.resize(img, (256, 256))
    edges = cv2.Canny(img, 50, 150)

    h, w = edges.shape
    cell_densities = []
    for i in range(4):
        for j in range(4):
            cell = edges[i * h // 4:(i + 1) * h // 4, j * w // 4:(j + 1) * w // 4]
            cell_densities.append(np.mean(cell > 0))

    max_cell = float(max(cell_densities))
    mean_cell = float(np.mean(cell_densities))
    ratio = max_cell / (mean_cell + 1e-8)
    return max_cell, mean_cell, ratio


def refine_fracture_prediction(fracture_prob, threshold, localization_ratio):
    """
    Combine CNN probability with classical edge localisation.

    Changes from original:
    - Removed hardcoded max(..., 0.55) clamp so saved thresholds are respected
    - Loosened localization conditions to reduce false negatives on hairline fractures
    """
    effective_threshold = threshold  # trust the calibrated threshold from train.py

    if fracture_prob < effective_threshold:
        return False

    # High confidence: always fracture
    if fracture_prob >= 0.70:
        return True

    # Focal cortical disruption supports fracture even when CNN score is borderline
    if localization_ratio >= 5.0:
        return True

    # Diffuse edges (normal joints) without strong CNN confidence -> normal
    if localization_ratio < 3.5 and fracture_prob < 0.65:
        return False

    return fracture_prob >= effective_threshold


def get_transforms(is_train=True):
    """
    Returns image transformations for training and validation.
    """
    common_preprocess = [
        transforms.Resize((256, 256)),
        CLAHETransform(),
        UnsharpMaskTransform(),
        transforms.CenterCrop(224),
    ]

    if is_train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            CLAHETransform(),
            UnsharpMaskTransform(),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(12),
            transforms.RandomAffine(degrees=0, translate=(0.04, 0.04), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.15, contrast=0.35, saturation=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose(common_preprocess + [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


def extract_image_features(image_path):
    """
    Extract advanced features for Risk Stratification:
    - Intensity distribution (mean, std)
    - Edge density (Canny)
    - Texture features (GLCM: contrast, homogeneity)
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        pil_img = Image.open(image_path).convert('L')
        img = np.array(pil_img)

    mean_intensity = np.mean(img)
    std_dev = np.std(img)

    edges = cv2.Canny(img, 100, 200)
    edge_density = np.sum(edges > 0) / (img.shape[0] * img.shape[1])

    img_small = cv2.resize(img, (256, 256))
    glcm = graycomatrix(img_small, distances=[1], angles=[0], levels=256,
                        symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

    return mean_intensity, std_dev, edge_density, contrast, homogeneity