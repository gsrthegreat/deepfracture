"""
Build / refresh the DeepFracture training dataset.
Run: python build_dataset.py
"""
import os
import json
import glob
import shutil
import urllib.request
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
from skimage import data
import cv2

DATA_DIR = "./data"
REFERENCE_DIR = "./data/reference"

WIKIMEDIA_FRACTURED = [
    "Collesfracture.jpg",
    "Tibia_fibula_fracture_AP.jpg",
    "Endo_Fracture.jpg",
    "Fractured_ribs.jpg",
    "Lateral_radiograph_displaying_the_displaced_tibial_spine_(arrow).jpg",
]

WIKIMEDIA_NORMAL = [
    "Knee_X-ray.jpg",
    "Normal_X-ray_of_the_elbow.jpg",
    "Normal_chest_X-ray.jpg",
    "Normal_hand_X-ray.jpg",
    "Normal_foot_X-ray.jpg",
]

TRAIN_COUNTS = {"Normal": 120, "Fractured": 120}
VAL_COUNTS = {"Normal": 30, "Fractured": 30}
AUGMENT_PER_REAL = 20
USER_AGENT = "DeepFracture/1.0 (university research; contact local maintainer)"


def setup_directories():
    for split in ["train", "val"]:
        for cls in ["Fractured", "Normal"]:
            os.makedirs(os.path.join(DATA_DIR, split, cls), exist_ok=True)
    os.makedirs(os.path.join(REFERENCE_DIR, "Fractured"), exist_ok=True)
    os.makedirs(os.path.join(REFERENCE_DIR, "Normal"), exist_ok=True)


def wikimedia_download(filename, dest):
    if os.path.exists(dest):
        return True
    api = (
        "https://commons.wikimedia.org/w/api.php"
        f"?action=query&titles=File:{filename}&prop=imageinfo&iiprop=url&format=json"
    )
    try:
        req = urllib.request.Request(api, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.load(resp)
        pages = payload["query"]["pages"]
        page = next(iter(pages.values()))
        url = page["imageinfo"][0]["url"]
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data_bytes = resp.read()
        with open(dest, "wb") as f:
            f.write(data_bytes)
        with Image.open(dest) as img:
            img.convert("RGB").save(dest)
        print(f"  Downloaded {filename}")
        return True
    except Exception as exc:
        print(f"  Skipped {filename}: {exc}")
        return False


def download_real_samples():
    print("Downloading real radiographs from Wikimedia Commons...")
    for split, frac_files, norm_files in [
        ("train", WIKIMEDIA_FRACTURED, WIKIMEDIA_NORMAL),
        ("val", WIKIMEDIA_FRACTURED[:3], WIKIMEDIA_NORMAL[:3]),
    ]:
        for i, fname in enumerate(frac_files):
            dest = os.path.join(DATA_DIR, split, "Fractured", f"real_frac_{i}.jpg")
            wikimedia_download(fname, dest)
        for i, fname in enumerate(norm_files):
            dest = os.path.join(DATA_DIR, split, "Normal", f"real_norm_{i}.jpg")
            wikimedia_download(fname, dest)


def import_reference_radiographs():
    """Copy bundled / local reference X-rays into the reference folder."""
    print("Importing local reference radiographs...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_globs = [
        os.path.join(script_dir, "data", "reference", "*", "*"),
        os.path.join(script_dir, "..", "..", ".cursor", "projects", "*", "assets", "*.png"),
        os.path.join(script_dir, "..", ".cursor", "projects", "*", "assets", "*.png"),
    ]
    candidates = []
    for pattern in search_globs:
        candidates.extend(glob.glob(pattern))

    for path in sorted(set(candidates)):
        if not path.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        name = os.path.basename(path).lower()
        cls = "Normal" if ("elbow" in name or "normal" in name or "knee" in name) else "Fractured"
        dest = os.path.join(REFERENCE_DIR, cls, os.path.basename(path))
        if not os.path.exists(dest):
            shutil.copy2(path, dest)
            print(f"  Reference: {cls} <- {path}")

    for cls in ["Fractured", "Normal"]:
        ref_dir = os.path.join(REFERENCE_DIR, cls)
        for split in ["train", "val"]:
            files = [
                f for f in os.listdir(ref_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            target_dir = os.path.join(DATA_DIR, split, cls)
            for i, fname in enumerate(files):
                prefix = "ref" if split == "train" else "ref_val"
                dest = os.path.join(target_dir, f"{prefix}_{i}_{fname}")
                if not os.path.exists(dest):
                    shutil.copy2(os.path.join(ref_dir, fname), dest)
                    print(f"  Added {dest}")


def _bone_base(seed):
    rng = np.random.default_rng(seed)
    h, w = 224, 224
    base = np.zeros((h, w), dtype=np.float32)
    for _ in range(3):
        cx, cy = rng.integers(0, w), rng.integers(0, h)
        y, x = np.ogrid[:h, :w]
        blob = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * (rng.integers(30, 90) ** 2)))
        base += blob * rng.uniform(40, 120)
    base = cv2.GaussianBlur(base, (0, 0), sigmaX=6)
    base += rng.normal(0, 12, (h, w))
    return np.clip(base, 0, 255).astype(np.uint8)


def _draw_fracture_line(img, hairline=False):
    h, w = img.shape[:2]
    x1, y1 = np.random.randint(w // 5, 4 * w // 5), np.random.randint(h // 5, 4 * h // 5)
    angle = np.random.uniform(-np.pi / 3, np.pi / 3)
    length = np.random.randint(25, 70)
    x2 = int(x1 + length * np.cos(angle))
    y2 = int(y1 + length * np.sin(angle))
    if hairline:
        intensity = int(np.random.uniform(165, 215))
        cv2.line(img, (x1, y1), (x2, y2), (intensity, intensity, intensity), 1)
        cv2.GaussianBlur(img, (3, 3), 0, dst=img)
    else:
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
    return img


def generate_synthetic_dataset():
    print("Generating synthetic hairline-focused samples...")
    moon = cv2.resize(data.moon(), (224, 224))
    for split, counts in [("train", TRAIN_COUNTS), ("val", VAL_COUNTS)]:
        for cls, total in counts.items():
            for i in range(total):
                seed = hash((split, cls, i)) % (2 ** 32)
                base = moon.copy() if i % 3 == 0 else _bone_base(seed)
                noisy = np.clip(base.astype(np.float32) + np.random.normal(0, 14, base.shape), 0, 255)
                noisy = noisy.astype(np.uint8)
                if cls == "Normal":
                    out, fname = noisy, f"syn_norm_{i}.jpg"
                else:
                    hairline = i < int(total * 0.75)
                    out = _draw_fracture_line(noisy.copy(), hairline=hairline)
                    fname = f"syn_frac_{'hair' if hairline else 'obv'}_{i}.jpg"
                Image.fromarray(out).convert("RGB").save(
                    os.path.join(DATA_DIR, split, cls, fname))


def _augment_pil(img):
    aug = img.copy()
    if np.random.rand() > 0.5:
        aug = ImageOps.mirror(aug)
    aug = aug.rotate(np.random.uniform(-12, 12), resample=Image.BILINEAR)
    aug = ImageEnhance.Brightness(aug).enhance(np.random.uniform(0.85, 1.15))
    aug = ImageEnhance.Contrast(aug).enhance(np.random.uniform(0.8, 1.25))
    return aug


def augment_real_images():
    print(f"Augmenting real/reference images ({AUGMENT_PER_REAL} each)...")
    for split in ["train", "val"]:
        for cls in ["Fractured", "Normal"]:
            folder = os.path.join(DATA_DIR, split, cls)
            sources = [
                f for f in os.listdir(folder)
                if (f.startswith("real_") or f.startswith("ref"))
                and "_aug" not in f
                and f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            for fname in sources:
                path = os.path.join(folder, fname)
                try:
                    img = Image.open(path).convert("RGB")
                except Exception:
                    continue
                stem = os.path.splitext(fname)[0]
                for j in range(AUGMENT_PER_REAL):
                    out = os.path.join(folder, f"{stem}_aug{j}.jpg")
                    if not os.path.exists(out):
                        _augment_pil(img).save(out)


def print_dataset_stats():
    for split in ["train", "val"]:
        print(f"\n{split.upper()}:")
        for cls in ["Fractured", "Normal"]:
            folder = os.path.join(DATA_DIR, split, cls)
            files = os.listdir(folder)
            real = sum(1 for f in files if f.startswith(("real_", "ref")))
            syn = sum(1 for f in files if f.startswith("syn_"))
            print(f"  {cls}: {len(files)} total ({real} real/ref, {syn} synthetic)")


def build_dataset(clean_synthetic=False):
    setup_directories()
    if clean_synthetic:
        for split in ["train", "val"]:
            for cls in ["Fractured", "Normal"]:
                folder = os.path.join(DATA_DIR, split, cls)
                for fname in os.listdir(folder):
                    if fname.startswith(("syn_", "frac_", "norm_")):
                        os.remove(os.path.join(folder, fname))
    download_real_samples()
    import_reference_radiographs()
    generate_synthetic_dataset()
    augment_real_images()
    print_dataset_stats()


if __name__ == "__main__":
    build_dataset(clean_synthetic=True)
