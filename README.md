# DeepFractureAI 🦴

Deep learning X-ray fracture detection with Grad-CAM heatmap visualisation.

## Features
- **3 pretrained models**: DenseNet121, EfficientNet-B3, ResNet50 (ImageNet backbone)
- **Grad-CAM heatmaps**: highlights the exact region driving the prediction
- **CLAHE + Unsharp masking**: preprocessing tuned for hairline fracture detection
- **Adjustable threshold**: tune sensitivity vs specificity in the sidebar
- **Edge analysis**: Canny edge map for fracture line localisation
- **GLCM biomarkers**: texture features for supplementary analysis

## Project Structure

```
fracture_app/
├── app.py          # Streamlit UI
├── model.py        # Model definitions + loader
├── gradcam.py      # Grad-CAM implementation
├── preprocess.py   # CLAHE + unsharp preprocessing pipeline
├── requirements.txt
├── weights/        # (optional) drop fine-tuned .pth files here
│   ├── densenet121.pth
│   ├── efficientnet_b3.pth
│   └── resnet50.pth
└── README.md
```

## Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploying to Streamlit Cloud

1. Push this folder to a GitHub repo
2. Go to share.streamlit.io → New app
3. Set **Main file path** to `app.py`
4. Deploy

The app works out-of-the-box with ImageNet pretrained weights.
To use your own fine-tuned weights, add them to the `weights/` folder
and push to GitHub (use Git LFS if files exceed 100 MB):

```bash
git lfs install
git lfs track "weights/*.pth"
git add .gitattributes weights/
git commit -m "add fine-tuned weights"
git push
```

## Fine-tuning (optional)

To fine-tune on your own dataset (e.g. MURA):

```
dataset/
  train/
    Fractured/   ← X-ray images of fractures
    Normal/      ← X-ray images without fractures
  val/
    Fractured/
    Normal/
```

Then run:

```bash
python train.py --model densenet121 --data_dir ./dataset --epochs 20
```

The best weights are saved to `weights/densenet121.pth` automatically.

## Class Index Convention
- **Index 0** = Fractured
- **Index 1** = Normal

(ImageFolder assigns alphabetically: F < N)
