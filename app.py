import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import os

from model import load_model, MODELS
from gradcam import GradCAM, render_heatmap
from preprocess import preprocess

st.set_page_config(
    page_title="DeepFractureAI",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.header {
    background: linear-gradient(135deg, #1e3a8a 0%, #1d4ed8 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}
.header h1 { font-size: 2.4rem; font-weight: 700; margin: 0; letter-spacing: -0.5px; }
.header p  { font-size: 1rem; opacity: 0.85; margin: 0.4rem 0 0; }

.card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}

.result-fractured {
    background: #fef2f2;
    border-left: 5px solid #ef4444;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
}
.result-normal {
    background: #f0fdf4;
    border-left: 5px solid #22c55e;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
}
.result-title { font-size: 1.5rem; font-weight: 700; margin-bottom: 0.3rem; }
.result-fractured .result-title { color: #b91c1c; }
.result-normal   .result-title { color: #15803d; }
.result-sub { font-size: 0.95rem; color: #475569; }

.metric-row { display: flex; gap: 1rem; margin-top: 1rem; }
.metric-box {
    flex: 1;
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    text-align: center;
}
.metric-label { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }
.metric-value { font-size: 1.3rem; font-weight: 700; color: #1e293b; margin-top: 0.2rem; }

.disclaimer {
    background: #fffbeb;
    border: 1px solid #fcd34d;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.82rem;
    color: #92400e;
    margin-top: 1rem;
}

.section-title { font-size: 1.1rem; font-weight: 600; color: #1e293b; margin-bottom: 0.6rem; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header">
  <h1>🦴 DeepFractureAI</h1>
  <p>Deep learning X-ray fracture detection with Grad-CAM visualisation</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
model_name = st.sidebar.selectbox(
    "Model",
    list(MODELS.keys()),
    help="DenseNet121 is fastest; EfficientNet-B3 is most accurate."
)
threshold = st.sidebar.slider(
    "Fracture threshold", 0.30, 0.80, 0.45, 0.01,
    help="Lower = more sensitive (fewer missed fractures). Higher = more specific."
)
show_raw_cam = st.sidebar.checkbox("Show raw CAM (no overlay)", False)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**How it works**
1. Upload an X-ray image
2. The model scores the probability of a fracture
3. Grad-CAM highlights which region drove the prediction
4. Edge analysis further localises fracture lines
""")

# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model weights…")
def get_model(name):
    return load_model(name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, target_layer = get_model(model_name)
model.to(device).eval()

# ── Upload ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Upload X-Ray Image</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Supported formats: JPG, PNG, JPEG",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded is None:
    st.info("👆 Upload an X-ray image to begin analysis.")
    st.stop()

# ── Inference ──────────────────────────────────────────────────────────────────
pil_img = Image.open(uploaded).convert("RGB")
orig_np  = np.array(pil_img)

tensor, preprocessed_np = preprocess(pil_img)
tensor = tensor.unsqueeze(0).to(device)

with st.spinner("Analysing…"):
    # Classification
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1)
        frac_prob = float(probs[0, 0])   # index 0 = Fractured

    is_fractured = frac_prob >= threshold
    pred_label   = "Fractured" if is_fractured else "Normal"
    confidence   = frac_prob if is_fractured else 1.0 - frac_prob

    # Grad-CAM (always for the predicted class)
    pred_idx = 0 if is_fractured else 1
    gcam     = GradCAM(model, target_layer)
    heatmap  = gcam.generate(tensor, pred_idx)           # float32 0-1, 224×224

    # Overlay on 224-px model view
    overlay_224 = render_heatmap(preprocessed_np, heatmap)

    # Overlay on original resolution
    h_orig, w_orig = orig_np.shape[:2]
    heatmap_orig   = cv2.resize(heatmap, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
    overlay_orig   = render_heatmap(orig_np, heatmap_orig)

    # Edge-based fracture localisation (Canny on CLAHE-enhanced image)
    gray     = cv2.cvtColor(preprocessed_np, cv2.COLOR_RGB2GRAY)
    edge_map = cv2.Canny(gray, 40, 120)

# ── Layout ─────────────────────────────────────────────────────────────────────
col_img, col_result = st.columns([1, 1], gap="large")

with col_img:
    st.markdown('<div class="section-title">Input Image</div>', unsafe_allow_html=True)
    st.image(pil_img, use_container_width=True)

with col_result:
    css_class = "result-fractured" if is_fractured else "result-normal"
    icon      = "🔴" if is_fractured else "🟢"
    st.markdown(f"""
    <div class="{css_class}">
        <div class="result-title">{icon} {pred_label}</div>
        <div class="result-sub">Model: {model_name} &nbsp;|&nbsp; Threshold: {threshold:.2f}</div>
        <div class="metric-row">
            <div class="metric-box">
                <div class="metric-label">Fracture probability</div>
                <div class="metric-value">{frac_prob:.1%}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Confidence</div>
                <div class="metric-value">{confidence:.1%}</div>
            </div>
        </div>
    </div>
    <div class="disclaimer">
        ⚠️ For research use only. Always consult a qualified radiologist for clinical decisions.
    </div>
    """, unsafe_allow_html=True)

# ── Heatmaps ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-title">Grad-CAM Visualisation — highlighted region drove the prediction</div>',
            unsafe_allow_html=True)

hcol1, hcol2, hcol3 = st.columns(3)
with hcol1:
    st.image(pil_img, caption="Original X-Ray", use_container_width=True)
with hcol2:
    st.image(overlay_224, caption="Grad-CAM (model resolution 224×224)", use_container_width=True)
with hcol3:
    st.image(overlay_orig, caption="Grad-CAM (original resolution)", use_container_width=True)

if show_raw_cam:
    raw_uint8 = np.uint8(255 * heatmap)
    colored   = cv2.applyColorMap(raw_uint8, cv2.COLORMAP_INFERNO)
    colored   = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    st.image(colored, caption="Raw CAM (no overlay)", use_container_width=True)

# ── Edge detail ────────────────────────────────────────────────────────────────
with st.expander("🔬 Edge Analysis — fracture line localisation"):
    ecol1, ecol2 = st.columns(2)
    with ecol1:
        st.image(preprocessed_np, caption="CLAHE-enhanced (model input)", use_container_width=True)
    with ecol2:
        edge_rgb = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2RGB)
        st.image(edge_rgb, caption="Canny edge map", use_container_width=True)

# ── Biomarkers ─────────────────────────────────────────────────────────────────
with st.expander("📊 Biomarkers"):
    from skimage.feature import graycomatrix, graycoprops
    gray_orig = cv2.cvtColor(orig_np, cv2.COLOR_RGB2GRAY)
    gray_small = cv2.resize(gray_orig, (256, 256))
    glcm = graycomatrix(gray_small, distances=[1], angles=[0],
                        levels=256, symmetric=True, normed=True)
    contrast    = float(graycoprops(glcm, 'contrast')[0, 0])
    homogeneity = float(graycoprops(glcm, 'homogeneity')[0, 0])
    edge_density = float(np.sum(edge_map > 0) / edge_map.size)

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Fracture prob",  f"{frac_prob:.4f}")
    b2.metric("Edge density",   f"{edge_density:.4f}")
    b3.metric("GLCM contrast",  f"{contrast:.1f}")
    b4.metric("GLCM homogeneity", f"{homogeneity:.4f}")
