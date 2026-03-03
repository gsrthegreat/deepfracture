import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import os
import cv2
import numpy as np

from model import DeepFractureModel, BaselineCNN
from utils import get_transforms, extract_image_features
from gradcam import GradCAM, overlay_heatmap
from risk_model import RiskStratificationModel

st.set_page_config(
    page_title="DeepFracture: Research Edition",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1E3A8A; font-weight: bold; text-align: center; margin-bottom: 2rem; }
    .prediction-box { padding: 20px; border-radius: 10px; background-color: #F3F4F6; color: #1F2937; border-left: 5px solid #3B82F6; margin-bottom: 20px;}
    .risk-box { padding: 15px; border-radius: 8px; font-weight: bold; text-align: center; font-size: 1.2rem; }
    .risk-low { background-color: #D1FAE5; color: #065F46; border: 1px solid #34D399; }
    .risk-medium { background-color: #FEF3C7; color: #92400E; border: 1px solid #FBBF24; }
    .risk-high { background-color: #FEE2E2; color: #991B1B; border: 1px solid #F87171; }
    .xai-box { padding: 15px; background-color: #EEF2FF; color: #3730A3; border: 1px solid #C7D2FE; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models(model_type, _cache_buster="v2"):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if 'ResNet50' in model_type:
        model = DeepFractureModel(num_classes=2, freeze_layers=True)
        weight_path = 'best_resnet.pth'
    else:
        model = BaselineCNN(num_classes=2)
        weight_path = 'best_baseline.pth'
        
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device))
    
    model.to(device)
    model.eval()
    
    risk_model = RiskStratificationModel()
    return model, risk_model, device

def main():
    st.markdown("<div class='main-header'>🦴 DeepFracture: XAI & Risk-Aware Framework</div>", unsafe_allow_html=True)
    
    st.sidebar.title("Research Settings")
    model_choice = st.sidebar.radio("Select Model Architecture:", ['ResNet50 (Transfer Learning)', 'Baseline CNN (Scratch)'])
    
    st.sidebar.info("""
    **DeepFracture (Research Edition)** integrates:
    1. **Rigorous Pipelines**: CLAHE, Imbalance Handling.
    2. **XAI**: Grad-CAM for clinical trust.
    3. **Structural Risk**: Random Forest over GLCM, Density, Intensity features.
    """)
    
    model, risk_model, device = load_models(model_choice)
    # Target layer logic
    if 'ResNet50' in model_choice:
        target_layer = model.resnet.layer4[-1]
    else:
        target_layer = model.features[-3] # Last Conv's output
        
    classes = ['Fractured', 'Normal'] 
    
    tab1, tab2, tab3 = st.tabs(["🩺 Diagnosis", "📊 Performance Metrics", "🧠 XAI Details"])
    
    with tab1:
        st.write("### Upload Radiograph")
        uploaded_file = st.file_uploader("Choose X-ray image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            image = Image.open(uploaded_file).convert('RGB')
            temp_img_path = "temp_uploaded_img.jpg"
            image.save(temp_img_path)
            
            with col1:
                st.image(image, caption="Uploaded X-Ray", use_container_width=True)
                
            transform = get_transforms(is_train=False)
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            with st.spinner(f'Running framework using {model_choice}...'):
                with torch.no_grad():
                    output = model(input_tensor)
                    probs = F.softmax(output, dim=1)
                    conf_score, pred_class = torch.max(probs, 1)
                    pred_label = classes[pred_class.item()]
                    conf_val = conf_score.item()
                    
                # XAI (Grad-CAM depends on architecture)
                grad_cam = GradCAM(model, target_layer)
                heatmap, _ = grad_cam.generate_heatmap(input_tensor, target_class=pred_class.item())
                heatmap_path = f"heatmap_{pred_label}.jpg"
                overlay_heatmap(temp_img_path, heatmap, save_path=heatmap_path)
                
                # Risk Module
                mean_int, std_dev, edge_density, contrast, homogeneity = extract_image_features(temp_img_path)
                fracture_prob = probs[0][0].item() if classes[0] == 'Fractured' else probs[0][1].item()
                risk_category = risk_model.predict_risk(fracture_prob, mean_int, std_dev, edge_density, contrast, homogeneity)
                
            with col2:
                st.markdown(f"""
                <div class="prediction-box">
                    <h4>Diagnosis: <strong>{pred_label}</strong></h4>
                    <p>Confidence: <strong>{conf_val:.2%}</strong></p>
                    <progress value="{conf_val}" max="1" style="width: 100%;"></progress>
                </div>
                """, unsafe_allow_html=True)
                
                risk_class = {"Low Risk": "risk-low", "Medium Risk": "risk-medium", "High Risk": "risk-high"}.get(risk_category, "risk-low")
                    
                st.markdown("#### Structural Risk Factor")
                st.markdown(f'<div class="risk-box {risk_class}">{risk_category}</div>', unsafe_allow_html=True)
                
                with st.expander("View Extracted Biomarkers"):
                    st.write(f"- Fracture Probability: **{fracture_prob:.4f}**")
                    st.write(f"- GLCM Contrast: **{contrast:.2f}**")
                    st.write(f"- GLCM Homogeneity: **{homogeneity:.4f}**")
                    st.write(f"- Edge Density: **{edge_density:.4f}**")
                    st.write(f"- Intensity (Mean / Std): **{mean_int:.1f} / {std_dev:.1f}**")
                
            st.write("---")
            st.write("### Explainable AI (Grad-CAM)")
            hcol1, hcol2 = st.columns(2)
            with hcol1:
                st.image(image, caption="Original", use_container_width=True)
            with hcol2:
                st.image(heatmap_path, caption=f"Grad-CAM Heatmap via {model_choice}", use_container_width=True)
                
            if os.path.exists(temp_img_path): os.remove(temp_img_path)

    with tab2:
        model_key = 'resnet' if 'ResNet50' in model_choice else 'baseline'
        st.write(f"### Performance Dashboard ({model_choice})")
        
        col_cm, col_roc = st.columns(2)
        has_metrics = False
        if os.path.exists(f'confusion_matrix_{model_key}.png'):
            col_cm.image(f'confusion_matrix_{model_key}.png', caption="Confusion Matrix", use_container_width=True)
            has_metrics = True
        if os.path.exists(f'roc_curve_{model_key}.png'):
            col_roc.image(f'roc_curve_{model_key}.png', caption="ROC Curve", use_container_width=True)
            has_metrics = True
            
        if not has_metrics:
            st.info(f"Metrics not found for {model_choice}. Run `python train.py --model {model_key}` locally.")
            
        st.write("### Error Analysis")
        error_dir = f'error_analysis_{model_key}'
        if os.path.exists(error_dir) and len(os.listdir(error_dir)) > 0:
            st.warning("Misclassified images logged during latest evaluation:")
            err_files = os.listdir(error_dir)[:5] # Show max 5
            cols = st.columns(len(err_files))
            for i, f in enumerate(err_files):
                cols[i].image(os.path.join(error_dir, f), caption=f, use_container_width=True)
            st.write("**Possible Reasons for Misclassification:** Ambiguous fracture lines, poor contrast, overlapping bone structures, or foreign artifacts (implants/casts) disrupting the texture features.")
        else:
            st.write("No error analysis logs found yet.")

    with tab3:
        st.write("### The Importance of Explainable AI (XAI) in Medicine")
        st.markdown("""
        <div class="xai-box">
            <h4>Why do we need Grad-CAM?</h4>
            <p>Black-box neural networks often achieve high accuracy but fail to build <b>clinical trust</b>. In medical imaging, a model predicting "Fracture" is not enough; the physician must know <i>why</i> the model made that decision to rule out <b>spurious correlations</b> (e.g., the model looking at text markers on the X-ray instead of the bone).</p>
            <ul>
                <li><b>Transparency:</b> Grad-CAM highlights the exact spatial regions influencing the fully-connected layers.</li>
                <li><b>Verification:</b> Allows radiologists to verify if the AI focuses on actual cortical discontinuities or joint effusions.</li>
                <li><b>Error Diagnostics:</b> Helps AI engineers during error analysis to understand if misclassifications are due to model blindness or dataset bias.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
