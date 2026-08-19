import sys
import os
import json
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config
from src.predict import Predictor
from src.gradcam import GradCAM
from database.database import PredictionDB

# Streamlit Page Setup
st.set_page_config(
    page_title="PneumoScan AI | Medical Diagnostics System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Theme
st.markdown("""
<style>
    .stApp {
        background-color: #070d19;
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, rgba(16, 37, 66, 0.8) 0%, rgba(13, 27, 42, 0.95) 100%);
        border: 1px solid rgba(0, 242, 254, 0.2);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
    }
    
    .main-header h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .glass-card {
        background: rgba(17, 34, 64, 0.65);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
    }

    .badge-pneumonia {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(185, 28, 28, 0.4) 100%);
        border: 1px solid #ef4444;
        color: #fca5a5;
        padding: 16px 24px;
        border-radius: 12px;
        text-align: center;
        font-weight: 700;
        font-size: 1.4rem;
        margin-bottom: 16px;
    }
    
    .badge-healthy {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(4, 120, 87, 0.4) 100%);
        border: 1px solid #10b981;
        color: #6ee7b7;
        padding: 16px 24px;
        border-radius: 12px;
        text-align: center;
        font-weight: 700;
        font-size: 1.4rem;
        margin-bottom: 16px;
    }

    .metric-pill {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 10px;
        padding: 12px 16px;
        text-align: center;
    }
    .metric-pill .value { font-size: 1.5rem; font-weight: 700; color: #38bdf8; }
    .metric-pill .label { font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; }

    .disclaimer-banner {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.3);
        color: #fcd34d;
        padding: 14px 20px;
        border-radius: 10px;
        font-size: 0.9rem;
        margin-bottom: 20px;
    }

    .footer {
        text-align: center; color: #64748b; font-size: 0.85rem; padding: 24px 0;
        border-top: 1px solid rgba(255, 255, 255, 0.05); margin-top: 40px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Database
db = PredictionDB()

# Cache Model Engine
@st.cache_resource
def get_predictor(model_type):
    return Predictor(model_type=model_type)

# App Header
st.markdown("""
<div class="main-header">
    <h1>🫁 PneumoScan AI</h1>
    <p>Computer Vision TensorFlow Deep Learning System for Automated Chest X-Ray Pneumonia Detection</p>
</div>
""", unsafe_allow_html=True)

# Medical Disclaimer Banner
st.markdown("""
<div class="disclaimer-banner">
    ⚠️ <strong>Medical Disclaimer:</strong> This system is intended for educational and research purposes only. It is not a medical diagnostic tool and should not replace evaluation by a qualified healthcare professional.
</div>
""", unsafe_allow_html=True)

# Sidebar Controls
with st.sidebar:
    st.image("https://img.icons8.com/isometric-reflection/100/lungs.png", width=70)
    st.title("System Control")
    
    st.markdown("### 🤖 Model Selection")
    model_choice = st.selectbox(
        "Choose Deep Learning Architecture:",
        ["custom_cnn", "mobilenetv2", "resnet50"],
        format_func=lambda x: {"custom_cnn": "Custom CNN", "mobilenetv2": "MobileNetV2 (Transfer Learning)", "resnet50": "ResNet50 (Transfer Learning)"}[x]
    )
    
    st.markdown("### ⚙️ Diagnostics Settings")
    show_gradcam = st.toggle("Enable Grad-CAM Visual Heatmap", value=True)
    invert_view = st.checkbox("Invert Radiograph View (Negative)")

    st.markdown("---")
    predictor = get_predictor(model_choice)
    if predictor.is_loaded:
        st.success(f"✅ Active Model: `{model_choice.upper()}`")
    else:
        st.warning(f"⚠️ Model `{model_choice}` training required...")
    st.info(f"Compute Engine: `TensorFlow 2.x (Keras)`")

# Tabs Setup
tab_inference, tab_history = st.tabs([
    "🔍 Radiograph Analysis", 
    "🗄️ Database Prediction History"
])

# Tab 1: Radiograph Analysis
with tab_inference:
    col_input, col_results = st.columns([1, 1.2])
    image_to_process = None
    input_source_name = ""

    with col_input:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("1. Select Input X-Ray Image")
        input_mode = st.radio("Input Source:", ["Sample X-Ray Gallery", "Upload X-Ray File"], horizontal=True)

        if input_mode == "Upload X-Ray File":
            uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "png", "jpeg"])
            if uploaded_file is not None:
                image_to_process = Image.open(uploaded_file).convert("RGB")
                input_source_name = uploaded_file.name
        else:
            test_normal_dir = os.path.join(config.DATA_DIR, 'test', 'NORMAL')
            test_pneu_dir = os.path.join(config.DATA_DIR, 'test', 'PNEUMONIA')

            sample_options = {}
            if os.path.exists(test_normal_dir):
                for f in os.listdir(test_normal_dir)[:5]:
                    if f.endswith(('.jpeg', '.jpg', '.png')):
                        sample_options[f"Normal Case - {f}"] = os.path.join(test_normal_dir, f)
            if os.path.exists(test_pneu_dir):
                for f in os.listdir(test_pneu_dir)[:5]:
                    if f.endswith(('.jpeg', '.jpg', '.png')):
                        sample_options[f"Pneumonia Case - {f}"] = os.path.join(test_pneu_dir, f)

            if sample_options:
                selected_sample_key = st.selectbox("Choose a sample chest X-ray:", list(sample_options.keys()))
                sample_path = sample_options[selected_sample_key]
                image_to_process = Image.open(sample_path).convert("RGB")
                input_source_name = selected_sample_key
            else:
                st.warning("No sample files found in test folder.")

        if image_to_process is not None:
            processed_img = image_to_process.copy()
            if invert_view:
                processed_img = Image.fromarray(255 - np.array(processed_img))

            st.image(processed_img, caption=f"Active Image: {input_source_name}", use_column_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col_results:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("2. AI Diagnostic Inference")

        if image_to_process is not None:
            if not predictor.is_loaded:
                st.warning(f"Please train model `{model_choice}` using `python src/train.py` first.")
            else:
                with st.spinner("Processing X-Ray image through TensorFlow model..."):
                    res = predictor.predict_image(image_to_process)
                    pred_class = res['label']
                    normal_prob = res['normal_probability']
                    pneu_prob = res['pneumonia_probability']
                    conf = res['confidence']

                    # Log prediction to SQLite Database
                    db.log_prediction(
                        filename=input_source_name,
                        prediction=pred_class,
                        confidence=round(conf * 100, 2),
                        normal_prob=round(normal_prob * 100, 2),
                        pneumonia_prob=round(pneu_prob * 100, 2),
                        model_used=model_choice
                    )

                # Diagnostic Badge
                if pred_class == "PNEUMONIA":
                    st.markdown(f"""
                    <div class="badge-pneumonia">
                        🚨 PNEUMONIA DETECTED<br>
                        <span style="font-size: 1rem; font-weight: 500;">Confidence: {pneu_prob*100:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="badge-healthy">
                        ✅ HEALTHY / NORMAL<br>
                        <span style="font-size: 1rem; font-weight: 500;">Confidence: {normal_prob*100:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)

                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.markdown(f'<div class="metric-pill"><div class="value">{normal_prob*100:.1f}%</div><div class="label">Normal Probability</div></div>', unsafe_allow_html=True)
                with col_m2:
                    st.markdown(f'<div class="metric-pill"><div class="value">{pneu_prob*100:.1f}%</div><div class="label">Pneumonia Probability</div></div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Risk Gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=pneu_prob * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Pneumonia Infection Risk Score", 'font': {'color': "#ffffff", 'size': 16}},
                    number={'suffix': "%", 'font': {'color': "#ef4444" if pred_class == "PNEUMONIA" else "#10b981"}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
                        'bar': {'color': "#ef4444" if pred_class == "PNEUMONIA" else "#10b981"},
                        'bgcolor': "rgba(15, 23, 42, 0.8)",
                        'bordercolor': "rgba(255, 255, 255, 0.1)",
                        'steps': [
                            {'range': [0, 50], 'color': "rgba(16, 185, 129, 0.15)"},
                            {'range': [50, 100], 'color': "rgba(239, 68, 68, 0.15)"}
                        ],
                    }
                ))
                fig_gauge.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_gauge, use_container_width=True)

                # Grad-CAM Heatmap Analysis
                if show_gradcam:
                    st.markdown("#### 🎯 Grad-CAM Visual Heatmap Analysis")
                    gradcam_engine = GradCAM(predictor.model)
                    cam, _, _ = gradcam_engine.generate_heatmap(res['img_batch'])
                    heatmap, overlay = gradcam_engine.overlay_heatmap(image_to_process, cam)

                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(heatmap, caption="Grad-CAM Activation Map", use_column_width=True)
                    with c2:
                        st.image(overlay, caption="Visual Overlay", use_column_width=True)

        else:
            st.info("👈 Select or upload an X-ray image from the left panel to run diagnostic analysis.")

        st.markdown('</div>', unsafe_allow_html=True)

# Tab 2: Database History Log
with tab_history:
    st.subheader("🗄️ Diagnostic Predictions Database Log")
    st.caption(f"Connected to SQLite Database: `{config.DB_PATH}`")

    rows = db.fetch_all_predictions(limit=100)
    if rows:
        df_history = pd.DataFrame(rows, columns=["ID", "Filename", "Prediction", "Confidence (%)", "Normal Prob (%)", "Pneumonia Prob (%)", "Model Architecture", "Timestamp"])
        
        st.dataframe(df_history, use_container_width=True)

        col_dl, col_clr = st.columns([1, 1])
        with col_dl:
            csv_data = df_history.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Export Logs to CSV", csv_data, "predictions_log.csv", "text/csv")
        with col_clr:
            if st.button("🗑️ Clear Database Log"):
                db.clear_history()
                st.rerun()
    else:
        st.info("No prediction history recorded in database yet.")

# Footer
st.markdown("""
<div class="footer">
    PneumoScan AI Modular Diagnostic System • TensorFlow 2.x & Streamlit
</div>
""", unsafe_allow_html=True)
