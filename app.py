#TUBERCULOSIS DETECTION SYSTEM - MobileNetV2 Production Version
#================================================================
#A robust, production-grade system for TB detection from chest X-rays using MobileNetV2 transfer learning. 
#Author: Joseph Marindi
#Date: 2024-06-01
#Version: 1.0.0 
#This system is designed for deployment in clinical settings, providing accurate TB screening with explainable AI insights via Grad-CAM visualizations.

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageOps
from fpdf import FPDF
import datetime
import io

# --- 1. Page Configuration ---
st.set_page_config(page_title="TB AI Diagnostic Pro", page_icon="🩻", layout="wide")

# --- 2. Robust Model & Layer Detection ---
@st.cache_resource
def load_mobilenet_model():
    try:
        model = tf.keras.models.load_model('models/mobilenetv2_best.keras')
        
        # IMPROVED LAYER SEARCH: Look for the specific MobileNetV2 bottleneck
        last_conv_layer_name = None
        
        # Strategy A: Look for common MobileNetV2 names in Keras 3
        for layer_name in ['out_relu', 'Conv_1', 'top_activation', 'conv2d_last']:
            try:
                model.get_layer(layer_name)
                last_conv_layer_name = layer_name
                break
            except ValueError:
                continue
        
        # Strategy B: If names fail, find the last layer that has 4D output tensors
        if not last_conv_layer_name:
            for layer in reversed(model.layers):
                # In Keras 3, we use the layer's output property safely
                if hasattr(layer, 'output') and len(layer.output.shape) == 4:
                    last_conv_layer_name = layer.name
                    break
                    
        return model, last_conv_layer_name
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# --- 3. Grad-CAM & PDF Logic ---
def generate_gradcam(img_array, model, last_conv_layer_name):
    if not last_conv_layer_name:
        return None
    try:
        grad_model = tf.keras.models.Model(
            model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            class_channel = preds[:, 0]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        return np.power(heatmap.numpy(), 2) 
    except:
        return None

def create_pdf(verdict, score, threshold):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "TB X-Ray Analysis Report", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, f"Final Verdict: {verdict}", ln=True)
    pdf.cell(0, 10, f"Confidence Score: {score*100:.2f}%", ln=True)
    pdf.cell(0, 10, f"Decision Threshold used: {threshold}", ln=True)
    pdf.ln(20)
    pdf.set_font("Arial", 'I', 10)
    pdf.multi_cell(0, 10, "Disclaimer: This is an AI-generated screening assistance report. It is NOT a final diagnosis. Please consult a qualified radiologist.")
    return pdf.output(dest='S').encode('latin-1')

# --- 4. Main App UI ---
def main():
    st.sidebar.header("⚙️ Settings")
    threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.05)
    
    st.title("🩻 TB Screening & Explainable AI")
    model, last_conv_name = load_mobilenet_model()

    uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        
        if st.button("🔍 Run Full Analysis"):
            if model:
                with st.spinner("Analyzing Lung Pathology..."):
                    img_resized = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
                    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

                    preds = model.predict(img_preprocessed)
                    score = float(preds[0][0])
                    verdict = "TB POSITIVE" if score >= threshold else "NORMAL"

                    heatmap = generate_gradcam(img_preprocessed, model, last_conv_name)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image, caption="Original X-ray", use_container_width=True)
                    with col2:
                        if heatmap is not None:
                            heatmap_resized = cv2.resize(heatmap, (image.size[0], image.size[1]))
                            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                            superimposed = cv2.addWeighted(heatmap_colored, 0.4, np.array(image), 0.6, 0)
                            st.image(superimposed, caption="AI Heatmap", use_container_width=True)
                        else:
                            st.error("Grad-CAM Layer Identification Failed.")

                    st.divider()
                    if verdict == "TB POSITIVE":
                        st.error(f"## Result: {verdict}")
                    else:
                        st.success(f"## Result: {verdict}")
                    
                    st.metric("Model Confidence", f"{score*100:.2f}%")

                    # PDF Download Feature
                    pdf_data = create_pdf(verdict, score, threshold)
                    st.download_button(label="📥 Download Diagnostic Report", data=pdf_data, 
                                       file_name="TB_Report.pdf", mime="application/pdf")

if __name__ == "__main__":
    main()