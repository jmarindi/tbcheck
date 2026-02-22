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

# --- 1. Page Configuration ---
st.set_page_config(page_title="TB AI Diagnostic Pro", page_icon="🩻", layout="wide")

# --- 2. Enhanced Model & Layer Detection ---
@st.cache_resource
def load_mobilenet_model():
    try:
        # Loading your best-performing Keras 3 model from the file list
        model = tf.keras.models.load_model('models/mobilenetv2_best.keras')
        
        # DYNAMIC SEARCH: Recursive check to find the deep-seated conv layer
        def find_last_conv(layer):
            if hasattr(layer, 'layers'): # Check if it's a nested model
                for sub_layer in reversed(layer.layers):
                    res = find_last_conv(sub_layer)
                    if res: return res
            if isinstance(layer, tf.keras.layers.Conv2D) or 'conv' in layer.name.lower():
                # Ensure it's not a 1x1 bottleneck by checking output rank
                if len(layer.output.shape) == 4:
                    return layer.name
            return None

        last_conv_layer_name = find_last_conv(model)
        
        # Fallback to known MobileNetV2 bottleneck names if recursion fails
        if not last_conv_layer_name:
            for name in ['Conv_1', 'out_relu', 'top_conv']:
                try:
                    model.get_layer(name)
                    last_conv_layer_name = name
                    break
                except: continue
                    
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
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, 0]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
        return np.power(heatmap.numpy(), 2) 
    except:
        return None

def create_pdf(verdict, score, threshold):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", 'B', 16)
    pdf.cell(0, 10, "TB X-Ray Analysis Report", new_x="LMARGIN", new_y="NEXT", align='C')
    pdf.set_font("helvetica", size=12)
    pdf.cell(0, 10, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", new_x="LMARGIN", new_y="NEXT", align='C')
    pdf.ln(10)
    pdf.set_font("helvetica", 'B', 12)
    pdf.cell(0, 10, f"Final Result: {verdict}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 10, f"Confidence: {score*100:.2f}%", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    pdf.set_font("helvetica", 'I', 10)
    pdf.multi_cell(0, 10, "Disclaimer: This AI report is for screening assistance only. Please consult a professional radiologist for final diagnosis.")
    # Safe output for modern FPDF2
    return pdf.output()

# --- 4. Main UI ---
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
                with st.spinner("Analyzing Pathology..."):
                    # Preprocessing
                    img_resized = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
                    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

                    # Predictions
                    preds = model.predict(img_preprocessed)
                    score = float(preds[0][0])
                    verdict = "TB POSITIVE" if score >= threshold else "NORMAL"

                    heatmap = generate_gradcam(img_preprocessed, model, last_conv_name)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image, caption="Uploaded Scan", use_container_width=True)
                    with col2:
                        if heatmap is not None:
                            h_resized = cv2.resize(heatmap, (image.size[0], image.size[1]))
                            h_colored = cv2.applyColorMap(np.uint8(255 * h_resized), cv2.COLORMAP_JET)
                            superimposed = cv2.addWeighted(h_colored, 0.4, np.array(image), 0.6, 0)
                            st.image(superimposed, caption="AI Pathology Heatmap", use_container_width=True)
                        else:
                            st.warning("Heatmap layer could not be localized for this model.")

                    st.divider()
                    if verdict == "TB POSITIVE":
                        st.error(f"## Result: {verdict}")
                    else:
                        st.success(f"## Result: {verdict}")
                    
                    st.metric("Confidence", f"{score*100:.2f}%")

                    # Robust PDF Download
                    try:
                        pdf_bytes = create_pdf(verdict, score, threshold)
                        st.download_button(label="📥 Download PDF Report", data=pdf_bytes, 
                                           file_name="TB_Report.pdf", mime="application/pdf")
                    except Exception as e:
                        st.error(f"PDF Generation Failed: {e}")

if __name__ == "__main__":
    main()