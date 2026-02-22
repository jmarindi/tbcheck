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
import datetime

# --- 1. Page Configuration ---
st.set_page_config(page_title="TB AI Diagnostic Pro", page_icon="🩻", layout="wide")

# --- 2. Robust Model Loading ---
@st.cache_resource
def load_mobilenet_model():
    try:
        # UPDATED: Using the best .keras file as suggested
        model = tf.keras.models.load_model('models/mobilenetv2_best.keras')
        
        # FIND CONV LAYER: Specifically search for the last layer with 4D output
        last_conv_layer_name = None
        for layer in reversed(model.layers):
            # Grad-CAM needs a layer with spatial dimensions (typically Conv2D)
            if len(layer.output_shape) == 4 and "conv" in layer.name.lower():
                last_conv_layer_name = layer.name
                break
        
        # Fallback: Many MobileNetV2 models use 'out_relu' or 'Conv_1'
        if not last_conv_layer_name:
            for layer_name in ['out_relu', 'Conv_1', 'top_conv']:
                try:
                    model.get_layer(layer_name)
                    last_conv_layer_name = layer_name
                    break
                except ValueError:
                    continue
                    
        return model, last_conv_layer_name
    except Exception as e:
        st.error(f"Error loading model or identifying layers: {e}")
        return None, None

# --- 3. Grad-CAM Logic ---
def generate_gradcam(img_array, model, last_conv_layer_name):
    # Ensure a layer name was actually found
    if not last_conv_layer_name:
        return None

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
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

def display_gradcam(original_image, heatmap, alpha=0.4):
    if heatmap is None:
        return None
    heatmap = cv2.resize(heatmap, (original_image.size[0], original_image.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(heatmap, alpha, np.array(original_image), 1 - alpha, 0)
    return superimposed_img

# --- 4. Main UI ---
def main():
    st.sidebar.header("⚙️ Settings")
    threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.05)
    
    st.title("🩻 TB Screening & Explainable AI")
    model, last_conv_name = load_mobilenet_model()
    
    # Debug info (hidden by default)
    with st.expander("System Info"):
        st.write(f"Detected Target Layer: `{last_conv_name}`")

    uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        
        if st.button("🔍 Run Full Analysis"):
            if model and last_conv_name:
                with st.spinner("Analyzing..."):
                    img_resized = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
                    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

                    preds = model.predict(img_preprocessed)
                    score = float(preds[0][0])
                    verdict = "TB POSITIVE" if score >= threshold else "NORMAL"

                    heatmap = generate_gradcam(img_preprocessed, model, last_conv_name)
                    cam_image = display_gradcam(image, heatmap)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image, caption="Original", use_container_width=True)
                    with col2:
                        if cam_image is not None:
                            st.image(cam_image, caption="AI Heatmap", use_container_width=True)
                        else:
                            st.warning("Could not generate heatmap.")

                    st.divider()
                    if verdict == "TB POSITIVE":
                        st.error(f"## Result: {verdict}")
                    else:
                        st.success(f"## Result: {verdict}")
                    st.metric("Confidence", f"{score*100:.2f}%")
            else:
                st.error("Missing model or target layers. Check 'System Info'.")

if __name__ == "__main__":
    main()