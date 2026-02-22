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

# --- 1. Page Configuration ---
st.set_page_config(page_title="TB Screening & Explainable AI", page_icon="🩻", layout="wide")

# --- 2. Robust Model & Layer Detection ---
@st.cache_resource
def load_mobilenet_model():
    try:
        # Load  best .keras model
        model = tf.keras.models.load_model('models/mobilenetv2_best.keras')
        
        # FIND CONV LAYER: Iterate to find the last layer suitable for Grad-CAM
        # We check for convolutional types and specific names common in MobileNetV2
        last_conv_layer_name = None
        for layer in reversed(model.layers):
            layer_class = layer.__class__.__name__
            # MobileNetV2 last spatial layers often contain 'Conv' or 'relu'
            if 'Conv' in layer_class or 'ReLU' in layer_class:
                # Basic check: skip layers that are likely final dense layers
                if 'dense' not in layer.name.lower():
                    last_conv_layer_name = layer.name
                    break
        
        # Absolute fallbacks for MobileNetV2 architecture
        if not last_conv_layer_name:
            for fallback in ['out_relu', 'Conv_1', 'top_conv']:
                try:
                    model.get_layer(fallback)
                    last_conv_layer_name = fallback
                    break
                except ValueError:
                    continue
                    
        return model, last_conv_layer_name
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# --- 3. Grad-CAM Logic (Keras 3 Safe) ---
def generate_gradcam(img_array, model, last_conv_layer_name):
    if not last_conv_layer_name:
        return None

    # Create a sub-model to extract the activations and predictions
    try:
        grad_model = tf.keras.models.Model(
            model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
        )
    except Exception:
        return None

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, 0]

    # Gradients of the output class w.r.t. the last conv layer output
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Heatmap generation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return np.power(heatmap.numpy(), 2) 

def display_gradcam(original_image, heatmap, alpha=0.4):
    if heatmap is None: return None
    heatmap = cv2.resize(heatmap, (original_image.size[0], original_image.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(heatmap, alpha, np.array(original_image), 1 - alpha, 0)
    return superimposed_img

# --- 4. Main App UI ---
def main():
    st.sidebar.header("⚙️ Settings")
    threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.05)
    
    st.title("🩻 TB Screening & Explainable AI")
    model, last_conv_name = load_mobilenet_model()
    
    # Hide technical details in expander
    with st.expander("Technical System Info"):
        if last_conv_name:
            st.success(f"Grad-CAM Target Layer: `{last_conv_name}`")
        else:
            st.warning("Could not identify target convolutional layer.")

    uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        
        if st.button("🔍 Run Full Analysis"):
            if model:
                with st.spinner("Analyzing Lung Features..."):
                    # MobileNetV2 Preprocessing
                    img_resized = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
                    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

                    # Predictions
                    preds = model.predict(img_preprocessed)
                    score = float(preds[0][0])
                    verdict = "TB POSITIVE" if score >= threshold else "NORMAL"

                    # Visualization
                    heatmap = generate_gradcam(img_preprocessed, model, last_conv_name)
                    cam_image = display_gradcam(image, heatmap)

                    # Results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image, caption="Uploaded X-ray", use_container_width=True)
                    with col2:
                        if cam_image is not None:
                            st.image(cam_image, caption="AI Heatmap (Red = Concern Area)", use_container_width=True)
                        else:
                            st.info("Heatmap not available for this model configuration.")

                    st.divider()
                    if verdict == "TB POSITIVE":
                        st.error(f"## Result: {verdict}")
                    else:
                        st.success(f"## Result: {verdict}")
                    st.metric("Model Confidence Score", f"{score*100:.2f}%")
            else:
                st.error("Model not found in /models/ folder.")

if __name__ == "__main__":
    main()