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

# --- 2. Model & Layer Setup ---
@st.cache_resource
def load_mobilenet_model():
    try:
        # UPDATED: Pointing to your best .keras model file
        model = tf.keras.models.load_model('models/mobilenetv2_best.keras')
        
        # Identify the last convolutional layer automatically for Grad-CAM
        last_conv_layer_name = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break
        return model, last_conv_layer_name
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# --- 3. Grad-CAM Logic ---
def generate_gradcam(img_array, model, last_conv_layer_name):
    # Create a model that maps input image to activations of the last conv layer as well as output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Compute the gradient of the top predicted class for the input image
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, 0]

    # This is the gradient of the output neuron with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Vector of intensity values over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in the feature map array by "how important this channel is" 
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization, we normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    # Apply power transform to sharpen the focal points
    return np.power(heatmap.numpy(), 2) 

def display_gradcam(original_image, heatmap, alpha=0.4):
    # Resize and colorize heatmap
    heatmap = cv2.resize(heatmap, (original_image.size[0], original_image.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Overlay on original
    original_img_np = np.array(original_image)
    superimposed_img = cv2.addWeighted(heatmap, alpha, original_img_np, 1 - alpha, 0)
    return superimposed_img

# --- 4. Main App UI ---
def main():
    st.sidebar.header("⚙️ Diagnostic Settings")
    # User-adjustable threshold for prediction sensitivity
    threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.05)
    st.sidebar.info(f"Predictions above {threshold*100:.0f}% will be flagged as TB Positive.")

    st.title("🩻 TB Screening & Explainable AI")
    st.markdown("This system uses **MobileNetV2** for screening and **Grad-CAM** to highlight areas of concern.")
    
    model, last_conv_name = load_mobilenet_model()
    uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        
        if st.button("🔍 Run Full Analysis"):
            if model:
                with st.spinner("Analyzing Lung Features..."):
                    # Preprocessing for MobileNetV2
                    size = (224, 224)
                    img_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
                    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

                    # Model Prediction
                    preds = model.predict(img_preprocessed)
                    score = float(preds[0][0])
                    verdict = "TB POSITIVE" if score >= threshold else "NORMAL"

                    # Visualization
                    heatmap = generate_gradcam(img_preprocessed, model, last_conv_name)
                    cam_image = display_gradcam(image, heatmap)

                    # UI Layout
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image, caption="Uploaded X-ray", use_container_width=True)
                    with col2:
                        st.image(cam_image, caption="Pathology Heatmap (Red = Concern Area)", use_container_width=True)

                    st.divider()
                    
                    if verdict == "TB POSITIVE":
                        st.error(f"## Result: {verdict}")
                    else:
                        st.success(f"## Result: {verdict}")

                    st.metric(label="Model Confidence Score", value=f"{score*100:.2f}%")
                    
                    if score >= threshold:
                        st.warning("Heatmap highlights regions that influenced the TB prediction. Please review with a medical professional.")
            else:
                st.error("Model not found. Please ensure 'models/mobilenetv2_best.keras' exists.")

if __name__ == "__main__":
    main()