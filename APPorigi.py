import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="TB X-Ray Diagnostic Ensemble",
    page_icon="🩻",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .metric-container { background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .stAlert { margin-top: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. Load Models (Optimized with Caching) ---
@st.cache_resource
def load_all_models():
    """Load the three specific architectures from the /models directory."""
    try:
        # Update these filenames to match exactly what you uploaded to GitHub
        custom_cnn = tf.keras.models.load_model('models/custom_cnn.h5')
        mobilenet_v2 = tf.keras.models.load_model('models/mobilenetv2_tb.h5')
        resnet50 = tf.keras.models.load_model('models/resnet50_tb.h5')
        return custom_cnn, mobilenet_v2, resnet50
    except Exception as e:
        st.error(f"Error loading models: {e}. Check if files exist in the 'models/' folder.")
        return None, None, None

# --- 3. Specialized Preprocessing ---
def preprocess_image(image, model_type):
    """
    Applies specific preprocessing based on the architecture requirements.
    Standard input size for these models is 224x224.
    """
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)

    if model_type == "mobilenet":
        # MobileNetV2 expects scaling between -1 and 1
        return tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    elif model_type == "resnet":
        # ResNet50 expects Zero-centering (Caffe style) or 0-1 depending on training
        # Defaulting to standard ResNet50 preprocess_input
        return tf.keras.applications.resnet50.preprocess_input(img_array)
    else:
        # Custom CNN typically uses simple 0-1 scaling
        return img_array / 255.0

# --- 4. Main UI ---
def main():
    st.title("🩻 TB X-Ray Ensemble Diagnostic System")
    st.write("Professional-grade screening using Custom CNN, MobileNetV2, and ResNet50.")
    
    # Load Models
    cnn_model, mobile_model, res_model = load_all_models()
    
    # File Upload
    uploaded_file = st.file_uploader("Upload Chest X-ray (JPEG/PNG)", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        
        # Display layout
        col_img, col_res = st.columns([1, 1])
        
        with col_img:
            st.image(image, caption="Uploaded X-ray Scan", use_column_width=True)
        
        if st.button(" Analyze X-ray"):
            if cnn_model and mobile_model and res_model:
                with st.spinner("Processing through multiple neural networks..."):
                    # 1. Individual Preprocessing & Prediction
                    p1 = cnn_model.predict(preprocess_image(image, "custom"))[0][0]
                    p2 = mobile_model.predict(preprocess_image(image, "mobilenet"))[0][0]
                    p3 = res_model.predict(preprocess_image(image, "resnet"))[0][0]
                    
                    scores = [p1, p2, p3]
                    names = ["Custom CNN", "MobileNetV2", "ResNet50"]
                    
                    # 2. Results Dashboard
                    with col_res:
                        st.subheader("Model Breakdown")
                        for i in range(3):
                            label = "TB Detected" if scores[i] > 0.5 else "Normal"
                            color = "inverse" if scores[i] > 0.5 else "normal"
                            st.metric(names[i], label, delta=f"{scores[i]*100:.1f}% Score", delta_color=color)
                        
                        # 3. Final Weighted Consensus
                        # (Averaging the 3 probabilities)
                        final_prob = sum(scores) / 3
                        
                        st.divider()
                        if final_prob > 0.5:
                            st.error(f"### CONSENSUS: POSITIVE ({final_prob*100:.1f}%)")
                            st.write("**Recommendation:** Clinical correlation is required. Refer for GeneXpert or Sputum Culture.")
                        else:
                            st.success(f"### CONSENSUS: NEGATIVE ({(1-final_prob)*100:.1f}% Confidence)")
            else:
                st.error("Models not found. Please ensure they are in the 'models/' directory.")

    st.caption("Disclaimer: This tool is for research assistance only. Not a replacement for professional medical advice.")

if __name__ == "__main__":
    main()