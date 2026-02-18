import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
from fpdf import FPDF
import datetime

# --- 1. Page Configuration ---
st.set_page_config(page_title="TB Diagnostic Ensemble", page_icon="🩻", layout="wide")

# --- 2. Model Loading ---
@st.cache_resource
def load_all_models():
    try:
        # Replace with your actual .h5 filenames
        cnn = tf.keras.models.load_model('models/custom_cnn.h5')
        mobile = tf.keras.models.load_model('models/mobilenetv2_tb.h5')
        resnet = tf.keras.models.load_model('models/resnet50_tb.h5')
        return cnn, mobile, resnet
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

# --- 3. Preprocessing Logic ---
def preprocess_image(image, model_type):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)

    if model_type == "mobilenet":
        return tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    elif model_type == "resnet":
        return tf.keras.applications.resnet50.preprocess_input(img_array)
    else:
        return img_array / 255.0

# --- 4. PDF Generation Function ---
def create_pdf(results, final_verdict):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "TB X-Ray Analysis Report", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Model Predictions:", ln=True)
    pdf.set_font("Arial", size=12)
    for name, score in results.items():
        status = "Positive" if score > 0.5 else "Negative"
        pdf.cell(0, 10, f"- {name}: {status} ({score*100:.1f}%)", ln=True)
    
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, f"Final Consensus: {final_verdict}", ln=True)
    
    pdf.ln(20)
    pdf.set_font("Arial", 'I', 10)
    pdf.multi_cell(0, 10, "Disclaimer: This is an AI-generated screening report and must be reviewed by a qualified medical professional.")
    
    return pdf.output(dest='S').encode('latin-1')

# --- 5. Main UI ---
def main():
    st.title("🩻 TB X-Ray Ensemble Diagnostic System")
    cnn_model, mobile_model, res_model = load_all_models()
    
    uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Uploaded Scan", use_column_width=True)
        
        if st.button(" Analyze X-ray"):
            if cnn_model:
                with st.spinner("Analyzing..."):
                    # Predictions
                    s1 = cnn_model.predict(preprocess_image(image, "custom"))[0][0]
                    s2 = mobile_model.predict(preprocess_image(image, "mobilenet"))[0][0]
                    s3 = res_model.predict(preprocess_image(image, "resnet"))[0][0]
                    
                    results_dict = {"Custom CNN": s1, "MobileNetV2": s2, "ResNet50": s3}
                    avg_score = (s1 + s2 + s3) / 3
                    verdict = "TB POSITIVE" if avg_score > 0.5 else "NORMAL"

                    with col2:
                        st.subheader("Results Dashboard")
                        for name, score in results_dict.items():
                            st.write(f"**{name}:** {score*100:.1f}% confidence")
                        
                        st.divider()
                        if avg_score > 0.5:
                            st.error(f"### {verdict}")
                        else:
                            st.success(f"### {verdict}")

                        # PDF Download
                        pdf_data = create_pdf(results_dict, verdict)
                        st.download_button(
                            label="📥 Download Diagnostic Report",
                            data=pdf_data,
                            file_name="tb_report.pdf",
                            mime="application/pdf"
                        )
            else:
                st.error("Models missing in /models folder.")

if __name__ == "__main__":
    main()