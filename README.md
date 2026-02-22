# tbcheck
Publicly available Machine Learning model for checking for presence of Tuberculosis in an X-rays 
written and deployed by [Joseph Marindi](https://www.linkedin.com/in/joseph-marindi-a6b79825/)

# 🩻 TB Screening & Explainable AI (MobileNetV2)

An AI-powered diagnostic assistant designed to screen Chest X-rays for Tuberculosis using a deep learning ensemble approach. This tool utilizes **MobileNetV2** for high-speed classification and **Grad-CAM** (Gradient-weighted Class Activation Mapping) to provide visual interpretability for clinicians.

## 🚀 Features
* **Automated Screening:** Rapidly classifies X-ray images into 'Normal' or 'TB Positive'.
* **Explainable AI (Grad-CAM):** Generates heatmaps highlighting the specific lung regions that influenced the AI's decision.
* **Adjustable Sensitivity:** Sidebar slider allows users to set the confidence threshold (0.1 - 0.9) to balance between precision and recall.
* **Keras 3 Ready:** Optimized for the latest TensorFlow/Keras 3 environments.

## 📂 Project Structure
```text
├── app.py               # Streamlit application logic
├── requirements.txt     # Python dependencies
├── packages.txt         # System-level dependencies for Cloud deployment
└── models/              
    └── mobilenetv2_best.keras # Your trained model file

🛠️ Installation & Local Setup
**1. Clone the repository:**

git clone [https://github.com/yourusername/tb-diagnostic-app.git](https://github.com/yourusername/tb-diagnostic-app.git)
cd tb-diagnostic-app

**2.Install Dependencies:**
Ensure you have Python 3.10+ installed.

 pip install -r requirements.txt

Run the App:

streamlit run app.py

🧠 How to Interpret Results
Red/Orange Zones: These indicate high-intensity regions where the model detected patterns consistent with TB (e.g., opacities, infiltrates, or cavitations).

Blue/Green Zones: These indicate regions with low influence on the model's decision.

Confidence Score: Represents the model's certainty. If the score is higher than your set threshold, the result is flagged as positive.

⚠️ Disclaimer
This tool is for screening assistance only and is not a replacement for professional medical diagnosis. All AI results should be reviewed by a qualified radiologist or pulmonologist.