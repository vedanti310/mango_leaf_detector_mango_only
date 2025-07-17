import streamlit as st
from PIL import Image
import numpy as np
import joblib
from streamlit_cropper import st_cropper

# Load the model
model = joblib.load("model.pkl")  # Update with your actual model filename

# Streamlit page setup
st.set_page_config(page_title="🍃 Mango Leaf Detector", layout="centered")

# CSS Styling – Clean and Visible
st.markdown("""
    <style>
    .stApp {
        background-color: #f0fff0;  /* Very light green */
        color: #000000;
    }

    .title {
        font-size: 36px;
        font-weight: bold;
        color: #006400;  /* Deep green */
        text-align: center;
        margin-top: 20px;
    }

    .subtitle {
        font-size: 18px;
        text-align: center;
        color: #333333;
        margin-bottom: 20px;
    }

    .footer {
        text-align: center;
        color: #4d4d4d;
        font-size: 14px;
        margin-top: 50px;
    }

    /* Ensure alert boxes are clearly styled */
    .element-container:has(.stAlert) {
        background-color: #ffffff !important;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 0 6px rgba(0, 0, 0, 0.1);
    }

    .stAlert > div {
        color: black !important;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# Title and subtitle
st.markdown('<div class="title">🍃 Mango Leaf Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload and crop a leaf image to detect if it is a mango leaf!</div>', unsafe_allow_html=True)

# File upload
uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)

    # Cropping UI
    st.markdown("### ✂️ Crop the image (optional):")
    cropped_img = st_cropper(image, box_color='#2e7d32', aspect_ratio=(1, 1))
    st.image(cropped_img, caption="Cropped Image", use_container_width=True)

    # Feature extraction
    resized = cropped_img.resize((100, 100)).convert("RGB")
    pixels = np.array(resized)
    R = np.mean(pixels[:, :, 0])
    G = np.mean(pixels[:, :, 1])
    B = np.mean(pixels[:, :, 2])
    features = np.array([[R, G, B]])

    st.markdown(f"**🎨 RGB Averages:** R = `{R:.2f}`, G = `{G:.2f}`, B = `{B:.2f}`")

    # Predict
    prediction = model.predict(features)[0]

    st.markdown("### 🧪 Prediction Result:")
    if prediction == 1:
        st.success("✅ This is likely a **Mango Leaf** 🍃")
    else:
        st.error("❌ This does **not** appear to be a Mango Leaf 🌿")

# Footer
st.markdown('<div class="footer">Made with 💚 by Vedanti</div>', unsafe_allow_html=True)
