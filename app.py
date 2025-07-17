# app.py

import streamlit as st
import numpy as np
import cv2
import joblib
from PIL import Image

# Load the trained model
model = joblib.load("model.pkl")

# Streamlit UI
st.title("🌿 Mango Leaf Classifier")
st.markdown("Upload a leaf image to check if it's a Mango Leaf 🍃")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

def extract_average_rgb(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    avg_color_per_row = np.average(image, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    # BGR to RGB
    b, g, r = avg_color
    return [int(r), int(g), int(b)]

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption='Uploaded Image', use_container_width=True)


    # Extract average RGB
    r, g, b = extract_average_rgb(img)

    st.markdown(f"**Extracted RGB Values:** R = {r}, G = {g}, B = {b}")

    # Predict using model
    prediction = model.predict([[r, g, b]])
    st.success(f"Prediction: **{prediction[0]}**")
