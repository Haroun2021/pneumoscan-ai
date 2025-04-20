import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image

# --- Page Configuration ---
st.set_page_config(
    page_title="PneumoScan AI",
    page_icon="🧠",
    layout="centered"
)

# --- Load Model ---
@st.cache_resource
def load_model_cached():
    model = load_model("../models/vgg16_pneumoscan.keras")
    return model

model = load_model_cached()

# --- Header ---
st.markdown(
    '''
    <div style="text-align: center;">
        <h1 style="font-size: 2.5rem;">🧠 PneumoScan AI</h1>
        <p style="font-size: 1.1rem; color: #aaa;">
            Accurate Pneumonia Detection from Chest X-rays using Deep Learning.
        </p>
    </div>
    ''',
    unsafe_allow_html=True
)

# --- Sidebar ---
with st.sidebar:
    st.header("🔍 About PneumoScan")
    st.markdown("PneumoScan is an AI-powered tool for detecting pneumonia in chest X-ray scans. Built with VGG16 and deep learning.")
    st.markdown("---")
    st.write("📂 Upload only **chest X-rays** in JPG or PNG format.")
    st.markdown("🧪 Powered by **TensorFlow** + **Streamlit**")
    st.markdown("---")
    st.markdown("👤 Developed by **Haroun Tray**")

# --- Upload Section ---
uploaded_file = st.file_uploader("📤 Upload a chest X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = load_img(uploaded_file, target_size=(224, 224))
    st.image(image, caption="🖼️ Uploaded Chest X-ray", width=300)

    img_array = img_to_array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction_score = model.predict(img_array)[0][0]
    label = "🟢 NORMAL" if prediction_score < 0.5 else "🔴 PNEUMONIA"
    confidence = 1 - prediction_score if prediction_score < 0.5 else prediction_score
    confidence_percent = confidence * 100

    st.markdown("### 🧪 Prediction Result")
    if confidence >= 0.60:
        st.success(f"**Prediction:** {label}")
        st.write(f"**Confidence:** {confidence_percent:.2f}%")
    else:
        st.warning("⚠️ The model is unsure. Please upload a clearer chest X-ray.")
        st.write(f"**Prediction:** {label} (Low confidence)")
        st.write(f"**Confidence:** {confidence_percent:.2f}%")

# --- Footer ---
st.markdown(
    '''
    <hr>
    <div style="text-align: center; font-size: 0.9rem; color: #888;">
        © 2025 PneumoScan | Built with ❤️ by Haroun Tray
    </div>
    ''',
    unsafe_allow_html=True
)