import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

st.title("🌿 Plant Health Classifier (Healthy vs Diseased)")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_health_binary_cnn.h5")

model = load_model()

# Upload image
uploaded_file = st.file_uploader("Upload a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Leaf Image", use_container_width=True)

    img = img.resize((96, 96))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    result = "Healthy 🌱" if prediction[0][0] < 0.5 else "Diseased 🍂"
    st.success(f"Prediction: {result}")
