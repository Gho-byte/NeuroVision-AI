import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# 1. Page Configuration
st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="centered"
)

LABELS = ['pituitary', 'glioma', 'meningioma', 'notumor']
TARGET_SIZE = (224, 224)
MODEL_PATH = "./models/model.keras"

# 3. Load Model with Caching (Prevents reloading on every interaction)
@st.cache_resource
def load_trained_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model from '{model_path}': {e}")
        return None

model = load_trained_model(MODEL_PATH)

# 4. Preprocessing Function matching your training pipeline
def preprocess_image(uploaded_file, target_size=(224, 224)):
    # Read image buffer into a numpy array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    
    # Decode as grayscale (matching your cv.IMREAD_GRAYSCALE)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
        
    # Resize to model input shape
    img_resized = cv2.resize(img, target_size)
    
    # Min-max normalization to [0.0, 1.0] (float32)
    img_norm = cv2.normalize(img_resized, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    # Expand dims: (224, 224) -> (1, 224, 224, 1) for batch inference
    img_tensor = np.expand_dims(img_norm, axis=-1)
    img_batch = np.expand_dims(img_tensor, axis=0)
    
    return img, img_batch

# 5. UI Layout
st.title("🧠 Brain Tumor Classification")
st.write("Upload an MRI scan to generate a model prediction and confidence score.")

uploaded_file = st.file_uploader(
    "Choose an MRI image...", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1], gap="medium")
    
    # Preprocess
    orig_img, processed_tensor = preprocess_image(uploaded_file, TARGET_SIZE)
    
    with col1:
        st.subheader("Uploaded Scan")
        st.image(orig_img, caption="Original Image (Grayscale)", use_container_width=True)

    if model is not None and processed_tensor is not None:
        # Run prediction
        predictions = model.predict(processed_tensor)
        probabilities = predictions[0]  # Shape: (num_classes,)
        
        predicted_idx = int(np.argmax(probabilities))
        predicted_label = LABELS[predicted_idx]
        confidence = float(probabilities[predicted_idx]) * 100.0

        with col2:
            st.subheader("Result")
            st.metric(label="Predicted Class", value=predicted_label.capitalize())
            st.metric(label="Confidence", value=f"{confidence:.2f}%")
            
            # Progress bar for top prediction
            st.progress(float(probabilities[predicted_idx]))
            
        # Display full probability distribution across all classes
        st.divider()
        st.subheader("Class Probabilities")
        chart_data = {label: float(prob) for label, prob in zip(LABELS, probabilities)}
        st.bar_chart(chart_data)