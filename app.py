import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2
import os

# Page configuration
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="centered"
)

st.title("🌿 Plant Disease Detection")
st.markdown("---")

# Configuration
CLASS_NAMES_PATH = "class_names.txt"
MODEL_PATH = "plant_disease_model.onnx"
IMG_SIZE = 224

@st.cache_resource
def load_model():
    try:
        session = ort.InferenceSession(MODEL_PATH)
        st.sidebar.success("✅ Model loaded!")
        return session
    except Exception as e:
        st.sidebar.error(f"❌ Model error: {e}")
        return None

@st.cache_data
def load_classes():
    try:
        with open(CLASS_NAMES_PATH, 'r') as f:
            return [line.strip() for line in f.readlines()]
    except:
        return []

def preprocess_image(image):
    img_array = np.array(image)
    img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    
    img_array = img_array.astype(np.float32)
    img_array = (img_array / 127.5) - 1.0
    return img_array

# Load resources
model = load_model()
class_names = load_classes()

if model and class_names:
    st.success(f"✅ Ready! {len(class_names)} plant classes loaded")
    
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    
    uploaded_file = st.file_uploader("📤 Upload leaf image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
        
        with st.spinner("🔍 Analyzing..."):
            try:
                processed = preprocess_image(image.convert('RGB'))
                input_data = np.expand_dims(processed, axis=0)
                
                predictions = model.run([output_name], {input_name: input_data})[0]
                pred_idx = np.argmax(predictions[0])
                confidence = np.max(predictions[0]) * 100
                predicted_class = class_names[pred_idx]
                
                st.markdown(f"### 🌱 **{predicted_class}**")
                st.markdown(f"### 📊 **Confidence: {confidence:.1f}%**")
                
                if "healthy" in predicted_class.lower():
                    st.success("🎉 Healthy plant detected!")
                else:
                    st.warning("⚠️ Potential disease detected!")
                    
            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")
else:
    st.error("❌ Failed to load model or class names")

st.markdown("---")
st.markdown("*Plant Disease Detection | 89.24% Accuracy*")
