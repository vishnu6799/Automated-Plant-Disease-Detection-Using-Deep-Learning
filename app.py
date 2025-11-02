import streamlit as st
import tensorflow as tf
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

# Custom CSS for better performance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #f0f8f0;
        border-left: 5px solid #2E8B57;
        margin: 10px 0px;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
CLASS_NAMES_PATH = "class_names.txt"
IMG_SIZE = 224

@st.cache_resource
def load_model():
    """Load model with enhanced error handling"""
    try:
        # Try to load any available model file
        possible_models = [
            'plant_disease_model.keras',
            'plant_disease_model.h5', 
            'best_mobilenetv2_model.h5'
        ]
        
        for model_path in possible_models:
            if os.path.exists(model_path):
                try:
                    model = tf.keras.models.load_model(model_path, compile=False)
                    model.compile(optimizer='adam', 
                                loss='sparse_categorical_crossentropy', 
                                metrics=['accuracy'])
                    st.success(f"✅ Model loaded from {model_path}")
                    return model
                except Exception as e:
                    st.warning(f"⚠️ Failed to load {model_path}: {str(e)[:100]}...")
                    continue
        
        st.error("❌ No model files found or all failed to load")
        return None
        
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)[:100]}...")
        return None

@st.cache_data
def load_class_names():
    """Load class names"""
    try:
        with open(CLASS_NAMES_PATH, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        return class_names
    except:
        return []

def preprocess_image(image):
    """Preprocess image for prediction"""
    # Convert and resize
    img_array = np.array(image)
    img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    
    # Ensure 3 channels
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    
    # MobileNetV2 preprocessing
    img_array = img_array.astype('float32')
    img_array = (img_array / 127.5) - 1.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'class_names' not in st.session_state:
    st.session_state.class_names = []

# Load resources
with st.spinner("🔄 Loading model..."):
    if st.session_state.model is None:
        st.session_state.model = load_model()
    if not st.session_state.class_names:
        st.session_state.class_names = load_class_names()

# App header
st.markdown('<h1 class="main-header">🌿 Plant Disease Detection</h1>', unsafe_allow_html=True)
st.markdown("---")

# Check if resources loaded successfully
if st.session_state.model is None:
    st.error("""
    ❌ **Model not loaded!** Please ensure you have these files in your repository:
    - `plant_disease_model.keras` or `plant_disease_model.h5`
    - `class_names.txt`
    """)
    st.stop()

if not st.session_state.class_names:
    st.error("❌ **Class names not loaded!** Check `class_names.txt`")
    st.stop()

# Main app interface
st.success(f"✅ **Ready!** Loaded {len(st.session_state.class_names)} plant classes")

# File uploader
uploaded_file = st.file_uploader(
    "📤 **Upload a leaf image**", 
    type=["jpg", "jpeg", "png"],
    help="Upload a clear image of a plant leaf"
)

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Process and predict
    with st.spinner("🔍 Analyzing image..."):
        try:
            # Preprocess
            processed_image = preprocess_image(image.convert('RGB'))
            
            # Predict
            predictions = st.session_state.model.predict(processed_image, verbose=0)
            predicted_idx = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100
            predicted_class = st.session_state.class_names[predicted_idx]
            
            # Display results
            st.markdown("### 📊 **Results**")
            
            # Prediction box
            confidence_color = "#2E8B57" if confidence >= 70 else "#FFA500" if confidence >= 50 else "#FF4500"
            
            st.markdown(f"""
            <div class="prediction-box">
                <h3>🌱 {predicted_class}</h3>
                <p style="color: {confidence_color}; font-size: 1.2em; font-weight: bold;">
                    Confidence: {confidence:.1f}%
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Health status
            if "healthy" in predicted_class.lower():
                st.success("🎉 **Healthy plant detected!**")
            else:
                st.warning("⚠️ **Potential disease detected!** Consult an expert.")
            
            # Show top predictions
            st.markdown("### 🏆 **Top Predictions**")
            top_indices = np.argsort(predictions[0])[-3:][::-1]
            
            for i, idx in enumerate(top_indices):
                prob = predictions[0][idx] * 100
                st.progress(int(prob), text=f"{st.session_state.class_names[idx]}: {prob:.1f}%")
                
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")

else:
    # Instructions when no file uploaded
    st.info("👆 **Upload a leaf image** to get started!")
    
    st.markdown("""
    ### 📋 **How to use:**
    1. Upload a clear image of a plant leaf
    2. Wait for analysis (takes a few seconds)
    3. View the disease prediction and confidence level
    
    ### 🌿 **Supported diseases:**
    - Bacterial spot, Early blight, Late blight
    - Leaf Mold, Septoria leaf spot  
    - Spider mites, Target Spot
    - Yellow Leaf Curl Virus, Mosaic virus
    - Healthy plants
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Plant Disease Detection | Deep Learning | Streamlit"
    "</div>", 
    unsafe_allow_html=True
)
