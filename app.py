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

# Custom CSS
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
MODEL_PATH = "plant_disease_model.onnx"
IMG_SIZE = 224

@st.cache_resource
def load_onnx_model():
    """Load ONNX model"""
    try:
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ Model file not found: {MODEL_PATH}")
            return None
        
        # Set providers for compatibility
        providers = ['CPUExecutionProvider']
        session = ort.InferenceSession(MODEL_PATH, providers=providers)
        st.sidebar.success("✅ ONNX Model loaded successfully!")
        return session
    except Exception as e:
        st.sidebar.error(f"❌ Error loading ONNX model: {e}")
        return None

@st.cache_data
def load_class_names():
    """Load class names from file"""
    try:
        with open(CLASS_NAMES_PATH, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        return class_names
    except Exception as e:
        st.error(f"❌ Error loading class names: {e}")
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
    
    # MobileNetV2 preprocessing (scale to [-1, 1])
    img_array = img_array.astype(np.float32)
    img_array = (img_array / 127.5) - 1.0
    
    return img_array

# Load resources
model_session = load_onnx_model()
class_names = load_class_names()

# App header
st.markdown('<h1 class="main-header">🌿 Plant Disease Detection</h1>', unsafe_allow_html=True)
st.markdown("---")

# Check if resources loaded successfully
if model_session is None:
    st.error("""
    ❌ **Model not loaded!** 
    
    Please ensure you have:
    1. Converted your model to ONNX format using the conversion script
    2. Uploaded `plant_disease_model.onnx` to your repository
    """)
    st.stop()

if not class_names:
    st.error("❌ **Class names not loaded!** Check `class_names.txt`")
    st.stop()

st.success(f"✅ **Ready!** Loaded {len(class_names)} plant classes")

# Get model input/output names
input_name = model_session.get_inputs()[0].name
output_name = model_session.get_outputs()[0].name

# Sidebar
st.sidebar.title("ℹ️ About")
st.sidebar.info("""
Plant disease detection using **ONNX Runtime** for fast, compatible inference.
**89.24% accuracy** on validation set.
""")

st.sidebar.markdown("---")
st.sidebar.title("📋 Instructions")
st.sidebar.markdown("""
1. Upload clear leaf image
2. Wait for analysis
3. View results & confidence
""")

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
            
            # Add batch dimension and predict
            input_data = np.expand_dims(processed_image, axis=0)
            
            # Run inference
            predictions = model_session.run([output_name], {input_name: input_data})[0]
            
            # Get results
            predicted_idx = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100
            predicted_class = class_names[predicted_idx]
            
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
                st.progress(int(prob), text=f"{class_names[idx]}: {prob:.1f}%")
                
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")

else:
    # Instructions when no file uploaded
    st.info("👆 **Upload a leaf image** to get started!")
    
    st.markdown("""
    ### 🌿 **Supported diseases:**
    - Tomato: Bacterial spot, Early blight, Late blight
    - Tomato: Leaf Mold, Septoria leaf spot  
    - Tomato: Spider mites, Target Spot
    - Tomato: Yellow Leaf Curl Virus, Mosaic virus
    - Pepper: Bacterial spot, Healthy
    - Potato: Early blight, Late blight, Healthy
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Plant Disease Detection | ONNX Runtime | Streamlit"
    "</div>", 
    unsafe_allow_html=True
)
