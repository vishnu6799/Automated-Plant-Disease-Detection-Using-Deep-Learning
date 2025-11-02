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
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f8f0;
        border-left: 5px solid #2E8B57;
        margin: 10px 0px;
    }
    .confidence-high {
        color: #2E8B57;
        font-weight: bold;
    }
    .confidence-medium {
        color: #FFA500;
        font-weight: bold;
    }
    .confidence-low {
        color: #FF4500;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown('<h1 class="main-header">🌿 Plant Disease Detection</h1>', unsafe_allow_html=True)
st.markdown("---")

# Configuration
CLASS_NAMES_PATH = "class_names.txt"
IMG_SIZE = 224

@st.cache_resource
def load_model():
    """Load model with compatibility handling"""
    try:
        # Try multiple model loading strategies
        model_files = [
            'plant_disease_model.keras',
            'plant_disease_model.h5',
            'best_mobilenetv2_model.h5'
        ]
        
        for model_file in model_files:
            if os.path.exists(model_file):
                try:
                    model = tf.keras.models.load_model(model_file, compile=False)
                    st.sidebar.success(f"✅ Model loaded from {model_file}!")
                    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                    return model
                except Exception as e:
                    st.sidebar.warning(f"⚠️ Failed to load {model_file}: {e}")
                    continue
        
        # If all model files fail, try building from weights
        st.sidebar.info("🔄 Building model from weights...")
        return build_model_from_weights()
        
    except Exception as e:
        st.sidebar.error(f"❌ All model loading methods failed: {e}")
        return None

def build_model_from_weights():
    """Build model architecture and load weights"""
    try:
        # Load class names to determine output units
        with open(CLASS_NAMES_PATH, 'r') as f:
            num_classes = len(f.readlines())
        
        # Build MobileNetV2 architecture
        base_model = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights=None,
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
        
        inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
        x = inputs
        
        # Manual preprocessing for MobileNetV2
        x = tf.keras.layers.Lambda(lambda x: (x / 127.5) - 1.0)(x)
        
        x = base_model(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs)
        
        # Try to load weights
        weight_files = [
            'plant_disease_model_weights.weights.h5',
            'model_weights.h5'
        ]
        
        for weight_file in weight_files:
            if os.path.exists(weight_file):
                model.load_weights(weight_file)
                st.sidebar.success(f"✅ Weights loaded from {weight_file}!")
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                return model
        
        st.sidebar.warning("⚠️ No weight files found, using untrained model")
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
        
    except Exception as e:
        st.sidebar.error(f"❌ Failed to build model: {e}")
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
    """Preprocess image for MobileNetV2"""
    # Convert to array and resize
    img_array = np.array(image)
    img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    
    # Ensure 3 channels
    if len(img_array.shape) == 2:  # Grayscale
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = img_array[:, :, :3]
    
    # Manual MobileNetV2 preprocessing
    img_array = img_array.astype('float32')
    img_array = (img_array / 127.5) - 1.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Load model and class names
model = load_model()
class_names = load_class_names()

# Display app status
if model is None:
    st.error("❌ Model could not be loaded. Please check your model files.")
    st.stop()

if not class_names:
    st.error("❌ Class names could not be loaded. Please check class_names.txt.")
    st.stop()

st.sidebar.success(f"✅ Ready! {len(class_names)} classes loaded")

# Sidebar
st.sidebar.title("ℹ️ About")
st.sidebar.info("""
Plant disease detection using deep learning with **89.24% accuracy**.
Upload a leaf image for instant diagnosis.
""")

st.sidebar.markdown("---")
st.sidebar.title("📋 Instructions")
st.sidebar.markdown("""
1. Upload clear leaf image
2. Wait for analysis
3. View results & confidence
""")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Leaf Image")
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"],
        help="Select a clear image of a plant leaf"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

with col2:
    st.subheader("🔍 Analysis Results")
    
    if uploaded_file is not None:
        try:
            with st.spinner("🔄 Analyzing image..."):
                # Preprocess and predict
                image_rgb = Image.open(uploaded_file).convert('RGB')
                processed_image = preprocess_image(image_rgb)
                
                predictions = model.predict(processed_image, verbose=0)
                predicted_class_idx = np.argmax(predictions[0])
                confidence = np.max(predictions[0]) * 100
                predicted_class = class_names[predicted_class_idx]
                
                # Display results
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                
                if confidence >= 80:
                    confidence_class = "confidence-high"
                elif confidence >= 60:
                    confidence_class = "confidence-medium"
                else:
                    confidence_class = "confidence-low"
                
                st.markdown(f"### 🌱 **Prediction: {predicted_class}**")
                st.markdown(f"### 📊 **Confidence: <span class='{confidence_class}'>{confidence:.2f}%</span>**", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Health status
                if "healthy" in predicted_class.lower():
                    st.success("🎉 **Healthy Plant Detected!**")
                else:
                    st.warning("⚠️ **Potential Disease Detected!**")
                
                # Top predictions
                st.subheader("📈 Top Predictions")
                top_3_indices = np.argsort(predictions[0])[-3:][::-1]
                
                for i, idx in enumerate(top_3_indices):
                    prob = predictions[0][idx] * 100
                    if i == 0:
                        st.write(f"🥇 **{class_names[idx]}**: {prob:.2f}%")
                    elif i == 1:
                        st.write(f"🥈 {class_names[idx]}: {prob:.2f}%")
                    else:
                        st.write(f"🥉 {class_names[idx]}: {prob:.2f}%")
                        
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")
    
    else:
        st.info("👆 Please upload an image to get started")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🌿 Plant Disease Detection | Deep Learning | Streamlit"
    "</div>", 
    unsafe_allow_html=True
)
