import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2

st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿")
st.title("🌿 Plant Disease Detection")
st.markdown("---")

# Load model
@st.cache_resource
def load_model():
    try:
        return ort.InferenceSession('plant_disease_model.onnx')
    except:
        return None

# Load class names  
@st.cache_data
def load_classes():
    try:
        with open('class_names.txt', 'r') as f:
            return [line.strip() for line in f.readlines()]
    except:
        return []

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
                # Preprocess
                img_array = np.array(image.convert('RGB'))
                img_array = cv2.resize(img_array, (224, 224))
                img_array = img_array.astype(np.float32)
                img_array = (img_array / 127.5) - 1.0
                input_data = np.expand_dims(img_array, axis=0)
                
                # Predict
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
