import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

st.set_page_config(page_title='Fire Detection', page_icon='🔥')
st.title('🔥 Fire Detection App')

# Load model once and cache it
@st.cache_resource
def load_fire_model():
    return load_model('fire_model.h5')

model = load_fire_model()

# Preprocess image for prediction
def preprocess(img):
    img = img.convert('RGB').resize((64,64))
    arr = np.array(img)/255.0
    return np.expand_dims(arr, axis=0)

# Predict function
def predict(img):
    x = preprocess(img)
    prob = float(model.predict(x, verbose=0)[0][0])
    label = '🔥 Fire' if prob > 0.5 else '✅ No Fire'
    confidence = prob if prob > 0.5 else 1 - prob
    return label, confidence

# File uploader
uploaded_file = st.file_uploader('Upload an image', type=['jpg','jpeg','png'])
if uploaded_file:
    img = Image.open(uploaded_file)
    label, confidence = predict(img)
    st.image(img, caption=f'{label} ({confidence:.2f})', use_container_width=True)
    st.write(f"Prediction: **{label}** | Confidence: {confidence:.2f}")
