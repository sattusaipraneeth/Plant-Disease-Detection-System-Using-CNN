import streamlit as st
import numpy as np
import pickle
from keras.preprocessing import image
import os
from PIL import Image

# Set page config
st.set_page_config(page_title="🌱 Plant Disease Detection", page_icon="🌱", layout="wide")

# --- Sidebar Section ---
st.sidebar.title("Plant Disease Detection 🌿")
st.sidebar.info(
    """
    **Welcome to the Plant Disease Detection App!**  
    - 📂 Upload an image of a plant leaf  
    - 🧪 Our AI model will predict the disease  
    - 🌱 Get instant results  
    """
)

# --- Load the Model ---
model_path = "D:\My Work\Skills4Future AICTE Internship\Plant Disease Detection\Model\plant_disease_model.pkl"
with open(model_path, 'rb') as file:
    model = pickle.load(file)
st.sidebar.success("✅ Model Loaded Successfully")

# --- Labels ---
labels = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# --- Prediction Function ---
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    preds = model.predict(x)
    predicted_class = labels[np.argmax(preds)]
    return predicted_class.split('___')

# --- Main Section ---
st.title("🌱 Plant Disease Detection")
st.markdown("Upload an image of a plant leaf to detect the disease.")

# --- File Uploader ---
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

# --- Process Uploaded File ---
if uploaded_file is not None:
    # Show the uploaded image
    st.image(uploaded_file, caption="📸 Uploaded Image", use_column_width=True)

    # Ensure the "temp" directory exists
    os.makedirs("temp", exist_ok=True)

    # Save the uploaded file to the "temp" directory
    temp_file_path = os.path.join("temp", uploaded_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Add spinner and progress bar
    with st.spinner("🔄 Classifying... Please wait!"):
        progress_bar = st.progress(0)

        # Update the progress dynamically
        for percent in range(0, 100, 20):
            progress_bar.progress(percent + 20)

        # Predict the disease
        class_name = model_predict(temp_file_path, model)
        crop, disease = class_name

    # Display results
    st.success(f"✅ **Predicted Crop:** {crop}")
    st.warning(f"⚠️ **Predicted Disease:** {disease.replace('_', ' ')}")

    # Add info box
    st.info(
        f"""
        - 🌾 **Crop:** {crop}  
        - 🦠 **Disease:** {disease.replace('_', ' ')}  
        """
    )

    # Clean up temporary file
    os.remove(temp_file_path)
else:
    st.warning("📂 Please upload an image file.")

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.caption("🌟 Created by Saipraneeth S - AI for Agriculture")
