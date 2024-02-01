import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image

st.title("CNN Model Upload App")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image to be uploaded", type=["jpg", "png"])

# Load pre-trained CNN model
model_path = "cnn_2.h5"
model = tf.keras.models.load_model(model_path)

# Function to preprocess the image
def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((224, 224))  
    img = img.convert('L')
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = np.vstack([img_array])
    return img_array


threshold = 0.5
# Make predictions
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    image_array = preprocess_image(uploaded_file)
    predictions = model.predict(image_array)
    st.write(f"Predictions: {predictions}")
    if predictions [0, 0] > threshold:
        st.write("It's a Normal Condition")
    else:
        st.write("It's Diagnosed to be TBC!")