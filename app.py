
import streamlit as st
import cv2
from deepface import DeepFace
from PIL import Image
import os

st.title("Facial Trait Analysis with DeepFace")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img_path = "temp_image.jpg"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    image = Image.open(img_path)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    try:
        result = DeepFace.analyze(img_path=img_path, actions=['age', 'gender'], enforce_detection=False)[0]
        st.write("### Analysis Results")
        st.write(f"**Age**: {result['age']}")
        st.write(f"**Gender**: {result['gender']}")
    except Exception as e:
        st.error(f"Error analyzing image: {str(e)}")
    if os.path.exists(img_path):
        os.remove(img_path)
