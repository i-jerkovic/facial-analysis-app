
import streamlit as st
import cv2
from deepface import DeepFace
from PIL import Image
import os

# Streamlit app title
st.title("Facial Trait Analysis with DeepFace")

# File uploader for image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded image temporarily
    img_path = "temp_image.jpg"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display the uploaded image
    image = Image.open(img_path)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Analyze traits with DeepFace
    try:
        result = DeepFace.analyze(img_path=img_path, actions=['age', 'gender', 'emotion', 'race'], enforce_detection=False)[0]
        
        # Display results
        st.write("### Analysis Results")
        st.write(f"**Age**: {result['age']}")
        st.write(f"**Gender**: {result['gender']}")
        st.write(f"**Emotion**: {result['dominant_emotion']}")
        st.write(f"**Ethnicity**: {result['dominant_race']}")
    except Exception as e:
        st.error(f"Error analyzing image: {str(e)}")
    
    # Clean up temporary file
    if os.path.exists(img_path):
        os.remove(img_path)
