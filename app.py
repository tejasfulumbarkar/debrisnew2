import streamlit as st
import joblib
from ultralytics import YOLO
import numpy as np
import os 
import cv2
import time

# Load YOLOv8 model


def load_css(file_name:str)->str:
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


st.markdown(
    """
    <style>
        /* Allow interactions inside the iframe */
        iframe { 
            pointer-events: auto !important;  
        }
        
        /* Disable pointer events only for unnecessary UI elements */
        .stApp {
            overflow: hidden !important;
        }
        
        /* If an orb effect is interfering, ensure it allows clicks */
        
    </style>
    """,
    unsafe_allow_html=True
)




yolo_model = YOLO('best2.pt')

# Load Random Forest Classifier
rf_classifier = joblib.load('random_forest_model.pkl')

# Class ID for debris
DEBRIS_CLASS_ID = 1

# Detection and classification function
def classify_and_predict(image_path):
    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        return "Error: Unable to read image.", None

    # Run YOLO detection
    results = yolo_model.predict(source=image, imgsz=640, conf=0.5)

    if not results or not results[0].boxes:
        return "No debris detected in the image.", image

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        size = (x2 - x1) * (y2 - y1)

        if class_id == DEBRIS_CLASS_ID:
            # Draw bounding box for detected debris
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"Debris: {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Classify removal method
            feature = np.array([[size, confidence]])
            predicted_method = rf_classifier.predict(feature)
            method = ['Laser', 'Harpoon', 'Net'][predicted_method[0]]
            cv2.putText(image, f"Method: {method}", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            return f"Debris Detected! Size: {size}, Confidence: {confidence:.2f}, Suggested Method: {method}", image

    return "No debris detected in the image.", image

# Streamlit UI
st.title("🔍 Detecting and Classifying Space Debris with AI 🚀")
st.markdown("**🌍 Using AI to detect, classify, and track space debris for a cleaner orbit. ✨**")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Save uploaded file
    with open("uploaded_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Run classification
    result, annotated_image = classify_and_predict("uploaded_image.jpg")

    # Display results
    if annotated_image is not None:
        st.image(annotated_image, caption="Processed Image", use_column_width=True)
    st.write(result)
