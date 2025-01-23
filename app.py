import streamlit as st
from ultralytics import YOLO
import cv2
import os
import numpy as np
import joblib
from PIL import Image

# Load YOLOv8 and Random Forest Classifier models
yolo_model = YOLO('best1.pt')  # Replace with your YOLOv8 weights file
rfc_model = joblib.load('random_forest_model.pkl')  # Load saved Random Forest model

# Create directories for uploads and results if they don't exist
#os.makedirs('uploads', exist_ok=True)
#os.makedirs('results', exist_ok=True)

# Helper function to process YOLO detections
def detect_objects(image_path):
    """
    Run YOLOv8 detection on the image and return detected object details.
    """
    results = yolo_model(image_path)
    detected_objects = []

    for box in results[0].boxes:
    # Extract bounding box coordinates (x1, y1, x2, y2)
       x1, y1, x2, y2 = box.xyxy[0].tolist()
    
    # Extract confidence score (if available)
    conf = box.conf[0] if hasattr(box, 'conf') else None
    
    # Extract label (class name)
    label = results[0].names[int(box.cls[0])]  # Class label
    
    # Append detected object to the list
    detected_objects.append({
        'bbox': [x1, y1, x2, y2],
        'label': label,
        'confidence': conf
    })
    return detected_objects, results[0].plot()


# Streamlit app
st.title("Space Debris Detection and Classification")

st.write("""
Upload two images, and this application will:
1. Detect objects using the YOLOv8 model.
2. Classify detected objects as *debris* or *not debris* using a Random Forest Classifier.
""")

# Upload image files
uploaded_files = st.file_uploader(
    "Upload two images for analysis", 
    type=["jpg", "jpeg", "png"], 
    accept_multiple_files=True
)

if uploaded_files and len(uploaded_files) == 2:
    # Process each uploaded image
    for idx, uploaded_file in enumerate(uploaded_files):
        # Save uploaded file
        img_path = os.path.join("uploads", uploaded_file.name)
        with open(img_path, "wb") as f:
            f.write(uploaded_file.read())

        # Display uploaded image
        st.image(img_path, caption=f"Uploaded Image {idx + 1}", use_column_width=True)

        # Detect objects using YOLO
        detected_objects, plotted_image = detect_objects(img_path)

        # Display detection results
        st.image(plotted_image, caption=f"YOLO Detection Results for Image {idx + 1}", use_column_width=True)
        
        st.write("Detected Objects:")
        for obj in detected_objects:
            st.write(f"Label: {obj['label']}, Confidence: {obj['confidence']:.2f}")
        
        # Classify objects using Random Forest Classifier
        st.write("Classification Results:")
        
        #for obj in detected_objects:
             #Create feature vector for RFC
              #  feature_vector = np.array([obj['bbox'][0], obj['bbox'][1], obj['bbox'][2], obj['bbox'][3], obj['confidence']])
               # debris_label = rfc_model.predict([feature_vector])[0]  # Predict debris (1) or not (0)
               # st.write(f"Object: {obj['label']}, Is Debris: {'Yes' if debris_label == 1 else 'No'}")
    
        for obj in detected_objects:
    # Debug detected object details
                    st.write("Detected Object:", obj)

    # Validate bbox
    if 'bbox' in obj and len(obj['bbox']) == 4:
        feature_vector = np.array([obj['bbox'][2] - obj['bbox'][0],  # Width
                                   obj['bbox'][3] - obj['bbox'][1]]) # Height

        # Predict debris or not
        debris_label = rfc_model.predict([feature_vector])[0]
        st.write(f"Object: {obj['label']}, Is Debris: {'Yes' if debris_label == 1 else 'No'}")
    else:
        st.error("Bounding box data is missing or invalid for this object.")



