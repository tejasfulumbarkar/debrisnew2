import streamlit as st
import joblib
from ultralytics import YOLO
import numpy as np
import cv2
import time

# Load YOLOv8 model
yolo_model = YOLO('best2.pt')

# Load Random Forest Classifier
rf_classifier = joblib.load('random_forest_model.pkl')

# Class ID for debris
DEBRIS_CLASS_ID = 1

# Global variables for trajectory prediction
trajectory_state = {
    "previous_bbox": None,
    "previous_time": None
}

# Trajectory prediction function
def predict_trajectory(current_bbox, previous_bbox, time_elapsed):
    if previous_bbox is None or time_elapsed <= 0:
        return None, 0  # Cannot predict without prior data or invalid time

    # Calculate velocity based on the bounding box displacement
    x1, y1, x2, y2 = current_bbox
    prev_x1, prev_y1, prev_x2, prev_y2 = previous_bbox

    # Compute displacement
    dx = (x1 + x2) / 2 - (prev_x1 + prev_x2) / 2
    dy = (y1 + y2) / 2 - (prev_y1 + prev_y2) / 2

    # Velocity (pixels per second)
    velocity_x = dx / time_elapsed
    velocity_y = dy / time_elapsed

    # Predict future position (after 1 second)
    future_x1 = x1 + int(velocity_x)
    future_y1 = y1 + int(velocity_y)
    future_x2 = x2 + int(velocity_x)
    future_y2 = y2 + int(velocity_y)

    predicted_bbox = [future_x1, future_y1, future_x2, future_y2]
    speed = np.sqrt(velocity_x**2 + velocity_y**2)  # Total speed

    return predicted_bbox, speed

# Detection and classification function
def classify_and_predict(image_path):
    global trajectory_state

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

            # Trajectory prediction
            current_bbox = [x1, y1, x2, y2]
            current_time = time.time()

            if trajectory_state["previous_bbox"] is not None and trajectory_state["previous_time"] is not None:
                time_elapsed = current_time - trajectory_state["previous_time"]
                predicted_bbox, speed = predict_trajectory(
                    current_bbox, trajectory_state["previous_bbox"], time_elapsed)

                if predicted_bbox:
                    # Draw predicted trajectory
                    cv2.arrowedLine(image, (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                                    (int((predicted_bbox[0] + predicted_bbox[2]) / 2),
                                     int((predicted_bbox[1] + predicted_bbox[3]) / 2)),
                                    (255, 0, 255), 2, tipLength=0.2)
                    cv2.putText(image, f"Predicted Trajectory",
                                (predicted_bbox[0], predicted_bbox[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                    cv2.putText(image, f"Speed: {speed:.2f} px/s",
                                (predicted_bbox[0], predicted_bbox[1] - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            # Update trajectory state
            trajectory_state["previous_bbox"] = current_bbox
            trajectory_state["previous_time"] = current_time

            return f"Debris Detected! Size: {size}, Confidence: {confidence:.2f}, Suggested Method: {method}", image

    return "No debris detected in the image.", image

# Streamlit UI
st.title("Space Debris Detection ")
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Save uploaded file
    with open("uploaded_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Run classification and prediction
    result, annotated_image = classify_and_predict("uploaded_image.jpg")

    # Display results
    if annotated_image is not None:
        st.image(annotated_image, caption="Processed Image", use_column_width=True)
    st.write(result)
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




# Inject custom CSS for styling
st.markdown(
    """
    <style>
    /* Base style for the heading */
    .custom-heading {
        font-size: 2.75rem; /* Default font size */
        font-weight: bold;
        color: #333;
        margin: 0;
    }

    /* Responsive style for screens smaller than 600px */
    @media (max-width: 600px) {
        .custom-heading {
            font-size: 1rem; /* Smaller font size */
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add a heading with the custom class
st.markdown('<h1 class="custom-heading">Space Debris Detection and Classification</h1>', unsafe_allow_html=True)


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
       



