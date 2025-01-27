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
