import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time

# Load the YOLOv8 model
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # Make sure to change this to the path of your custom model

# Function to process the frame
def process_frame(frame, model):
    # Perform detection with YOLOv8
    results = model(frame)
    
    # Process YOLOv8 results
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].astype(int)
            class_id = box.cls[0].astype(int)
            conf = box.conf[0]
            
            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"{model.names[class_id]} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return frame

# Streamlit app
def main():
    st.title("YOLOv8 Object Detection with Webcam")
    
    # Load the model
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Error loading the YOLO model: {e}")
        return

    # Create a placeholder for the state label
    state_label = st.empty()

    # Create placeholders for the start and stop buttons
    start_button = st.empty()
    stop_button = st.empty()

    # Create a placeholder for the video frame
    video_placeholder = st.empty()

    # Initialize session state
    if 'running' not in st.session_state:
        st.session_state.running = False

    # Function to start the webcam
    def start_webcam():
        st.session_state.running = True

    # Function to stop the webcam
    def stop_webcam():
        st.session_state.running = False

    # Display start button when not running
    if not st.session_state.running:
        start_button.button("Start Webcam", on_click=start_webcam)

    # Main loop for webcam processing
    if st.session_state.running:
        # Replace start button with stop button
        stop_button.button("Stop Webcam", on_click=stop_webcam)

        state_label.text("State: Initializing camera...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Error: Could not open camera.")
            st.session_state.running = False
            return

        state_label.text("State: Camera initialized. Starting detection...")
        
        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                st.error("Error: Could not read frame.")
                break
            
            # Process the frame
            processed_frame = process_frame(frame, model)
            
            # Display the processed frame
            video_placeholder.image(processed_frame, channels="BGR", use_column_width=True)
            
            state_label.text("State: Detecting objects...")
            
            # Small delay to reduce CPU usage
            time.sleep(0.01)
        
        cap.release()
        state_label.text("State: Detection stopped.")
    
    # Display "Ready" state when not running
    if not st.session_state.running:
        state_label.text("State: Ready. Click 'Start Webcam' to begin.")

if __name__ == "__main__":
    main()