import cv2
import mediapipe as mp
import numpy as np
import time
import pandas as pd
import os
import face_recognition
from datetime import datetime

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Load Student Images and Encode Faces
STUDENT_IMAGES = {
    "CR7": "Student_img/cr7.png",
    "Elon Musk": "Student_img/elon_musk.png",
    "Maldini": "Student_img/maldini.png",
    "Messi": "Student_img/messi.png",
    "Ajay Dev": "Student_img/Ajay.jpg",
    "Abhay": "Student_img/Abhay.jpg"
}

known_encodings = []
known_names = []

for name, image_path in STUDENT_IMAGES.items():
    img = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(img)
    if encoding:
        known_encodings.append(encoding[0])
        known_names.append(name)

# Attendance Storage
attendance_time = {name: 0 for name in known_names}
start_times = {}

# Required Attendance Time (45 mins = 2700 sec)
REQUIRED_TIME = 20  # Reduced for testing purposes
CLASS_DURATION = 30000  # Reduced for testing purposes
class_start_time = time.time()

# Excel File Setup
excel_file = "Face_detection_Attendance_system.xlsx"
if not os.path.exists(excel_file):
    pd.DataFrame(columns=["Date", "Time", "Student Name", "Status"]).to_excel(excel_file, index=False)

# Initialize Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Starting Attendance System... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame from webcam.")
        break

    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces using MediaPipe
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            # Get bounding box coordinates
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            bbox = (int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h))

            # Crop & Recognize Face
            x, y, w, h = bbox
            if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                continue  # Skip if bounding box is out of range

            face_img = frame[y:y + h, x:x + w]  # Extract face region

            # Ensure the extracted face is valid
            if face_img.size == 0:
                continue  # Skip processing if face_img is empty

            # Convert to RGB for face recognition
            face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            # Encode detected face
            unknown_encoding = face_recognition.face_encodings(face_img_rgb)
            if unknown_encoding:
                matches = face_recognition.compare_faces(known_encodings, unknown_encoding[0])
                if True in matches:
                    match_idx = matches.index(True)
                    name = known_names[match_idx]
                else:
                    name = "Unknown"
            else:
                name = "Unknown"

            # Draw bounding box and name
            cv2.rectangle(frame, bbox, (0, 255, 0), 2)
            cv2.putText(frame, name, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Track Attendance Time
            if name != "Unknown":
                if name not in start_times:
                    start_times[name] = time.time()
                else:
                    attendance_time[name] += time.time() - start_times[name]
                    start_times[name] = time.time()

    # Show Video Feed
    cv2.imshow("Attendance System", frame)

    # Stop after CLASS_DURATION automatically
    if time.time() - class_start_time >= CLASS_DURATION:
        print("Class Ended. Recording Attendance...")
        break

    # Press 'q' to Quit Early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release Camera
cap.release()
cv2.destroyAllWindows()

# Save Attendance to Excel
current_date = datetime.now().strftime("%Y-%m-%d")
current_time = datetime.now().strftime("%H:%M:%S")

attendance_records = []
for name, duration in attendance_time.items():
    status = "Present" if duration >= REQUIRED_TIME else "Absent"
    if name != "Unknown":
        attendance_records.append([current_date, current_time, name, status])

# Load existing data & append new records
df_existing = pd.read_excel(excel_file)
df_new = pd.DataFrame(attendance_records, columns=["Date", "Time", "Student Name", "Status"])
df_combined = pd.concat([df_existing, df_new], ignore_index=True)
df_combined.to_excel(excel_file, index=False)

print("✅ Attendance Recorded Successfully!")