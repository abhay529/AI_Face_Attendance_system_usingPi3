import os
import cv2
import time
import pickle
import mediapipe as mp
import face_recognition
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, request, render_template, redirect, url_for
from openpyxl import load_workbook

app = Flask(__name__)

# Paths
STUDENT_DIR = r"Student_img"
MODEL_PATH = "face_recognition_model.pkl"
EXCEL_FILE = r"Face_detection_Attendance_system.xlsx"

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Load or Train Model
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        known_encodings, known_names = pickle.load(f)
else:
    known_encodings, known_names = [], []

# Attendance Tracking
attendance_time = {name: 0 for name in set(known_names)}
active_students = {}

# Class Parameters
CLASS_DURATION = 60  # 1-hour class (60 seconds for testing)
REQUIRED_TIME = 0.20 * CLASS_DURATION  # 20% of total class time (12 seconds for testing)

# Ensure Excel file exists
if not os.path.exists(EXCEL_FILE):
    pd.DataFrame(columns=["Date", "Student Name", "Total Time", "Status"]).to_excel(EXCEL_FILE, index=False)


def capture_images(student_name, num_images=100):
    """Capture and save images for a new student."""
    student_path = os.path.join(STUDENT_DIR, student_name)
    os.makedirs(student_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    print(f"Capturing {num_images} images for {student_name}...")
    img_count = 0
    while img_count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not capture frame.")
            break

        img_path = os.path.join(student_path, f"{img_count}.jpg")
        cv2.imwrite(img_path, frame)
        img_count += 1
        print(f"Saved: {img_path}")
        time.sleep(0.1)  # Faster image capture

    cap.release()
    print("✅ Image capture complete!")


def train_model():
    """Train and save facial recognition encodings using face_recognition."""
    global known_encodings, known_names
    known_encodings, known_names = [], []

    for student_name in os.listdir(STUDENT_DIR):
        student_path = os.path.join(STUDENT_DIR, student_name)
        if os.path.isdir(student_path):
            for image_name in os.listdir(student_path):
                image_path = os.path.join(student_path, image_name)
                try:
                    img = cv2.imread(image_path)
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    encodings = face_recognition.face_encodings(rgb_img)

                    if encodings:
                        known_encodings.append(encodings[0])
                        known_names.append(student_name)
                        print(f"✅ Encoded: {image_name} for {student_name}")
                except Exception as e:
                    print(f"❌ Error processing {image_path}: {e}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump((known_encodings, known_names), f)

    print("✅ Training complete! Model updated successfully.")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        student_name = request.form['name'].strip()
        if student_name:
            capture_images(student_name)
            train_model()
            return redirect(url_for('home'))
    return render_template('train.html')


@app.route('/start_attendance')
def start_attendance():
    """Real-time attendance tracking"""
    global attendance_time, active_students

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return "Error: Webcam access failed."

    class_start_time = time.time()
    print("📌 Starting Attendance System...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Failed to capture frame from webcam.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)
        detected_names = set()

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                bbox = (int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h))

                x, y, w, h = bbox
                if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                    continue

                face_img = frame[y:y + h, x:x + w]
                if face_img.size == 0:
                    continue

                face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                unknown_encoding = face_recognition.face_encodings(face_img_rgb)

                if unknown_encoding:
                    matches = face_recognition.compare_faces(known_encodings, unknown_encoding[0])
                    distances = face_recognition.face_distance(known_encodings, unknown_encoding[0])

                    if True in matches:
                        best_match_idx = np.argmin(distances)
                        name = known_names[best_match_idx]
                    else:
                        name = "Unknown"
                else:
                    name = "Unknown"

                if name != "Unknown":
                    detected_names.add(name)
                    if name not in active_students:
                        active_students[name] = time.time()

                cv2.rectangle(frame, bbox, (0, 255, 0), 2)
                cv2.putText(frame, name, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        for name in list(active_students.keys()):
            if name not in detected_names:
                time_out = time.time()
                time_in = active_students[name]
                attendance_time[name] += time_out - time_in
                del active_students[name]

        if time.time() - class_start_time >= CLASS_DURATION:
            print("⏳ Class Ended. Recording Attendance...")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save attendance to Excel (Fixed)
    if os.path.exists(EXCEL_FILE):
        df_existing = pd.read_excel(EXCEL_FILE, engine="openpyxl")
    else:
        df_existing = pd.DataFrame(columns=["Date", "Student Name", "Total Time", "Status"])

    df_new = pd.DataFrame([
        [datetime.now().strftime("%Y-%m-%d"), name, round(total_time, 2),
         "Present" if total_time >= REQUIRED_TIME else "Absent"]
        for name, total_time in attendance_time.items()
    ], columns=["Date", "Student Name", "Total Time", "Status"])

    with pd.ExcelWriter(EXCEL_FILE, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
        df_new.to_excel(writer, index=False, header=False, startrow=writer.sheets["Sheet1"].max_row)

    print("✅ Attendance Recorded Successfully!")
    return "Attendance recorded successfully!"


if __name__ == '_main_':
    app.run(debug=True)