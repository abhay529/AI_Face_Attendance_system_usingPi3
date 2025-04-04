# ========== PI 3 OPTIMIZATION HEADERS ==========
import os
try:
    os.nice(10)  # Works on Raspberry Pi (Linux)
except AttributeError:
    pass  # Skip if not available (like on Windows)

os.environ['OMP_NUM_THREADS'] = '1'  # Limit NumPy threads
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # Limit math ops threads
import gc
gc.enable()
gc.set_threshold(50, 10, 10)  # Aggressive garbage collection
# ==============================================

import cv2
import time
import pickle
import mediapipe as mp
import face_recognition
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, request, render_template, redirect, url_for, Response, flash, session
import logging
import threading
from queue import Queue
import json

# Suppress TensorFlow and MediaPipe warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger("mediapipe").setLevel(logging.WARNING)

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# ========== PI 3 CAMERA SETTINGS ==========
CAMERA_WIDTH = 640  # Reduced from likely 1280 (50% less pixels)
CAMERA_HEIGHT = 480  # Reduced from likely 720
FRAME_RATE = 15  # Reduced from likely 30
# =========================================

# Paths (unchanged)
STUDENT_DIR = r"Student_img/students"
MODEL_PATH = "face_recognition_model.pkl"
EXCEL_FILE = r"Face_detection_Attendance_system.xlsx"

# MediaPipe with Pi-3 optimized settings
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    min_detection_confidence=0.5,  # Reduced from 0.7
    model_selection=0  # 0=Short-range (faster), 1=Long-range
)

# Global variables (unchanged)
known_encodings = []
known_names = []
attendance_time = {}
active_students = {}
attendance_in_progress = False
attendance_thread = None
global_frame = None
CLASS_DURATION = 60
REQUIRED_TIME = 0.20 * CLASS_DURATION

# Directory setup (unchanged)
os.makedirs(STUDENT_DIR, exist_ok=True)


# Load or Train Model
def load_model():
    global known_encodings, known_names, attendance_time

    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                known_encodings, known_names = pickle.load(f)
            print(f"✅ Model loaded successfully with {len(known_names)} face encodings.")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            known_encodings, known_names = [], []
    else:
        known_encodings, known_names = [], []
        print("ℹ️ No existing model found. Training required.")

    # Initialize attendance tracking
    attendance_time = {name: 0 for name in set(known_names)}


# Ensure Excel file exists
def initialize_excel():
    if not os.path.exists(EXCEL_FILE):
        try:
            pd.DataFrame(columns=["Date", "Student Name", "Total Time", "Status"]).to_excel(EXCEL_FILE, index=False)
            print(f"✅ Created new Excel file: {EXCEL_FILE}")
        except Exception as e:
            print(f"❌ Error creating Excel file: {e}")


# Initialize on startup
load_model()
initialize_excel()


def generate_frames():
    """Generate frames from the global frame for video streaming"""
    global global_frame
    while True:
        if global_frame is not None:
            ret, buffer = cv2.imencode('.jpg', global_frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            # If no frame is available, yield a blank frame
            blank_frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
            cv2.putText(blank_frame, "Waiting for camera...", (150, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            ret, buffer = cv2.imencode('.jpg', blank_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.04)  # ~25 FPS


def capture_images(student_name, num_images=5000):
    """Capture and save images for a new student."""
    global global_frame
    student_path = os.path.join(STUDENT_DIR, student_name)
    os.makedirs(student_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not access the webcam.")
        return False

    print(f"📸 Capturing {num_images} images for {student_name}...")
    img_count = 0
    while img_count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Could not capture frame.")
            break

        global_frame = frame.copy()  # Update global frame for streaming

        # Detect face in the frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)

        # Only save if a face is detected
        if results.detections:
            img_path = os.path.join(student_path, f"{img_count}.jpg")
            try:
                cv2.imwrite(img_path, frame)
                img_count += 1
                print(f"✅ Saved: {img_path} ({img_count}/{num_images})")
                time.sleep(0.2)  # Small delay between captures
            except Exception as e:
                print(f"❌ Error saving image: {e}")
        else:
            print("⚠️ No face detected in frame. Position your face in view.")
            time.sleep(0.5)

        # Display the frame
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                bbox = (int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h))
                cv2.rectangle(frame, bbox, (0, 255, 0), 2)

        # Display counter
        cv2.putText(frame, f"Images: {img_count}/{num_images}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        global_frame = frame.copy()  # Update global frame with annotations

        cv2.waitKey(1)

    cap.release()
    global_frame = None  # Reset global frame
    print("✅ Image capture complete!")
    return img_count > 0


def train_model():
    """Train and save facial recognition encodings using face_recognition."""
    global known_encodings, known_names, attendance_time
    known_encodings, known_names = [], []

    print("🔄 Training model...")
    total_images = 0
    failed_images = 0

    for student_name in os.listdir(STUDENT_DIR):
        student_path = os.path.join(STUDENT_DIR, student_name)
        if os.path.isdir(student_path):
            student_encodings = 0
            for image_name in os.listdir(student_path):
                if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                image_path = os.path.join(student_path, image_name)
                total_images += 1

                try:
                    img = cv2.imread(image_path)
                    if img is None:
                        print(f"❌ Error: Unable to load image {image_path}")
                        failed_images += 1
                        continue

                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    encodings = face_recognition.face_encodings(rgb_img)

                    if encodings:
                        known_encodings.append(encodings[0])
                        known_names.append(student_name)
                        student_encodings += 1
                    else:
                        print(f"⚠️ No face found in {image_path}")
                        failed_images += 1
                except Exception as e:
                    print(f"❌ Error processing {image_path}: {e}")
                    failed_images += 1

            print(f"✅ Encoded {student_encodings} images for {student_name}")

    if known_encodings:
        try:
            with open(MODEL_PATH, "wb") as f:
                pickle.dump((known_encodings, known_names), f)
            print(f"✅ Training complete! Model updated with {len(known_encodings)} encodings.")
            print(f"📊 Stats: {total_images} total images, {failed_images} failed.")
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False
    else:
        print("❌ No valid face encodings found. Training failed.")
        return False

    # Update attendance_time with new students
    attendance_time = {name: 0 for name in set(known_names)}
    return True


def process_attendance_thread():
    """Optimized version for Pi 3"""
    global attendance_time, active_students, attendance_in_progress, global_frame

    try:
        cap = cv2.VideoCapture(0)
        # ===== PI 3 CAMERA CONFIG =====
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, FRAME_RATE)
        # =============================

        if not cap.isOpened():
            print("❌ Error: Could not open webcam.")
            attendance_in_progress = False
            return

        class_start_time = time.time()
        print("📌 Starting Attendance System...")

        # Reset attendance
        for name in attendance_time:
            attendance_time[name] = 0
        active_students = {}

        # ===== PI 3 FRAME PROCESSING OPTIMIZATION =====
        process_every_n_frames = 2  # Process every 2nd frame
        frame_counter = 0
        # =============================================

        while attendance_in_progress:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Failed to capture frame.")
                break

            # Calculate elapsed time
            elapsed_time = time.time() - class_start_time
            remaining_time = max(0, CLASS_DURATION - elapsed_time)

            # ===== SKIP FRAMES =====
            frame_counter += 1
            if frame_counter % process_every_n_frames != 0:
                continue
            # =======================

            # ===== DOWNSAMPLE FRAME =====
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            # ============================

            results = face_detection.process(rgb_frame)
            detected_names = set()

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, _ = small_frame.shape

                    # Scale coordinates back up
                    bbox = (int(bboxC.xmin * w * 2), int(bboxC.ymin * h * 2),
                            int(bboxC.width * w * 2), int(bboxC.height * h * 2))

                    x, y, w, h = bbox
                    if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                        continue

                    face_img = frame[y:y + h, x:x + w]
                    if face_img.size == 0:
                        continue

                    # ===== DOWNSAMPLE FACE =====
                    small_face = cv2.resize(face_img, (0, 0), fx=0.5, fy=0.5)
                    face_img_rgb = cv2.cvtColor(small_face, cv2.COLOR_BGR2RGB)
                    # ===========================

                    unknown_encoding = face_recognition.face_encodings(face_img_rgb)
                    if unknown_encoding:
                        matches = face_recognition.compare_faces(known_encodings, unknown_encoding[0])
                        distances = face_recognition.face_distance(known_encodings, unknown_encoding[0])

                        # With this stricter version:
                        matches = face_recognition.compare_faces(known_encodings, unknown_encoding[0],
                                                                 tolerance=0.5)  # Default is 0.6
                        # In process_attendance_thread(), replace the name matching logic with:
                        distances = face_recognition.face_distance(known_encodings, unknown_encoding[0])
                        best_match_idx = np.argmin(distances)

                        if distances[best_match_idx] < 0.4:  # Stricter threshold (default is 0.6)
                            name = known_names[best_match_idx]
                        else:
                            name = "Unknown"
                            cv2.putText(frame, f"Low Confidence: {distances[best_match_idx]:.2f}",
                                        (bbox[0], bbox[1] - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    if face_img.size < 2500:  # Minimum 50x50 pixels
                        name = "Unknown (too small)"
                        continue

                    if name != "Unknown":
                        detected_names.add(name)
                        if name not in active_students:
                            active_students[name] = time.time()

                    cv2.rectangle(frame, bbox, (0, 255, 0), 2)
                    cv2.putText(frame, name, (bbox[0], bbox[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Update attendance for students who are no longer visible
            for name in list(active_students.keys()):
                if name not in detected_names:
                    time_out = time.time()
                    time_in = active_students[name]
                    attendance_time[name] += time_out - time_in
                    del active_students[name]

            # Display attendance timer on frame
            cv2.putText(frame, f"Time remaining: {int(remaining_time)}s", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Display currently active students
            y_pos = 60
            cv2.putText(frame, "Active Students:", (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            for name in active_students:
                y_pos += 25
                cv2.putText(frame, name, (20, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Update global frame for streaming
            global_frame = frame.copy()

            if elapsed_time >= CLASS_DURATION:
                print("⏳ Class Ended. Recording Attendance...")
                # Capture remaining active students
                for name, time_in in active_students.items():
                    attendance_time[name] += time.time() - time_in
                break

            cv2.waitKey(1)

        cap.release()
        global_frame = None

        # Save attendance to Excel
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            if os.path.exists(EXCEL_FILE):
                df_existing = pd.read_excel(EXCEL_FILE)
            else:
                df_existing = pd.DataFrame(columns=["Date", "Student Name", "Total Time", "Status"])

            new_records = []
            for name, total_time in attendance_time.items():
                if total_time > 0:
                    status = "Present" if total_time >= REQUIRED_TIME else "Absent"
                    new_records.append([today, name, round(total_time, 2), status])

            if new_records:
                df_new = pd.DataFrame(new_records, columns=["Date", "Student Name", "Total Time", "Status"])
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                df_combined.to_excel(EXCEL_FILE, index=False)
                print("✅ Attendance Recorded Successfully!")
            else:
                print("⚠️ No students detected during this session.")

        except Exception as e:
            print(f"❌ Error saving attendance to Excel: {e}")

    except Exception as e:
        print(f"❌ General error in attendance processing: {e}")

    attendance_in_progress = False

@app.route('/')
def home():
    global attendance_in_progress
    status = request.args.get('status', None)
    if status == 'success':
        flash('Attendance recorded successfully!', 'success')
    elif status == 'error':
        flash('Error recording attendance. Please check logs.', 'error')

    return render_template('index.html',
                           attendance_active=attendance_in_progress,
                           student_count=len(set(known_names)))

@app.route('/video_feed')
def video_feed():
    """Route to serve the video stream"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        student_name = request.form['name'].strip()
        if student_name:
            success = capture_images(student_name)
            if success:
                if train_model():
                    flash(f'Successfully trained model for {student_name}!', 'success')
                else:
                    flash('Error training model. Please check logs.', 'error')
            else:
                flash('Error capturing images. Please try again.', 'error')
            return redirect(url_for('home'))
    return render_template('train.html')

@app.route('/start_attendance')
def start_attendance():
    """Start the attendance tracking process"""
    global attendance_in_progress, attendance_thread

    if attendance_in_progress:
        flash('Attendance tracking is already in progress!', 'info')
        return redirect(url_for('home'))

    if not known_names:
        flash('No students in database. Please train the model first.', 'error')
        return redirect(url_for('home'))

    attendance_in_progress = True
    attendance_thread = threading.Thread(target=process_attendance_thread)
    attendance_thread.daemon = True
    attendance_thread.start()

    flash(f'Attendance tracking started for {CLASS_DURATION} seconds!', 'success')
    return redirect(url_for('home'))

@app.route('/stop_attendance')
def stop_attendance():
    """Stop the attendance tracking process"""
    global attendance_in_progress

    if attendance_in_progress:
        attendance_in_progress = False
        flash('Attendance tracking stopped!', 'info')
    else:
        flash('No attendance tracking in progress.', 'info')

    return redirect(url_for('home'))

@app.route('/view_attendance')
def view_attendance():
    """View recorded attendance"""
    try:
        if os.path.exists(EXCEL_FILE):
            df = pd.read_excel(EXCEL_FILE)
            records = df.to_dict('records')
            return render_template('attendance.html', records=records)
        else:
            flash('No attendance records found.', 'info')
            return redirect(url_for('home'))
    except Exception as e:
        flash(f'Error reading attendance records: {str(e)}', 'error')
        return redirect(url_for('home'))

if __name__ == '__main__':
    # ===== PI 3 FLASK OPTIMIZATION =====
    app.run(debug=True, host='0.0.0.0', threaded=True)
    # ===================================