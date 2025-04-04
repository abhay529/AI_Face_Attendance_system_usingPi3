import os
import face_recognition
import pickle

# Path where student images are stored
STUDENT_IMG_PATH = "Student_img/students/"

# Output model file
MODEL_SAVE_PATH = "face_recognition_model.pkl"

# Dictionary to store encodings
known_encodings = []
known_names = []

# Loop through each student folder
for student_name in os.listdir(STUDENT_IMG_PATH):
    student_folder = os.path.join(STUDENT_IMG_PATH, student_name)

    # Skip if not a folder
    if not os.path.isdir(student_folder):
        continue

    print(f"🔄 Processing {student_name}...")

    # Process each image inside the folder
    for image_name in os.listdir(student_folder):
        image_path = os.path.join(student_folder, image_name)

        # Load the image
        image = face_recognition.load_image_file(image_path)

        # Detect and encode face
        face_encodings = face_recognition.face_encodings(image)

        # If at least one face is found
        if len(face_encodings) > 0:
            known_encodings.append(face_encodings[0])  # Only take the first face
            known_names.append(student_name)
        else:
            print(f"⚠️ No face detected in {image_name}")

# Save trained encodings
print("✅ Training complete! Saving model...")
with open(MODEL_SAVE_PATH, "wb") as f:
    pickle.dump({"encodings": known_encodings, "names": known_names}, f)

print(f"🎉 Model saved as {MODEL_SAVE_PATH}")
