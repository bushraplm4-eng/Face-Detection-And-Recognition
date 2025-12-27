import cv2
import face_recognition
import os
import numpy as np

# --- STEP 1: Load Known Faces ---
known_face_encodings = []
known_face_names = []

# Path to your folder with images of known people
known_faces_dir = "known_faces"

for filename in os.listdir(known_faces_dir):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        # Load image
        img = face_recognition.load_image_file(f"{known_faces_dir}/{filename}")
        # Get encoding (numerical representation of face features)
        encoding = face_recognition.face_encodings(img)[0]
        
        known_face_encodings.append(encoding)
        # Use filename (minus extension) as the person's name
        known_face_names.append(os.path.splitext(filename)[0])

print(f"Learned {len(known_face_names)} faces: {known_face_names}")

# --- STEP 2: Initialize Webcam ---
video_capture = cv2.VideoCapture(0)

while True:
    # Grab a single frame from video
    ret, frame = video_capture.read()
    
    # Convert image from BGR (OpenCV) to RGB (face_recognition)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- STEP 3: Find and Encode Faces in current frame ---
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare current face to known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # --- STEP 4: Draw Results ---
        # Draw box
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        # Draw label
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Face Recognition Internship Task', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()