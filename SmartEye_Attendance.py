# SmartEye: Attendance and Behavior Detection System
# Phase 1: Static Image-Based Attendance System
# Status: Phase 1 Completed

import cv2
import os
import numpy as np
from datetime import datetime
import csv

# Load face images and names from the 'data' folder
def load_faces(data_path):
    faces = []
    labels = []
    names = []
    label_id = 0

    for filename in os.listdir(data_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = cv2.imread(os.path.join(data_path, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                faces.append(img)
                labels.append(label_id)
                names.append(os.path.splitext(filename)[0])
                label_id += 1

    return faces, labels, names

# Mark attendance in CSV
def mark_attendance(name):
    filename = 'Attendance.csv'
    now = datetime.now()
    time_string = now.strftime('%H:%M:%S')
    date_string = now.strftime('%Y-%m-%d')

    file_exists = os.path.isfile(filename)

    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Name', 'Date', 'Time', 'Status'])
        writer.writerow([name, date_string, time_string, 'Present'])

# Real-time Face Recognition with Attendance
def recognize_face(faces, labels, names):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Use webcam
    cap = cv2.VideoCapture(0)

    # Use video simulation if webcam not available
    # cap = cv2.VideoCapture('sample_video.mp4')

    recognized_names = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_detected = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces_detected:
            face_roi = gray[y:y + h, x:x + w]
            try:
                face_roi_resized = cv2.resize(face_roi, (faces[0].shape[1], faces[0].shape[0]))
            except Exception as e:
                continue

            label, confidence = recognizer.predict(face_roi_resized)

            if confidence < 50:  # Adjust confidence threshold if needed
                name = names[label]
                cv2.putText(frame, f'{name} ({int(confidence)})', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if name not in recognized_names:
                    mark_attendance(name)
                    recognized_names.add(name)
            else:
                cv2.putText(frame, 'Unknown', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Sleeping detection placeholder
        cv2.putText(frame, 'Sleeping Detection: Coming Soon...', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 0), 2)

        cv2.imshow('SmartEye: Real-time Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main function
def main():
    data_path = 'data'
    faces, labels, names = load_faces(data_path)

    if not faces:
        print("No face data found. Please add images to the 'data' folder.")
        return

    print("Training completed. Starting real-time recognition...")
    recognize_face(faces, labels, names)
    print("Session ended. Attendance marked successfully.")

if __name__ == '__main__':
    main()

# End of Phase 1
# Upcoming: Phase 2 - Real-Time Webcam-Based Attendance System

