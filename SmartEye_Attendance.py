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
    names = []
    for filename in os.listdir(data_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = cv2.imread(os.path.join(data_path, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                faces.append(img)
                names.append(os.path.splitext(filename)[0])
    return faces, names

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

# Train face recognizer and mark attendance
def recognize_face(faces, names):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    labels = np.arange(len(names))
    recognizer.train(faces, labels)

    for i in range(len(names)):
        mark_attendance(names[i])

# Main function
def main():
    data_path = 'data'
    faces, names = load_faces(data_path)
    if not faces:
        print("No face data found. Please add images to the 'data' folder.")
        return

    recognize_face(faces, names)
    print("Attendance marked successfully.")

if __name__ == '__main__':
    main()

# End of Phase 1
# Upcoming: Phase 2 - Real-Time Webcam-Based Attendance System

