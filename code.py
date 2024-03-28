import cv2
import os
from keras.models import load_model
import numpy as np

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the pre-trained face recognition model
model = load_model('keras_Model.h5')

# Load the images from the "faces" folder
faces_folder = "faces"
face_images = []
face_names = []
for filename in os.listdir(faces_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img = cv2.imread(os.path.join(faces_folder, filename))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_images.append(gray)
        face_names.append(os.path.splitext(filename)[0])

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face_roi = gray[y:y+h, x:x+w]
        face_roi_resized = cv2.resize(face_roi, (100, 100))
        face_input = np.expand_dims(np.expand_dims(face_roi_resized, axis=0), axis=3) / 255.0
        prediction = model.predict(face_input)
        if prediction[0][0] > 0.3:  # Assuming 0.5 as the threshold for face recognition
            cv2.putText(frame, "Recognized", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow('Face Verification', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
