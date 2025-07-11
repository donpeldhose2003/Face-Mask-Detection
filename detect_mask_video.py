import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import pickle
import time
import playsound  # for sound alerts (optional, install via `pip install playsound`)

# Load face detector
face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

# Load trained model and label encoder
model = load_model("model/mask_detector.model")
lb = pickle.loads(open("model/label_encoder.pickle", "rb").read())

# Optional: alert sound path
ALERT_SOUND_PATH = "static/alert_sound.wav"

# Start webcam
cap = cv2.VideoCapture(0)
print("[INFO] Starting video stream...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                          minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        # Predict mask or no-mask
        (mask, withoutMask) = model.predict(face)[0]
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # Alert if no mask
        if label == "No Mask":
            # Optional sound alert (comment this out if not needed)
            if os.path.exists(ALERT_SOUND_PATH):
                playsound.playsound(ALERT_SOUND_PATH, block=False)

        # Display label and bounding box
        label_text = f"{label}: {max(mask, withoutMask) * 100:.2f}%"
        cv2.putText(frame, label_text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Show output
    cv2.imshow("Face Mask Detector", frame)
    key = cv2.waitKey(1) & 0xFF

    # Exit on 'q'
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
