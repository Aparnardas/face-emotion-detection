import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time

# Load models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_model = load_model('fer2013_mini_XCEPTION.102-0.66.hdf5', compile=False)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

cap = cv2.VideoCapture(0)

# Variables for timing control
last_detection_time = 0
detection_interval = 5  # seconds
current_emotion = "Neutral 0.0%"  # Default emotion

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    current_time = time.time()
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Only perform emotion detection if 5 seconds have passed since last detection
        if current_time - last_detection_time >= detection_interval:
            face_roi = gray[y:y + h, x:x + w]
            roi = cv2.resize(face_roi, (64, 64))
            roi = roi.astype('float32') / 255.0
            roi = np.expand_dims(roi, axis=-1)
            roi = np.expand_dims(roi, axis=0)
            
            preds = emotion_model.predict(roi)[0]
            emotion_index = np.argmax(preds)
            confidence = preds[emotion_index] * 100
            current_emotion = f"{emotion_labels[emotion_index]} {confidence:.1f}%"
            last_detection_time = current_time
        
        # Display current emotion with confidence percentage
        cv2.putText(frame, current_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Display time until next detection
        time_remaining = max(0, detection_interval - (current_time - last_detection_time))
        timer_text = f"Next detection in: {time_remaining:.1f}s"
        cv2.putText(frame, timer_text, (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
