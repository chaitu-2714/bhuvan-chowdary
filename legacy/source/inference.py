import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from utils import mediapipe_detection, draw_styled_landmarks, extract_keypoints

# Load the trained model
try:
    model = load_model('action.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Make sure you have run train_model.py first to generate action.h5")
    exit()

# Actions that we try to detect
actions = np.array(['hello', 'thanks', 'iloveyou', 'thumbs_up', 'thumbs_down', 'peace', 'ok', 'stop', 'point_up'])

# Setup variables for prediction
sequence = []
sentence = []
predictions = []
threshold = 0.5 # Confidence threshold

cap = cv2.VideoCapture(0)

# Set mediapipe model 
mp_hands = mp.solutions.hands
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()
        if not ret:
            break

        # Make detections
        image, results = mediapipe_detection(frame, hands)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)
        
        # 1. Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:] # Keep only the last 30 frames
        
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            predicted_action = actions[np.argmax(res)]
            confidence = res[np.argmax(res)]
            
            # 2. Visualization logic
            if confidence > threshold: 
                cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
                cv2.putText(image, f"{predicted_action} ({confidence:.2f})", (3,30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
cap.release()
cv2.destroyAllWindows()
