import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe components
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def mediapipe_detection(image, model):
    """
    Passes an image through the MediaPipe model and returns the processed results.
    """
    # Convert the BGR image to RGB before processing.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to pass by reference.
    image_rgb.flags.writeable = False 
    results = model.process(image_rgb)
    image_rgb.flags.writeable = True
    
    return image, results

def draw_styled_landmarks(image, results):
    """
    Draws the hand landmarks and connections on the image.
    """
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            
def extract_keypoints(results):
    """
    Extracts the (x, y, z) coordinates of the 21 hand landmarks.
    To ensure translation invariance, coordinates are made relative 
    to the wrist (landmark 0).
    Returns a flattened 1D numpy array of size 63.
    """
    if results.multi_hand_landmarks:
        # For this baseline, we focus on the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Landmark 0 is the wrist
        wrist = hand_landmarks.landmark[0]
        
        keypoints = []
        for landmark in hand_landmarks.landmark:
            # Calculate relative coordinates to the wrist
            rel_x = landmark.x - wrist.x
            rel_y = landmark.y - wrist.y
            rel_z = landmark.z - wrist.z
            keypoints.extend([rel_x, rel_y, rel_z])
            
        return np.array(keypoints)
    else:
        # If no hand is detected, return an array of 63 zeros
        return np.zeros(21 * 3)
