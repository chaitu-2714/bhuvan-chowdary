import numpy as np
import math
import logging

logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(self):
        pass

    def extract_keypoints(self, results, relative_to_wrist=True, scale_invariant=True):
        """
        Extracts the (x, y, z) coordinates of the 21 hand landmarks.
        By default, normalizes relative to wrist (landmark 0) for translation invariance
        and normalizes by palm scale (distance from landmark 0 to 9) for scaling invariance.
        Returns a flattened 1D numpy array of size 63.
        """
        if not results or not results.multi_hand_landmarks:
            return np.zeros(21 * 3)

        # Focus on the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        
        wrist = hand_landmarks.landmark[0]
        w_x, w_y, w_z = wrist.x, wrist.y, wrist.z
        
        # Calculate scale factor (distance between wrist (0) and middle MCP (9))
        scale = 1.0
        if scale_invariant:
            mcp = hand_landmarks.landmark[9]
            scale = math.sqrt((mcp.x - w_x)**2 + (mcp.y - w_y)**2 + (mcp.z - w_z)**2)
            if scale == 0:
                scale = 1.0
                
        keypoints = []
        for landmark in hand_landmarks.landmark:
            if relative_to_wrist:
                rel_x = (landmark.x - w_x) / scale
                rel_y = (landmark.y - w_y) / scale
                rel_z = (landmark.z - w_z) / scale
            else:
                rel_x = landmark.x
                rel_y = landmark.y
                rel_z = landmark.z
            keypoints.extend([rel_x, rel_y, rel_z])
            
        return np.array(keypoints)

    def calculate_velocity(self, current_kps, prev_kps):
        """
        Calculates the average euclidean movement distance between consecutive frame keypoints.
        Assumes keypoints are size 63 arrays.
        """
        if current_kps is None or prev_kps is None:
            return 0.0
            
        # If either is zero vector (no hand), velocity is not defined or is high transition
        if np.all(current_kps == 0) or np.all(prev_kps == 0):
            return 0.0
            
        # Reshape to (21, 3)
        curr = current_kps.reshape(21, 3)
        prev = prev_kps.reshape(21, 3)
        
        # Compute differences
        diffs = curr - prev
        distances = np.linalg.norm(diffs, axis=1)
        return float(np.mean(distances))

    def check_stability(self, velocity, threshold=0.04):
        """
        Returns True if the movement is below the threshold, meaning the hand is static.
        """
        return velocity < threshold

    def check_quality(self, results, image_width=640, image_height=480):
        """
        Checks if the hand landmarks are high quality.
        Rejects if:
        1. Hand is not detected.
        2. Landmarks are out of image boundaries (partial detection).
        3. Confidence score is below threshold.
        Returns (is_valid, reason).
        """
        if not results or not results.multi_hand_landmarks:
            return False, "No hand detected"
            
        # Get first hand
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Check boundary out of screen
        out_of_bounds = False
        for landmark in hand_landmarks.landmark:
            if landmark.x < -0.05 or landmark.x > 1.05 or landmark.y < -0.05 or landmark.y > 1.05:
                out_of_bounds = True
                break
                
        if out_of_bounds:
            return False, "Hand is partially outside camera frame"
            
        # Check confidence from multi_handedness
        confidence = 1.0
        if results.multi_handedness:
            confidence = results.multi_handedness[0].classification[0].score
            
        if confidence < 0.6:
            return False, f"Low landmark confidence ({confidence:.2f})"
            
        return True, "Good quality"
