import cv2
import mediapipe as mp
import logging

logger = logging.getLogger(__name__)

class LandmarkDetector:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "MediaPipe has no attribute 'solutions'. "
                "This occurs when running under Python 3.14+ where MediaPipe solutions are unsupported. "
                "Please run using the project's Python 3.12 environment: .\\.venv\\Scripts\\python.exe"
            )
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        try:
            self.hands = self.mp_hands.Hands(
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                max_num_hands=2  # Support dual hand checking
            )
            logger.info("MediaPipe Hands model initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing MediaPipe Hands: {e}")
            self.hands = None

    def detect(self, frame):
        """
        Processes the image frame through MediaPipe and returns (frame, results).
        """
        if self.hands is None:
            return frame, None
            
        try:
            # Convert BGR OpenCV image to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False 
            results = self.hands.process(image_rgb)
            image_rgb.flags.writeable = True
            return frame, results
        except Exception as e:
            logger.error(f"Error during MediaPipe detection: {e}")
            return frame, None

    def draw_landmarks(self, image, results):
        """
        Draws hand landmarks and connections on the image.
        """
        if results and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())

    def count_hands(self, results):
        """
        Returns the number of hands detected.
        """
        if results and results.multi_hand_landmarks:
            return len(results.multi_hand_landmarks)
        return 0

    def close(self):
        """
        Releases the MediaPipe resources.
        """
        if self.hands is not None:
            try:
                self.hands.close()
                logger.info("MediaPipe Hands model closed.")
            except Exception as e:
                logger.error(f"Error closing MediaPipe: {e}")
            finally:
                self.hands = None
