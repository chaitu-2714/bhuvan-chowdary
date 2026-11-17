import numpy as np
import collections
import logging

logger = logging.getLogger(__name__)

class UnifiedRecognitionEngine:
    def __init__(self, static_clf=None, dynamic_clf=None, feature_extractor=None, db_manager=None):
        self.static_clf = static_clf
        self.dynamic_clf = dynamic_clf
        self.feature_extractor = feature_extractor
        self.db = db_manager
        
        # State & Buffers
        self.sequence_buffer = collections.deque(maxlen=30)
        self.velocity_buffer = collections.deque(maxlen=10)
        self.prediction_history = collections.deque(maxlen=10)
        
        self.current_state = "NO_HAND"  # NO_HAND, TRACKING, CANDIDATE, CONFIRMED, UNKNOWN, UNCERTAIN, ERROR
        self.active_prediction = None
        self.active_confidence = 0.0
        self.confirmed_gesture = None
        
        # Sentence state
        self.sentence = []
        self.last_confirmed_gesture = None
        
        # Configuration
        self.mode = "static"  # static, dynamic, auto
        self.confirm_threshold = 0.70
        self.uncertain_threshold = 0.50
        self.consecutive_frames_req = 6  # stable frames needed to confirm a static character
        self.no_hand_reset_frames = 15  # frames with no hand to reset duplicate suppression
        self.no_hand_counter = 0
        
        # Candidate tracking
        self.candidate_gesture = None
        self.candidate_counter = 0
        self.prev_keypoints = None

    def set_mode(self, mode):
        if mode in ["static", "dynamic", "auto"]:
            self.mode = mode
            self.reset_buffers()
            logger.info(f"Recognition mode changed to {mode}.")

    def reset_buffers(self):
        self.sequence_buffer.clear()
        self.velocity_buffer.clear()
        self.prediction_history.clear()
        self.candidate_gesture = None
        self.candidate_counter = 0
        self.prev_keypoints = None

    def clear_sentence(self):
        self.sentence = []
        self.last_confirmed_gesture = None
        logger.info("Sentence cleared.")

    def add_space(self):
        self.sentence.append(" ")
        self.last_confirmed_gesture = None
        logger.info("Space added to sentence.")

    def backspace(self):
        if self.sentence:
            removed = self.sentence.pop()
            self.last_confirmed_gesture = None
            logger.info(f"Backspace removed: '{removed}'")

    def process_frame(self, results, translation_service=None):
        """
        Processes a single frame's MediaPipe hand landmark results.
        Returns a dictionary representing the state updates.
        """
        if not results or not results.multi_hand_landmarks:
            # Handle No Hand state
            self.no_hand_counter += 1
            if self.no_hand_counter >= self.no_hand_reset_frames:
                self.last_confirmed_gesture = None  # Reset duplicate suppression boundary
                
            self.current_state = "NO_HAND"
            self.active_prediction = None
            self.active_confidence = 0.0
            self.candidate_gesture = None
            self.candidate_counter = 0
            self.sequence_buffer.clear()
            self.prev_keypoints = None
            
            return self.get_state_dict()

        # Hand detected, reset counter
        self.no_hand_counter = 0
        
        # Extract features
        keypoints = self.feature_extractor.extract_keypoints(results)
        
        # Calculate velocity/stability
        velocity = 0.0
        if self.prev_keypoints is not None:
            velocity = self.feature_extractor.calculate_velocity(keypoints, self.prev_keypoints)
            self.velocity_buffer.append(velocity)
        self.prev_keypoints = keypoints
        
        # Update sequence buffer for dynamic sign evaluation
        self.sequence_buffer.append(keypoints)
        
        # Mode auto-detection
        active_mode = self.mode
        if self.mode == "auto":
            avg_vel = np.mean(self.velocity_buffer) if self.velocity_buffer else 0.0
            # If moving fast, evaluate as dynamic sign, else static alphabet
            active_mode = "dynamic" if avg_vel > 0.04 else "static"
            
        # Run classifier
        pred_class = None
        confidence = 0.0
        
        if active_mode == "static":
            if self.static_clf:
                pred_class, confidence = self.static_clf.predict(keypoints)
                self.current_state = "TRACKING"
        else:  # dynamic mode
            if len(self.sequence_buffer) == 30:
                if self.dynamic_clf:
                    pred_class, confidence = self.dynamic_clf.predict(list(self.sequence_buffer))
                    self.current_state = "TRACKING"
            else:
                self.current_state = "TRACKING"
                # Buffering sequence, return early
                return self.get_state_dict()
                
        # Process predictions through thresholds and stability checks
        if pred_class is not None:
            self.active_prediction = pred_class
            self.active_confidence = confidence
            
            # 1. Rejection layer (Unknown / Uncertain)
            if confidence < self.uncertain_threshold:
                self.current_state = "UNKNOWN"
                self.candidate_gesture = None
                self.candidate_counter = 0
            elif confidence < self.confirm_threshold:
                self.current_state = "UNCERTAIN"
                self.candidate_gesture = None
                self.candidate_counter = 0
            else:
                # 2. Stability layer
                self.current_state = "CANDIDATE"
                
                # Check static alphabet stability over consecutive frames
                if active_mode == "static":
                    if pred_class == self.candidate_gesture:
                        self.candidate_counter += 1
                        if self.candidate_counter >= self.consecutive_frames_req:
                            self._confirm_gesture(pred_class, confidence, "static", translation_service)
                    else:
                        self.candidate_gesture = pred_class
                        self.candidate_counter = 1
                else:
                    # For dynamic sign, once a sequence finishes with high confidence, confirm it immediately
                    self._confirm_gesture(pred_class, confidence, "dynamic", translation_service)
                    self.sequence_buffer.clear()  # Clear sequence after match to prevent double-firing
        else:
            self.current_state = "UNKNOWN"
            self.active_prediction = None
            self.active_confidence = 0.0
            
        return self.get_state_dict()

    def _confirm_gesture(self, gesture, confidence, mode, translation_service=None):
        """
        Confirms the candidate gesture. Enforces duplicate suppression.
        """
        self.current_state = "CONFIRMED"
        self.confirmed_gesture = gesture
        
        # Boundary / Duplicate suppression: One Gesture = One Output
        # E.g. holding 'A' does not repeatedly append 'A'
        if gesture != self.last_confirmed_gesture:
            if mode == "static":
                # Convert class name to display representation (A -> "A")
                char = gesture.upper()
                self.sentence.append(char)
                self.last_confirmed_gesture = gesture
                logger.info(f"Gesture CONFIRMED: {gesture} (Static). Added to sentence.")
            else:
                # For dynamic sign, convert to word representation (hello -> "hello")
                word = gesture.lower()
                # If sentence has letters, space separates words
                if self.sentence and self.sentence[-1] != " ":
                    self.sentence.append(" ")
                self.sentence.append(word)
                self.sentence.append(" ")
                self.last_confirmed_gesture = gesture
                logger.info(f"Gesture CONFIRMED: {gesture} (Dynamic). Added to sentence.")
                
            # Log prediction to SQLite database
            if self.db:
                current_full_sentence = self.get_sentence()
                trans_hi, trans_te = "", ""
                if translation_service:
                    try:
                        trans_hi = translation_service.translate(current_full_sentence, "en", "hi")
                        trans_te = translation_service.translate(current_full_sentence, "en", "te")
                    except:
                        pass
                self.db.log_prediction(
                    gesture=gesture,
                    mode=mode,
                    confidence=confidence,
                    sentence=current_full_sentence,
                    trans_hi=trans_hi,
                    trans_te=trans_te
                )
        else:
            # Suppress duplicate
            pass
            
        self.candidate_gesture = None
        self.candidate_counter = 0

    def get_sentence(self):
        """
        Builds the sentence text string from individual letter and word tokens.
        """
        if not self.sentence:
            return ""
            
        # Clean double spaces or leading/trailing spaces
        result = "".join(self.sentence)
        return result.replace("  ", " ").strip()

    def get_state_dict(self):
        return {
            "state": self.current_state,
            "mode": self.mode,
            "prediction": self.active_prediction,
            "confidence": round(self.active_confidence, 2),
            "confirmed_gesture": self.confirmed_gesture,
            "sentence": self.get_sentence(),
            "velocity": round(float(np.mean(self.velocity_buffer)) if self.velocity_buffer else 0.0, 4)
        }
