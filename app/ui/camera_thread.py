import time
import logging
import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

logger = logging.getLogger(__name__)

class CameraThread(QThread):
    # Signals to communicate with the main GUI thread
    frame_ready = Signal(QImage)
    status_ready = Signal(dict)

    def __init__(self, camera_manager, landmark_detector, feature_extractor, unified_engine, dataset_manager, translation_service):
        super().__init__()
        self.camera_manager = camera_manager
        self.landmark_detector = landmark_detector
        self.feature_extractor = feature_extractor
        self.unified_engine = unified_engine
        self.dataset_manager = dataset_manager
        self.translation_service = translation_service
        
        self.running = False
        
        # Collection state variables
        self.collection_active = False
        self.target_class = ""
        self.target_type = "static"
        self.target_samples = 20
        self.samples_collected = 0
        self.participant_id = "default"
        self.session_id = "default"
        
        self.dynamic_collection_buffer = []
        self.current_rejection_reason = "No hand detected"

    def set_collection_params(self, active, class_name, gesture_type, target_samples, participant_id, session_id):
        """
        Configures dataset collection parameters from the UI.
        """
        self.collection_active = active
        self.target_class = class_name.strip()
        self.target_type = gesture_type
        self.target_samples = int(target_samples)
        self.participant_id = participant_id
        self.session_id = session_id
        self.samples_collected = 0
        self.dynamic_collection_buffer = []
        logger.info(f"Thread collection config: {self.target_class} ({self.target_type}), active={active}")

    def run(self):
        self.running = True
        
        # Ensure camera is open
        if not self.camera_manager.is_opened():
            self.camera_manager.start()
            
        while self.running:
            success, frame = self.camera_manager.read_frame()
            if not success:
                time.sleep(0.03)  # Wait briefly if frame read fails
                continue
                
            # Copy frame for drawing
            draw_frame = frame.copy()
            
            # 1. MediaPipe Tracking
            draw_frame, results = self.landmark_detector.detect(draw_frame)
            
            # 2. Keypoints extraction & Recognition state updates
            if results and results.multi_hand_landmarks:
                # Assess landmark quality
                is_valid, reason = self.feature_extractor.check_quality(results)
                
                if is_valid:
                    self.current_rejection_reason = "ACCEPTED"
                    
                    # Process frame in unified recognition engine
                    state = self.unified_engine.process_frame(results, self.translation_service)
                    
                    # Dataset collection logic
                    if self.collection_active and self.target_class:
                        keypoints = self.feature_extractor.extract_keypoints(results)
                        velocity = state["velocity"]
                        
                        if self.target_type == "static":
                            # Must be stable for static collections
                            is_stable = self.feature_extractor.check_stability(velocity)
                            if is_stable:
                                save_ok = self.dataset_manager.save_static_sample(
                                    class_name=self.target_class,
                                    keypoints=keypoints,
                                    metadata={
                                        "participant_id": self.participant_id,
                                        "session_id": self.session_id,
                                        "hand_count": 1,
                                        "quality": "Good",
                                        "confidence": results.multi_handedness[0].classification[0].score if results.multi_handedness else 1.0
                                    }
                                )
                                if save_ok:
                                    self.samples_collected += 1
                                    if self.samples_collected >= self.target_samples:
                                        self.collection_active = False
                                    time.sleep(0.4) # Brief pause to allow signer position shift
                            else:
                                self.current_rejection_reason = "Rejected: Hand is unstable"
                                
                        elif self.target_type == "dynamic":
                            # Dynamic sign sequences
                            self.dynamic_collection_buffer.append(keypoints)
                            
                            # Complete sequence
                            if len(self.dynamic_collection_buffer) == 30:
                                save_ok = self.dataset_manager.save_dynamic_sequence(
                                    class_name=self.target_class,
                                    sequence_keypoints=self.dynamic_collection_buffer,
                                    metadata={
                                        "participant_id": self.participant_id,
                                        "session_id": self.session_id,
                                        "hand_count": 1,
                                        "quality": "Good",
                                        "confidence": results.multi_handedness[0].classification[0].score if results.multi_handedness else 1.0
                                    }
                                )
                                self.dynamic_collection_buffer = []
                                if save_ok:
                                    self.samples_collected += 1
                                    if self.samples_collected >= self.target_samples:
                                        self.collection_active = False
                                time.sleep(1.0) # Break between sequences
                else:
                    self.current_rejection_reason = f"Rejected: {reason}"
                    # Update engine with empty results if landmarks are poor
                    self.unified_engine.process_frame(None, self.translation_service)
            else:
                self.current_rejection_reason = "Rejected: No hand detected"
                self.dynamic_collection_buffer = []
                # Update engine with empty frame
                self.unified_engine.process_frame(None, self.translation_service)
                
            # Draw overlay graphics
            self.landmark_detector.draw_landmarks(draw_frame, results)
            
            # Format and convert raw frame to QImage
            rgb_image = cv2.cvtColor(draw_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Emit signals
            self.frame_ready.emit(q_img.copy())  # Send image clone
            
            # Construct status dict
            engine_state = self.unified_engine.get_state_dict()
            status_dict = {
                "state": engine_state["state"],
                "mode": engine_state["mode"],
                "prediction": engine_state["prediction"],
                "confidence": engine_state["confidence"],
                "confirmed_gesture": engine_state["confirmed_gesture"],
                "sentence": engine_state["sentence"],
                "velocity": engine_state["velocity"],
                "rejection_reason": self.current_rejection_reason,
                "hand_count": 1 if engine_state["state"] != "NO_HAND" else 0,
                "fps": round(self.camera_manager.get_fps(), 1),
                "collection": {
                    "active": self.collection_active,
                    "target_class": self.target_class,
                    "gesture_type": self.target_type,
                    "target_samples": self.target_samples,
                    "collected": self.samples_collected,
                    "remaining": max(0, self.target_samples - self.samples_collected),
                    "progress": round((self.samples_collected / self.target_samples) * 100, 1) if self.target_samples > 0 else 0
                }
            }
            self.status_ready.emit(status_dict)
            
            time.sleep(0.01)  # Limit CPU load

    def stop(self):
        self.running = False
        self.wait()  # Block until loop breaks and thread exits
        self.camera_manager.stop()
