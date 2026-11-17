import unittest
import os
import sys
import numpy as np
import tempfile
import json
import shutil
import sqlite3

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import components
from app.features.feature_extractor import FeatureExtractor
from app.recognition.unified_engine.unified_engine import UnifiedRecognitionEngine
from app.translation.translator import TranslationService, OfflineDictionaryTranslator, GoogleTranslateTranslator
from app.speech.text_to_speech import TextToSpeechEngine
from app.database.database import DatabaseManager
from app.models.model_manager import ModelManager

# Mocks for MediaPipe outputs
class MockLandmark:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class MockHandLandmarks:
    def __init__(self, landmarks):
        self.landmark = landmarks

class MockClassification:
    def __init__(self, score):
        self.score = score

class MockClassifications:
    def __init__(self, classifications):
        self.classification = classifications

class MockResults:
    def __init__(self, landmarks_list, handedness_list=None):
        self.multi_hand_landmarks = landmarks_list
        self.multi_handedness = handedness_list

class TestISLSystem(unittest.TestCase):
    def setUp(self):
        self.extractor = FeatureExtractor()
        
        # Setup temporary directories for files
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "test_database.db")
        self.db = DatabaseManager(self.db_path)
        
        self.static_models_dir = os.path.join(self.test_dir, "models_static")
        self.dynamic_models_dir = os.path.join(self.test_dir, "models_dynamic")
        self.legacy_models_dir = os.path.join(self.test_dir, "models_legacy")
        
        self.model_manager = ModelManager(
            static_dir=self.static_models_dir,
            dynamic_dir=self.dynamic_models_dir,
            legacy_dir=self.legacy_models_dir,
            db_manager=self.db
        )

    def tearDown(self):
        # Shutdown TTS if running to release audio threads
        if hasattr(self, 'tts'):
            self.tts.shutdown()
        # Close database references and force garbage collection
        self.db = None
        self.model_manager = None
        import gc
        gc.collect()
        
        # Clean temp directory
        try:
            shutil.rmtree(self.test_dir)
        except PermissionError:
            pass

    def generate_mock_hand(self, wrist_x=0.5, wrist_y=0.5, scale=0.1):
        """
        Generates 21 hand landmarks relative to a wrist position.
        Joint 9 (middle MCP) is placed to establish scale.
        """
        landmarks = []
        # Landmark 0: Wrist
        landmarks.append(MockLandmark(wrist_x, wrist_y, 0.0))
        # Add remaining 20 landmarks
        for i in range(1, 21):
            if i == 9:
                # Joint 9 creates palm length scale
                landmarks.append(MockLandmark(wrist_x, wrist_y + scale, 0.0))
            else:
                # Make offset scale-proportionate
                rel_x = (i * 0.01) * (scale / 0.1)
                rel_y = (i * 0.01) * (scale / 0.1)
                rel_z = 0.05 * (scale / 0.1)
                landmarks.append(MockLandmark(wrist_x + rel_x, wrist_y + rel_y, rel_z))
        return MockHandLandmarks(landmarks)

    def test_feature_normalization(self):
        """
        1. Tests that landmark extraction is wrist-relative (translation invariant) 
        and scaled (scale invariant).
        """
        hand1 = self.generate_mock_hand(wrist_x=0.5, wrist_y=0.5, scale=0.1)
        results1 = MockResults([hand1])
        kps1 = self.extractor.extract_keypoints(results1)
        
        # Test translation: move hand wrist to (0.2, 0.8), keep same scale
        hand2 = self.generate_mock_hand(wrist_x=0.2, wrist_y=0.8, scale=0.1)
        results2 = MockResults([hand2])
        kps2 = self.extractor.extract_keypoints(results2)
        
        # Verify keypoints are exactly identical after translation
        np.testing.assert_array_almost_equal(kps1, kps2, decimal=5)
        
        # Verify wrist (first 3 elements) is normalized to exactly (0,0,0)
        self.assertEqual(kps1[0], 0.0)
        self.assertEqual(kps1[1], 0.0)
        self.assertEqual(kps1[2], 0.0)
        
        # Test scaling: double hand size, shift wrist
        hand3 = self.generate_mock_hand(wrist_x=0.1, wrist_y=0.1, scale=0.2)
        results3 = MockResults([hand3])
        kps3 = self.extractor.extract_keypoints(results3)
        
        # Verify scale invariance
        np.testing.assert_array_almost_equal(kps1, kps3, decimal=5)

    def test_dataset_validation_quality(self):
        """
        2. Tests validation layer checks boundaries and confidence scores.
        """
        # Test out of bounds x > 1.05
        hand_bad = self.generate_mock_hand(wrist_x=1.1, wrist_y=0.5, scale=0.1)
        results_bad = MockResults([hand_bad])
        is_val, reason = self.extractor.check_quality(results_bad)
        self.assertFalse(is_val)
        self.assertIn("outside", reason)
        
        # Test low confidence from handedness
        hand_ok = self.generate_mock_hand(wrist_x=0.5, wrist_y=0.5, scale=0.1)
        mock_class = MockClassification(score=0.45)
        mock_handedness = MockClassifications([mock_class])
        results_low_conf = MockResults([hand_ok], [mock_handedness])
        is_val, reason = self.extractor.check_quality(results_low_conf)
        self.assertFalse(is_val)
        self.assertIn("confidence", reason)

    def test_velocity_and_stability(self):
        """
        3. Checks movement speed and static classification stability.
        """
        kps_static = np.ones(63)
        vel = self.extractor.calculate_velocity(kps_static, kps_static)
        self.assertEqual(vel, 0.0)
        self.assertTrue(self.extractor.check_stability(vel))
        
        # Check motion
        kps_moved = np.ones(63) * 1.5
        vel_moved = self.extractor.calculate_velocity(kps_static, kps_moved)
        self.assertGreater(vel_moved, 0.0)

    def test_database_operations(self):
        """
        4. Verifies database saves prediction history logs.
        """
        # Log prediction
        success = self.db.log_prediction(
            gesture="hello",
            mode="dynamic",
            confidence=0.92,
            sentence="hello",
            trans_hi="नमस्ते",
            trans_te="నమస్తే"
        )
        self.assertTrue(success)
        
        # Retrieve history
        history = self.db.get_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["gesture"], "hello")
        self.assertEqual(history[0]["translation_hi"], "नमस्ते")
        self.assertEqual(history[0]["translation_te"], "నమస్తే")

    def test_translation_fallback(self):
        """
        5. Tests translation provider interface offline fallback.
        """
        translator = TranslationService(provider="offline")
        
        # Telugu direct lookup
        telugu = translator.translate("hello", "en", "te")
        self.assertEqual(telugu, "నమస్తే")
        
        # Hindi direct lookup
        hindi = translator.translate("stop", "en", "hi")
        self.assertEqual(hindi, "रुकें")
        
        # Fallback to source when no dictionary word exists
        fallback = translator.translate("unknown_word_xyz", "en", "hi")
        self.assertEqual(fallback, "unknown_word_xyz")

    def test_model_manager(self):
        """
        6. Verifies model registry active model selection.
        """
        meta_file = os.path.join(self.static_models_dir, "metadata.json")
        self.assertTrue(os.path.exists(meta_file))
        
        # Load and write metadata to simulate registered model
        with open(meta_file, 'r') as f:
            meta = json.load(f)
            
        # Add mock model
        meta["models"]["mock_model.pkl"] = {
            "name": "Mock RF Classifier",
            "version": "v1.0",
            "classes": ["A", "B", "C"],
            "accuracy": 0.95,
            "f1_score": 0.94,
            "created_date": "2026-08-17"
        }
        with open(meta_file, 'w') as f:
            json.dump(meta, f)
            
        # Create empty pkl file to pass existence check
        with open(os.path.join(self.static_models_dir, "mock_model.pkl"), 'wb') as f:
            f.write(b"mock_binary_data")
            
        # Activate model
        success = self.model_manager.activate_model("static", "mock_model.pkl")
        self.assertTrue(success)
        
        # Verify active in registry listing
        available = self.model_manager.list_available_models()
        mock_record = [m for m in available if m["file_name"] == "mock_model.pkl"]
        self.assertEqual(len(mock_record), 1)
        self.assertEqual(mock_record[0]["status"], "Active")

    def test_unified_engine_state_machine(self):
        """
        7. Tests Unified Engine state transitions and duplicate suppression.
        """
        class MockClassifier:
            def __init__(self, pred, conf):
                self.pred = pred
                self.conf = conf
            def predict(self, kps):
                return self.pred, self.conf
                
        # Inject mock static classifier returning 'A' with 95% confidence
        mock_clf = MockClassifier("A", 0.95)
        
        engine = UnifiedRecognitionEngine(
            static_clf=mock_clf,
            feature_extractor=self.extractor,
            db_manager=self.db
        )
        engine.set_mode("static")
        
        # Hand inputs
        hand = self.generate_mock_hand()
        results = MockResults([hand])
        
        # Feed 1 frame => Expect state CANDIDATE, but not CONFIRMED yet
        state_dict = engine.process_frame(results)
        self.assertEqual(state_dict["state"], "CANDIDATE")
        self.assertEqual(state_dict["prediction"], "A")
        self.assertEqual(state_dict["sentence"], "")
        
        # Feed remaining stable frames (req 6 frames to confirm)
        for _ in range(5):
            state_dict = engine.process_frame(results)
            
        # Frame 6 => Expect CONFIRMED state, sentence = "A"
        self.assertEqual(state_dict["state"], "CONFIRMED")
        self.assertEqual(state_dict["sentence"], "A")
        
        # Feed more stable frames for 'A' => Sentence must remain "A" (SUPPRESSED)
        for _ in range(10):
            state_dict = engine.process_frame(results)
        self.assertEqual(state_dict["sentence"], "A")
        
        # Drop hand (no hand detected) for 16 frames to reset duplicate suppression
        for _ in range(16):
            state_dict = engine.process_frame(None)
        self.assertEqual(state_dict["state"], "NO_HAND")
        
        # Feed 'A' again stable frames => Expect sentence = "AA"
        for _ in range(6):
            state_dict = engine.process_frame(results)
        self.assertEqual(state_dict["sentence"], "AA")

    def test_unknown_gesture_rejection(self):
        """
        8. Tests low confidence values are properly rejected.
        """
        class MockUncertainClassifier:
            def predict(self, kps):
                return "A", 0.42  # Below threshold
                
        mock_clf = MockUncertainClassifier()
        engine = UnifiedRecognitionEngine(
            static_clf=mock_clf,
            feature_extractor=self.extractor,
            db_manager=self.db
        )
        
        hand = self.generate_mock_hand()
        results = MockResults([hand])
        
        # Process frame => Must trigger UNKNOWN or UNCERTAIN and not confirm
        state_dict = engine.process_frame(results)
        self.assertEqual(state_dict["state"], "UNKNOWN")
        self.assertNotEqual(state_dict["state"], "CONFIRMED")

    def test_speech_integration(self):
        """
        9. Verifies text-to-speech launches and enqueues text.
        """
        self.tts = TextToSpeechEngine()
        self.tts.speak("test sentence")
        self.assertEqual(self.tts.speech_queue.qsize(), 1)
        self.tts.stop()
        self.assertEqual(self.tts.speech_queue.qsize(), 0)

if __name__ == "__main__":
    unittest.main()
