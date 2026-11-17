import os
import json
import numpy as np
import logging
# Suppress TensorFlow logs to prevent cluttering console
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logger = logging.getLogger(__name__)

class DynamicClassifier:
    def __init__(self, model_dir=os.path.join("models", "dynamic"), legacy_dir=os.path.join("models", "legacy")):
        self.model_dir = model_dir
        self.legacy_dir = legacy_dir
        self.model = None
        self.classes = []
        self.model_path = None
        self.metadata = {}
        self.sequence_length = 30
        self.load_active_model()

    def load_active_model(self):
        """
        Loads the active dynamic model. Looks in:
        1. models/dynamic/metadata.json & dynamic models
        2. Fallback to models/legacy/action.h5
        """
        try:
            from tensorflow.keras.models import load_model
            from tensorflow.keras.layers import LSTM
            
            # Custom subclass to pop time_major for Keras 3 compatibility
            class PatchedLSTM(LSTM):
                def __init__(self, *args, **kwargs):
                    kwargs.pop('time_major', None)
                    super().__init__(*args, **kwargs)
            
            custom_objs = {'LSTM': PatchedLSTM}

            # Check dynamic models folder first
            metadata_path = os.path.join(self.model_dir, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                    
            model_file = self.metadata.get("active_model_file", None)
            
            if model_file:
                full_model_path = os.path.join(self.model_dir, model_file)
                if os.path.exists(full_model_path):
                    self.model = load_model(full_model_path, custom_objects=custom_objs)
                    self.model_path = full_model_path
                    self.classes = self.metadata.get("classes", [])
                    self.sequence_length = self.metadata.get("sequence_length", 30)
                    logger.info(f"Dynamic model loaded successfully: {full_model_path}")
                    return True

            # If no custom dynamic model is configured, search for legacy baseline
            legacy_model_path = os.path.join(self.legacy_dir, "action.h5")
            if os.path.exists(legacy_model_path):
                self.model = load_model(legacy_model_path, custom_objects=custom_objs)
                self.model_path = legacy_model_path
                # Baseline actions
                self.classes = ['hello', 'thanks', 'iloveyou', 'thumbs_up', 'thumbs_down', 'peace', 'ok', 'stop', 'point_up']
                self.sequence_length = 30
                logger.info(f"Legacy baseline dynamic model loaded: {legacy_model_path}")
                return True
                
            # If no model found in models/legacy/, check root action.h5
            root_legacy_path = os.path.join("action.h5")
            if os.path.exists(root_legacy_path):
                self.model = load_model(root_legacy_path, custom_objects=custom_objs)
                self.model_path = root_legacy_path
                self.classes = ['hello', 'thanks', 'iloveyou', 'thumbs_up', 'thumbs_down', 'peace', 'ok', 'stop', 'point_up']
                self.sequence_length = 30
                logger.info(f"Root baseline dynamic model loaded: {root_legacy_path}")
                return True

            logger.warning("No dynamic model found.")
            self.model = None
            self.classes = []
            return False
        except Exception as e:
            logger.error(f"Error loading dynamic model: {e}")
            self.model = None
            self.classes = []
            return False

    def predict(self, sequence):
        """
        Runs prediction over a sequence of keypoint frames (shape: sequence_length, 63).
        Returns (predicted_class, confidence_score) or (None, 0.0).
        """
        if self.model is None or not self.classes:
            return None, 0.0
            
        try:
            if len(sequence) != self.sequence_length:
                return None, 0.0
                
            seq_arr = np.array(sequence)
            
            # Check inputs shape
            expected_shape = self.model.input_shape
            # input_shape is typically (None, sequence_length, 63)
            if seq_arr.shape[0] != self.sequence_length or seq_arr.shape[1] != 63:
                logger.error(f"Input sequence shape mismatch. Expected ({self.sequence_length}, 63), got {seq_arr.shape}")
                return None, 0.0
                
            # Expand dimensions to (1, sequence_length, 63)
            features = np.expand_dims(seq_arr, axis=0)
            
            res = self.model.predict(features, verbose=0)[0]
            max_idx = np.argmax(res)
            predicted_class = self.classes[max_idx]
            confidence = float(res[max_idx])
            
            return predicted_class, confidence
        except Exception as e:
            logger.error(f"Dynamic inference error: {e}")
            return None, 0.0

    def get_info(self):
        if self.model is None:
            return {"status": "No Model Loaded"}
        return {
            "status": "Active",
            "path": self.model_path,
            "classes": self.classes,
            "sequence_length": self.sequence_length,
            "metadata": self.metadata
        }
