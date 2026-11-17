import os
import joblib
import json
import numpy as np
import logging

logger = logging.getLogger(__name__)

class StaticClassifier:
    def __init__(self, model_dir=os.path.join("models", "static")):
        self.model_dir = model_dir
        self.model = None
        self.classes = []
        self.model_path = None
        self.metadata = {}
        self.load_active_model()

    def load_active_model(self):
        """
        Loads the active static model from the static model directory.
        Looks for metadata.json and best_model.pkl.
        """
        try:
            metadata_path = os.path.join(self.model_dir, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                    
            model_file = self.metadata.get("active_model_file") or "best_model.pkl"
            full_model_path = os.path.join(self.model_dir, model_file)
            
            if os.path.exists(full_model_path):
                self.model = joblib.load(full_model_path)
                self.model_path = full_model_path
                self.classes = self.metadata.get("classes", [])
                logger.info(f"Static model loaded successfully: {full_model_path}")
                return True
            else:
                # If no active model, look for any .pkl file in model_dir
                pkl_files = [f for f in os.listdir(self.model_dir) if f.endswith(".pkl")]
                if pkl_files:
                    path = os.path.join(self.model_dir, pkl_files[0])
                    self.model = joblib.load(path)
                    self.model_path = path
                    
                    # Try to load labels.json or classes.json if exists
                    labels_path = os.path.join(self.model_dir, "classes.json")
                    if os.path.exists(labels_path):
                        with open(labels_path, 'r') as f:
                            self.classes = json.load(f)
                    else:
                        self.classes = list(self.model.classes_) if hasattr(self.model, "classes_") else []
                        
                    logger.info(f"Fallback: Loaded static model {path}")
                    return True
                else:
                    logger.warning("No static model found.")
                    self.model = None
                    self.classes = []
                    return False
        except Exception as e:
            logger.error(f"Error loading static model: {e}")
            self.model = None
            self.classes = []
            return False

    def predict(self, keypoints):
        """
        Predicts the class and probability score for a given feature vector (63,).
        Returns (predicted_class, confidence_score) or (None, 0.0) if no model loaded.
        """
        if self.model is None or not self.classes:
            return None, 0.0
            
        try:
            # Reshape keypoints to (1, 63)
            features = np.expand_dims(keypoints, axis=0)
            
            # Check dimensions match
            expected_dim = self.model.n_features_in_ if hasattr(self.model, "n_features_in_") else 63
            if features.shape[1] != expected_dim:
                logger.error(f"Feature dimension mismatch. Model expects {expected_dim}, got {features.shape[1]}")
                return None, 0.0
                
            # If the classifier supports predict_proba, get probabilities
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(features)[0]
                max_idx = np.argmax(probs)
                predicted_class = self.classes[max_idx]
                confidence = float(probs[max_idx])
            else:
                pred = self.model.predict(features)[0]
                predicted_class = str(pred)
                confidence = 1.0
                
            return predicted_class, confidence
        except Exception as e:
            logger.error(f"Static inference error: {e}")
            return None, 0.0
            
    def get_info(self):
        """
        Returns model details.
        """
        if self.model is None:
            return {"status": "No Model Loaded"}
        return {
            "status": "Active",
            "path": self.model_path,
            "classes": self.classes,
            "metadata": self.metadata
        }
