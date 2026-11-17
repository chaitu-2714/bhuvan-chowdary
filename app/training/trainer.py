import os
import time
import json
import numpy as np
import threading
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logger = logging.getLogger(__name__)

# Global state for tracking background training progress
_training_state = {
    "is_training": False,
    "progress": 0,
    "status": "Idle",
    "metrics": {},
    "model_type": None
}

def get_training_status():
    return _training_state

class ModelTrainer:
    def __init__(self, 
                 static_dir=os.path.join("data", "static_alphabet"),
                 dynamic_dir=os.path.join("data", "dynamic_signs"),
                 legacy_dir=os.path.join("data", "legacy", "MP_Data"),
                 models_static_dir=os.path.join("models", "static"),
                 models_dynamic_dir=os.path.join("models", "dynamic"),
                 model_manager=None):
        self.static_dir = static_dir
        self.dynamic_dir = dynamic_dir
        self.legacy_dir = legacy_dir
        self.models_static_dir = models_static_dir
        self.models_dynamic_dir = models_dynamic_dir
        self.model_manager = model_manager

    def train_static_alphabet_async(self, participant_id="all"):
        """
        Launches static training in a background thread.
        """
        if _training_state["is_training"]:
            logger.warning("Training already in progress.")
            return False
            
        _training_state["is_training"] = True
        _training_state["progress"] = 0
        _training_state["status"] = "Starting..."
        _training_state["metrics"] = {}
        _training_state["model_type"] = "static"
        
        thread = threading.Thread(target=self._train_static_alphabet, args=(participant_id,), daemon=True)
        thread.start()
        return True

    def _train_static_alphabet(self, participant_id="all"):
        try:
            logger.info("Starting background training for static alphabet...")
            _training_state["status"] = "Loading data..."
            _training_state["progress"] = 10
            
            # Load static alphabet features
            classes = [f for f in os.listdir(self.static_dir) if os.path.isdir(os.path.join(self.static_dir, f))]
            classes.sort()
            
            if len(classes) < 2:
                _training_state["status"] = "Error: Need at least 2 classes to train a model."
                _training_state["is_training"] = False
                return
                
            X = []
            y = []
            
            label_map = {cls: idx for idx, cls in enumerate(classes)}
            
            for cls in classes:
                cls_folder = os.path.join(self.static_dir, cls)
                files = [f for f in os.listdir(cls_folder) if f.endswith(".npy")]
                for file in files:
                    # Filter by participant if requested
                    if participant_id != "all":
                        if f"_{participant_id}_" not in file:
                            continue
                            
                    npy_path = os.path.join(cls_folder, file)
                    try:
                        kps = np.load(npy_path)
                        if kps.shape == (63,):
                            X.append(kps)
                            y.append(label_map[cls])
                    except Exception as e:
                        logger.error(f"Error loading npy file {npy_path}: {e}")
                        
            X = np.array(X)
            y = np.array(y)
            
            if len(X) < 10:
                _training_state["status"] = f"Error: Too few samples ({len(X)}) to train."
                _training_state["is_training"] = False
                return
                
            _training_state["status"] = "Splitting dataset..."
            _training_state["progress"] = 30
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 and np.min(np.bincount(y)) > 1 else None)
            
            # Evaluate classifiers
            _training_state["status"] = "Evaluating SVM classifier..."
            _training_state["progress"] = 50
            
            svm_clf = SVC(probability=True, kernel='linear', C=1.0, random_state=42)
            svm_clf.fit(X_train, y_train)
            svm_preds = svm_clf.predict(X_test)
            svm_acc = accuracy_score(y_test, svm_preds)
            
            _training_state["status"] = "Evaluating Random Forest..."
            _training_state["progress"] = 70
            
            rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_clf.fit(X_train, y_train)
            rf_preds = rf_clf.predict(X_test)
            rf_acc = accuracy_score(y_test, rf_preds)
            
            _training_state["status"] = "Evaluating MLP neural network..."
            _training_state["progress"] = 85
            
            mlp_clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
            mlp_clf.fit(X_train, y_train)
            mlp_preds = mlp_clf.predict(X_test)
            mlp_acc = accuracy_score(y_test, mlp_preds)
            
            # Find best
            clfs = [("SVM", svm_clf, svm_acc, svm_preds), 
                    ("RandomForest", rf_clf, rf_acc, rf_preds), 
                    ("MLP", mlp_clf, mlp_acc, mlp_preds)]
            clfs.sort(key=lambda x: x[2], reverse=True)
            best_name, best_model, best_acc, best_preds = clfs[0]
            
            # Calculate metrics
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, best_preds, average='macro', zero_division=0)
            
            _training_state["status"] = f"Saving best model ({best_name})..."
            _training_state["progress"] = 95
            
            import joblib
            model_filename = f"static_alphabet_{best_name.lower()}_model.pkl"
            model_path = os.path.join(self.models_static_dir, model_filename)
            joblib.dump(best_model, model_path)
            
            # Save classes json for convenience
            with open(os.path.join(self.models_static_dir, "classes.json"), 'w') as f:
                json.dump(classes, f)
                
            # Register in ModelManager
            if self.model_manager:
                self.model_manager.register_new_model(
                    model_type="static",
                    file_name=model_filename,
                    name=f"Static Alphabet {best_name} (Acc: {best_acc:.2f})",
                    classes=classes,
                    accuracy=best_acc,
                    f1_score=f1,
                    dataset_version=f"Static_{len(X)}_Samples"
                )
                self.model_manager.activate_model("static", model_filename)
                
            # Store metrics
            _training_state["metrics"] = {
                "model_name": f"{best_name} Classifier",
                "accuracy": round(float(best_acc), 4),
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1_score": round(float(f1), 4),
                "classes": classes,
                "samples_count": len(X),
                "confusion_matrix": self._compute_confusion_matrix(y_test, best_preds, len(classes))
            }
            
            _training_state["status"] = "Training Complete"
            _training_state["progress"] = 100
            logger.info("Static alphabet training completed successfully.")
        except Exception as e:
            logger.error(f"Error in background training: {e}")
            _training_state["status"] = f"Error during training: {e}"
        finally:
            _training_state["is_training"] = False

    def _compute_confusion_matrix(self, y_true, y_pred, num_classes):
        """
        Helper to compute confusion matrix.
        """
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        return cm.tolist()

    def train_dynamic_signs_async(self, epochs=20):
        """
        Triggers training pipeline for dynamic LSTM model.
        """
        if _training_state["is_training"]:
            logger.warning("Training already in progress.")
            return False
            
        _training_state["is_training"] = True
        _training_state["progress"] = 0
        _training_state["status"] = "Starting..."
        _training_state["metrics"] = {}
        _training_state["model_type"] = "dynamic"
        
        thread = threading.Thread(target=self._train_dynamic_signs, args=(epochs,), daemon=True)
        thread.start()
        return True

    def _train_dynamic_signs(self, epochs=20):
        try:
            logger.info("Starting background training for dynamic sign LSTM model...")
            _training_state["status"] = "Loading dynamic sign data..."
            _training_state["progress"] = 10
            
            # Prepare sequences from dynamic signs & legacy dataset (preserve legacy action.h5 but we can train a new version)
            # Scan dynamic_signs directory
            classes = [f for f in os.listdir(self.dynamic_dir) if os.path.isdir(os.path.join(self.dynamic_dir, f))]
            classes.sort()
            
            # Also check if we have dynamic samples under legacy dataset (MP_Data) to include them
            legacy_actions = []
            if os.path.exists(self.legacy_dir):
                legacy_actions = [f for f in os.listdir(self.legacy_dir) if os.path.isdir(os.path.join(self.legacy_dir, f))]
                
            all_classes = list(set(classes + legacy_actions))
            all_classes.sort()
            
            if len(all_classes) < 2:
                _training_state["status"] = "Error: Need at least 2 dynamic sign classes to train."
                _training_state["is_training"] = False
                return
                
            label_map = {cls: idx for idx, cls in enumerate(all_classes)}
            
            sequences = []
            labels = []
            
            # 1. Load legacy data if present
            if os.path.exists(self.legacy_dir):
                for act in legacy_actions:
                    act_path = os.path.join(self.legacy_dir, act)
                    seq_folders = [f for f in os.listdir(act_path) if os.path.isdir(os.path.join(act_path, f))]
                    for seq in seq_folders:
                        window = []
                        valid_seq = True
                        for frame_num in range(30):
                            npy_path = os.path.join(act_path, seq, f"{frame_num}.npy")
                            if os.path.exists(npy_path):
                                window.append(np.load(npy_path))
                            else:
                                valid_seq = False
                                break
                        if valid_seq and len(window) == 30:
                            sequences.append(window)
                            labels.append(label_map[act])

            # 2. Load newly collected dynamic signs data
            for cls in classes:
                cls_path = os.path.join(self.dynamic_dir, cls)
                seq_folders = [f for f in os.listdir(cls_path) if os.path.isdir(os.path.join(cls_path, f))]
                for seq in seq_folders:
                    window = []
                    valid_seq = True
                    for frame_num in range(30):
                        npy_path = os.path.join(cls_path, seq, f"{frame_num}.npy")
                        if os.path.exists(npy_path):
                            window.append(np.load(npy_path))
                        else:
                            valid_seq = False
                            break
                    if valid_seq and len(window) == 30:
                        sequences.append(window)
                        labels.append(label_map[cls])

            X = np.array(sequences)
            
            if len(X) < 10:
                _training_state["status"] = f"Error: Too few dynamic samples ({len(X)}) to train."
                _training_state["is_training"] = False
                return
                
            from tensorflow.keras.utils import to_categorical
            y = to_categorical(labels, num_classes=len(all_classes)).astype(int)
            
            _training_state["status"] = "Splitting and building LSTM network..."
            _training_state["progress"] = 30
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
            
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            
            model = Sequential()
            model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,63)))
            model.add(Dropout(0.2))
            model.add(LSTM(128, return_sequences=True, activation='relu'))
            model.add(Dropout(0.2))
            model.add(LSTM(64, return_sequences=False, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(64, activation='relu'))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(len(all_classes), activation='softmax'))
            
            model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
            
            _training_state["status"] = "Fitting LSTM model..."
            _training_state["progress"] = 40
            
            # Simple custom callback to report progress
            class ProgressCallback:
                def __init__(self, total_epochs):
                    self.total_epochs = total_epochs
                def on_epoch_end(self, epoch, logs=None):
                    prog = 40 + int((epoch + 1) / self.total_epochs * 50)
                    _training_state["progress"] = prog
                    _training_state["status"] = f"Training Epoch {epoch+1}/{self.total_epochs} (Loss: {logs.get('loss', 0):.4f})"
                    
            model.fit(
                X_train, y_train, 
                epochs=epochs, 
                batch_size=16,
                verbose=0,
                callbacks=[ProgressCallback(epochs)]
            )
            
            _training_state["status"] = "Evaluating model..."
            _training_state["progress"] = 90
            
            # Evaluate model
            predictions = model.predict(X_test, verbose=0)
            y_pred_classes = np.argmax(predictions, axis=1)
            y_test_classes = np.argmax(y_test, axis=1)
            
            acc = accuracy_score(y_test_classes, y_pred_classes)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test_classes, y_pred_classes, average='macro', zero_division=0)
            
            _training_state["status"] = "Saving trained LSTM weights..."
            _training_state["progress"] = 95
            
            model_filename = f"dynamic_signs_lstm_{int(time.time())}.h5"
            model_path = os.path.join(self.models_dynamic_dir, model_filename)
            model.save(model_path)
            
            # Register model
            if self.model_manager:
                self.model_manager.register_new_model(
                    model_type="dynamic",
                    file_name=model_filename,
                    name=f"LSTM Dynamic Sign Model (Acc: {acc:.2f})",
                    classes=all_classes,
                    accuracy=acc,
                    f1_score=f1,
                    dataset_version=f"Dynamic_{len(X)}_Sequences"
                )
                self.model_manager.activate_model("dynamic", model_filename)
                
            _training_state["metrics"] = {
                "model_name": "LSTM Neural Network",
                "accuracy": round(float(acc), 4),
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1_score": round(float(f1), 4),
                "classes": all_classes,
                "samples_count": len(X),
                "confusion_matrix": self._compute_confusion_matrix(y_test_classes, y_pred_classes, len(all_classes))
            }
            
            _training_state["status"] = "Training Complete"
            _training_state["progress"] = 100
            logger.info("Dynamic signs training completed successfully.")
        except Exception as e:
            logger.error(f"Error in dynamic background training: {e}")
            _training_state["status"] = f"Error during training: {e}"
        finally:
            _training_state["is_training"] = False
