import os
import numpy as np
import json
import time
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DatasetManager:
    def __init__(self, 
                 static_dir=os.path.join("data", "static_alphabet"), 
                 dynamic_dir=os.path.join("data", "dynamic_signs"),
                 legacy_dir=os.path.join("data", "legacy", "MP_Data"),
                 db_manager=None):
        self.static_dir = static_dir
        self.dynamic_dir = dynamic_dir
        self.legacy_dir = legacy_dir
        self.db = db_manager
        
        # Ensure folders exist
        os.makedirs(self.static_dir, exist_ok=True)
        os.makedirs(self.dynamic_dir, exist_ok=True)
        os.makedirs(os.path.join("data", "unknown"), exist_ok=True)

    def save_static_sample(self, class_name, keypoints, metadata):
        """
        Saves a static alphabet hand landmarks feature vector (63,) to disk.
        Saves as filename.npy and filename.json (metadata sidecar).
        """
        try:
            class_name = class_name.upper().strip()
            class_folder = os.path.join(self.static_dir, class_name)
            os.makedirs(class_folder, exist_ok=True)
            
            timestamp_ms = int(time.time() * 1000)
            participant_id = metadata.get("participant_id", "default")
            session_id = metadata.get("session_id", "default")
            
            base_filename = f"{class_name}_{participant_id}_{session_id}_{timestamp_ms}"
            npy_path = os.path.join(class_folder, f"{base_filename}.npy")
            json_path = os.path.join(class_folder, f"{base_filename}.json")
            
            # Save keypoints
            np.save(npy_path, keypoints)
            
            # Save metadata
            full_metadata = {
                "class_name": class_name,
                "gesture_type": "static",
                "participant_id": participant_id,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "hand_count": metadata.get("hand_count", 1),
                "landmark_quality": metadata.get("quality", "Good"),
                "detection_confidence": metadata.get("confidence", 1.0),
                "feature_version": "v1_normalized_wrist_scale"
            }
            with open(json_path, 'w') as f:
                json.dump(full_metadata, f, indent=4)
                
            # Log to SQLite
            if self.db:
                self.db.log_session_sample(
                    participant_id=participant_id,
                    session_id=session_id,
                    class_name=class_name,
                    gesture_type="static",
                    hand_count=metadata.get("hand_count", 1),
                    quality=metadata.get("quality", "Good"),
                    confidence=metadata.get("confidence", 1.0)
                )
                
            logger.info(f"Saved static sample for {class_name} at {npy_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving static sample: {e}")
            return False

    def save_dynamic_sequence(self, class_name, sequence_keypoints, metadata):
        """
        Saves a sequence of 30 hand landmark frames to disk under a subfolder.
        """
        try:
            class_name = class_name.lower().strip()
            class_folder = os.path.join(self.dynamic_dir, class_name)
            os.makedirs(class_folder, exist_ok=True)
            
            participant_id = metadata.get("participant_id", "default")
            session_id = metadata.get("session_id", "default")
            timestamp_ms = int(time.time() * 1000)
            
            sequence_folder_name = f"seq_{participant_id}_{session_id}_{timestamp_ms}"
            sequence_folder = os.path.join(class_folder, sequence_folder_name)
            os.makedirs(sequence_folder, exist_ok=True)
            
            # Save each frame keypoints
            for i, kps in enumerate(sequence_keypoints):
                np.save(os.path.join(sequence_folder, f"{i}.npy"), kps)
                
            # Save sequence metadata
            full_metadata = {
                "class_name": class_name,
                "gesture_type": "dynamic",
                "participant_id": participant_id,
                "session_id": session_id,
                "sequence_length": len(sequence_keypoints),
                "timestamp": datetime.now().isoformat(),
                "hand_count": metadata.get("hand_count", 1),
                "quality": metadata.get("quality", "Good"),
                "confidence": metadata.get("confidence", 1.0),
                "feature_version": "v1_normalized_wrist_scale"
            }
            with open(os.path.join(sequence_folder, "metadata.json"), 'w') as f:
                json.dump(full_metadata, f, indent=4)
                
            # Log to SQLite
            if self.db:
                self.db.log_session_sample(
                    participant_id=participant_id,
                    session_id=session_id,
                    class_name=class_name,
                    gesture_type="dynamic",
                    hand_count=metadata.get("hand_count", 1),
                    quality=metadata.get("quality", "Good"),
                    confidence=metadata.get("confidence", 1.0)
                )
                
            logger.info(f"Saved dynamic sign sequence for {class_name} at {sequence_folder}")
            return True
        except Exception as e:
            logger.error(f"Error saving dynamic sequence: {e}")
            return False

    def get_dataset_statistics(self):
        """
        Tallies files on disk to compute dataset counts.
        """
        stats = {
            "total_classes": 0,
            "static_classes": 0,
            "dynamic_classes": 0,
            "total_samples": 0,
            "static_samples": 0,
            "dynamic_samples": 0,
            "legacy_samples": 0,
            "classes": {}
        }
        
        # 1. Scan Static Dataset
        if os.path.exists(self.static_dir):
            static_folders = [f for f in os.listdir(self.static_dir) if os.path.isdir(os.path.join(self.static_dir, f))]
            stats["static_classes"] = len(static_folders)
            for folder in static_folders:
                path = os.path.join(self.static_dir, folder)
                # Count only .npy files
                npy_count = len([f for f in os.listdir(path) if f.endswith(".npy")])
                stats["static_samples"] += npy_count
                
                # Record class count
                stats["classes"][folder.upper()] = {
                    "type": "Static Alphabet",
                    "samples": npy_count
                }
                
        # 2. Scan Dynamic Dataset
        if os.path.exists(self.dynamic_dir):
            dynamic_folders = [f for f in os.listdir(self.dynamic_dir) if os.path.isdir(os.path.join(self.dynamic_dir, f))]
            stats["dynamic_classes"] = len(dynamic_folders)
            for folder in dynamic_folders:
                path = os.path.join(self.dynamic_dir, folder)
                seq_folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
                stats["dynamic_samples"] += len(seq_folders)
                
                stats["classes"][folder.lower()] = {
                    "type": "Dynamic Sign",
                    "samples": len(seq_folders)
                }

        # 3. Scan Legacy Dataset
        if os.path.exists(self.legacy_dir):
            legacy_folders = [f for f in os.listdir(self.legacy_dir) if os.path.isdir(os.path.join(self.legacy_dir, f))]
            for folder in legacy_folders:
                path = os.path.join(self.legacy_dir, folder)
                seq_folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
                stats["legacy_samples"] += len(seq_folders)
                
                # Check if class is already mapped
                if folder.lower() in stats["classes"]:
                    stats["classes"][folder.lower()]["samples"] += len(seq_folders)
                else:
                    stats["classes"][folder.lower()] = {
                        "type": "Legacy Dynamic",
                        "samples": len(seq_folders)
                    }

        stats["total_classes"] = len(stats["classes"])
        stats["total_samples"] = stats["static_samples"] + stats["dynamic_samples"] + stats["legacy_samples"]
        
        return stats
