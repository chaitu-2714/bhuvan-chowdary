import os
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, 
                 static_dir=os.path.join("models", "static"), 
                 dynamic_dir=os.path.join("models", "dynamic"),
                 legacy_dir=os.path.join("models", "legacy"),
                 db_manager=None):
        self.static_dir = static_dir
        self.dynamic_dir = dynamic_dir
        self.legacy_dir = legacy_dir
        self.db = db_manager
        
        # Ensure folders exist
        os.makedirs(self.static_dir, exist_ok=True)
        os.makedirs(self.dynamic_dir, exist_ok=True)
        os.makedirs(self.legacy_dir, exist_ok=True)
        
        self.init_metadata_files()

    def init_metadata_files(self):
        """
        Creates metadata.json in model directories if they don't exist.
        """
        for directory in [self.static_dir, self.dynamic_dir]:
            meta_path = os.path.join(directory, "metadata.json")
            if not os.path.exists(meta_path):
                try:
                    default_meta = {
                        "active_model_file": None,
                        "models": {},
                        "classes": []
                    }
                    with open(meta_path, 'w') as f:
                        json.dump(default_meta, f, indent=4)
                except Exception as e:
                    logger.error(f"Error initializing metadata file in {directory}: {e}")

    def list_available_models(self):
        """
        Scans filesystem and SQLite database to list all models.
        """
        models = []
        
        # 1. Static models
        static_meta_path = os.path.join(self.static_dir, "metadata.json")
        if os.path.exists(static_meta_path):
            try:
                with open(static_meta_path, 'r') as f:
                    meta = json.load(f)
                active_file = meta.get("active_model_file", None)
                for file_name, info in meta.get("models", {}).items():
                    models.append({
                        "name": info.get("name", file_name),
                        "file_name": file_name,
                        "type": "static",
                        "version": info.get("version", "v1.0"),
                        "dataset_version": info.get("dataset_version", "v1.0"),
                        "feature_version": info.get("feature_version", "v1"),
                        "classes": info.get("classes", []),
                        "accuracy": info.get("accuracy", 0.0),
                        "f1_score": info.get("f1_score", 0.0),
                        "status": "Active" if file_name == active_file else "Inactive",
                        "created_date": info.get("created_date", "")
                    })
            except Exception as e:
                logger.error(f"Error reading static models metadata: {e}")
                
        # 2. Dynamic models
        dynamic_meta_path = os.path.join(self.dynamic_dir, "metadata.json")
        if os.path.exists(dynamic_meta_path):
            try:
                with open(dynamic_meta_path, 'r') as f:
                    meta = json.load(f)
                active_file = meta.get("active_model_file", None)
                for file_name, info in meta.get("models", {}).items():
                    models.append({
                        "name": info.get("name", file_name),
                        "file_name": file_name,
                        "type": "dynamic",
                        "version": info.get("version", "v1.0"),
                        "dataset_version": info.get("dataset_version", "v1.0"),
                        "feature_version": info.get("feature_version", "v1"),
                        "classes": info.get("classes", []),
                        "accuracy": info.get("accuracy", 0.0),
                        "f1_score": info.get("f1_score", 0.0),
                        "status": "Active" if file_name == active_file else "Inactive",
                        "created_date": info.get("created_date", "")
                    })
            except Exception as e:
                logger.error(f"Error reading dynamic models metadata: {e}")

        # 3. Legacy baseline model
        legacy_h5 = os.path.join(self.legacy_dir, "action.h5")
        is_legacy_active = False
        if os.path.exists(legacy_h5):
            # Check if active dynamic model file is action.h5, or if dynamic active file is None (meaning fallback)
            try:
                with open(dynamic_meta_path, 'r') as f:
                    meta = json.load(f)
                active_file = meta.get("active_model_file", None)
                if active_file is None or active_file == "action.h5":
                    is_legacy_active = True
            except:
                is_legacy_active = True
                
            models.append({
                "name": "Legacy LSTM (Baseline)",
                "file_name": "action.h5",
                "type": "dynamic",
                "version": "Baseline v1.0",
                "dataset_version": "Legacy v1.0",
                "feature_version": "v1_relative_wrist",
                "classes": ['hello', 'thanks', 'iloveyou', 'thumbs_up', 'thumbs_down', 'peace', 'ok', 'stop', 'point_up'],
                "accuracy": 0.88,
                "f1_score": 0.87,
                "status": "Active" if is_legacy_active else "Inactive",
                "created_date": "Original Baseline"
            })
            
        return models

    def activate_model(self, model_type, file_name):
        """
        Activates a model of a specific type.
        """
        try:
            if model_type == "static":
                meta_path = os.path.join(self.static_dir, "metadata.json")
                model_dir = self.static_dir
            elif model_type == "dynamic":
                meta_path = os.path.join(self.dynamic_dir, "metadata.json")
                model_dir = self.dynamic_dir
            else:
                logger.error(f"Invalid model type: {model_type}")
                return False
                
            # If activating legacy model as dynamic active
            if model_type == "dynamic" and file_name == "action.h5":
                # Ensure action.h5 exists in legacy dir
                src_path = os.path.join(self.legacy_dir, "action.h5")
                if not os.path.exists(src_path):
                    logger.error("Legacy model action.h5 missing.")
                    return False
                    
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                meta["active_model_file"] = "action.h5"
                # Store class labels
                meta["classes"] = ['hello', 'thanks', 'iloveyou', 'thumbs_up', 'thumbs_down', 'peace', 'ok', 'stop', 'point_up']
                with open(meta_path, 'w') as f:
                    json.dump(meta, f, indent=4)
                    
                # Update SQLite status
                if self.db:
                    # Register model in DB if not present
                    models_db = self.db.get_models()
                    matching_db = [m for m in models_db if m["model_name"] == "Legacy LSTM (Baseline)"]
                    if not matching_db:
                        self.db.add_model(
                            model_name="Legacy LSTM (Baseline)",
                            model_type="dynamic",
                            version="Baseline v1.0",
                            dataset_version="Legacy v1.0",
                            feature_version="v1_relative_wrist",
                            classes=meta["classes"],
                            accuracy=0.88,
                            f1_score=0.87,
                            status="Active"
                        )
                    else:
                        self.db.activate_model(matching_db[0]["id"], "dynamic")
                return True
                
            # Verify file exists
            full_path = os.path.join(model_dir, file_name)
            if not os.path.exists(full_path):
                logger.error(f"Model file does not exist: {full_path}")
                return False
                
            # Update metadata.json
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                
            if file_name not in meta.get("models", {}):
                logger.error(f"Model details not found in metadata.json: {file_name}")
                return False
                
            meta["active_model_file"] = file_name
            meta["classes"] = meta["models"][file_name].get("classes", [])
            
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=4)
                
            # Log to SQLite DB and activate there
            if self.db:
                db_models = self.db.get_models()
                db_name = meta["models"][file_name].get("name", file_name)
                matching = [m for m in db_models if m["model_name"] == db_name]
                if matching:
                    self.db.activate_model(matching[0]["id"], model_type)
                else:
                    info = meta["models"][file_name]
                    self.db.add_model(
                        model_name=info.get("name", file_name),
                        model_type=model_type,
                        version=info.get("version", "v1.0"),
                        dataset_version=info.get("dataset_version", "v1.0"),
                        feature_version=info.get("feature_version", "v1"),
                        classes=info.get("classes", []),
                        accuracy=info.get("accuracy", 0.0),
                        f1_score=info.get("f1_score", 0.0),
                        status="Active"
                    )
                    
            logger.info(f"Activated model {file_name} of type {model_type}.")
            return True
        except Exception as e:
            logger.error(f"Error activating model: {e}")
            return False

    def deactivate_model(self, model_type):
        """
        Deactivates active model of type static or dynamic.
        """
        try:
            if model_type == "static":
                meta_path = os.path.join(self.static_dir, "metadata.json")
            elif model_type == "dynamic":
                meta_path = os.path.join(self.dynamic_dir, "metadata.json")
            else:
                return False
                
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                
            meta["active_model_file"] = None
            meta["classes"] = []
            
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=4)
                
            # Deactivate in DB
            if self.db:
                with self.db.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("UPDATE models SET status = 'Inactive' WHERE model_type = ?", (model_type,))
                    conn.commit()
                    
            logger.info(f"Deactivated active model of type {model_type}.")
            return True
        except Exception as e:
            logger.error(f"Error deactivating model: {e}")
            return False
            
    def register_new_model(self, model_type, file_name, name, classes, accuracy, f1_score, dataset_version="v1.0"):
        """
        Registers a newly trained model in metadata.json and the database.
        """
        try:
            if model_type == "static":
                meta_path = os.path.join(self.static_dir, "metadata.json")
            elif model_type == "dynamic":
                meta_path = os.path.join(self.dynamic_dir, "metadata.json")
            else:
                return False
                
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                
            if "models" not in meta:
                meta["models"] = {}
                
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            info = {
                "name": name,
                "version": f"v1.0_{int(time.time())}",
                "dataset_version": dataset_version,
                "feature_version": "v1_normalized_wrist_scale",
                "classes": classes,
                "accuracy": round(float(accuracy), 4),
                "f1_score": round(float(f1_score), 4),
                "created_date": now
            }
            
            meta["models"][file_name] = info
            # Set as active
            meta["active_model_file"] = file_name
            meta["classes"] = classes
            
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=4)
                
            # Write to DB
            if self.db:
                self.db.add_model(
                    model_name=name,
                    model_type=model_type,
                    version=info["version"],
                    dataset_version=dataset_version,
                    feature_version=info["feature_version"],
                    classes=classes,
                    accuracy=info["accuracy"],
                    f1_score=info["f1_score"],
                    status="Active"
                )
                
            logger.info(f"Registered new model: {name}")
            return True
        except Exception as e:
            logger.error(f"Error registering new model: {e}")
            return False
