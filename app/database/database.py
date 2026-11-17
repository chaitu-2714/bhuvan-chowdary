import sqlite3
import os
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path="database.db"):
        self.db_path = db_path
        self.init_db()

    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Access columns by name
        return conn

    def init_db(self):
        """
        Initializes the database schema if tables don't exist.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # History table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        gesture TEXT,
                        mode TEXT,
                        confidence REAL,
                        sentence TEXT,
                        translation_hi TEXT,
                        translation_te TEXT,
                        timestamp TEXT
                    )
                """)
                
                # Dataset sessions/samples table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        participant_id TEXT,
                        session_id TEXT,
                        class_name TEXT,
                        gesture_type TEXT,
                        hand_count INTEGER,
                        quality TEXT,
                        confidence REAL,
                        timestamp TEXT
                    )
                """)
                
                # Model management table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS models (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_name TEXT,
                        model_type TEXT,
                        version TEXT,
                        dataset_version TEXT,
                        feature_version TEXT,
                        classes TEXT,
                        accuracy REAL,
                        f1_score REAL,
                        status TEXT,
                        created_date TEXT
                    )
                """)
                
                conn.commit()
                logger.info("Database initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")

    def log_prediction(self, gesture, mode, confidence, sentence, trans_hi="", trans_te=""):
        """
        Logs a confirmed gesture/sentence state to the history table.
        """
        try:
            now = datetime.now().isoformat()
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO history (gesture, mode, confidence, sentence, translation_hi, translation_te, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (gesture, mode, confidence, sentence, trans_hi, trans_te, now))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error logging prediction: {e}")
            return False

    def log_session_sample(self, participant_id, session_id, class_name, gesture_type, hand_count, quality, confidence):
        """
        Logs a captured and validated landmark sample to the sessions log.
        """
        try:
            now = datetime.now().isoformat()
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO sessions (participant_id, session_id, class_name, gesture_type, hand_count, quality, confidence, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (participant_id, session_id, class_name, gesture_type, hand_count, quality, confidence, now))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error logging session sample: {e}")
            return False

    def add_model(self, model_name, model_type, version, dataset_version, feature_version, classes, accuracy, f1_score, status="Inactive"):
        """
        Adds a trained model metadata to the database models table.
        """
        try:
            now = datetime.now().isoformat()
            classes_str = json.dumps(classes) if isinstance(classes, list) else str(classes)
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # If activating, deactivate others of same type
                if status == "Active":
                    cursor.execute("UPDATE models SET status = 'Inactive' WHERE model_type = ?", (model_type,))
                    
                cursor.execute("""
                    INSERT INTO models (model_name, model_type, version, dataset_version, feature_version, classes, accuracy, f1_score, status, created_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (model_name, model_type, version, dataset_version, feature_version, classes_str, accuracy, f1_score, status, now))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error adding model: {e}")
            return False

    def activate_model(self, model_id, model_type):
        """
        Sets a model status to Active and deactivates other models of the same type.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE models SET status = 'Inactive' WHERE model_type = ?", (model_type,))
                cursor.execute("UPDATE models SET status = 'Active' WHERE id = ?", (model_id,))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error activating model: {e}")
            return False

    def get_history(self, limit=50):
        """
        Retrieves prediction history.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM history ORDER BY id DESC LIMIT ?", (limit,))
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting history: {e}")
            return []

    def get_models(self):
        """
        Retrieves all model records.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM models ORDER BY id DESC")
                rows = cursor.fetchall()
                result = []
                for row in rows:
                    row_dict = dict(row)
                    # Decode classes list
                    try:
                        row_dict["classes"] = json.loads(row_dict["classes"])
                    except:
                        pass
                    result.append(row_dict)
                return result
        except Exception as e:
            logger.error(f"Error getting models: {e}")
            return []

    def get_sessions_summary(self):
        """
        Summarizes session logs.
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT class_name, COUNT(*) as count 
                    FROM sessions 
                    GROUP BY class_name
                """)
                return {row["class_name"]: row["count"] for row in cursor.fetchall()}
        except Exception as e:
            logger.error(f"Error getting sessions summary: {e}")
            return {}

    def clear_history(self):
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM history")
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error clearing history: {e}")
            return False
