import numpy as np
import logging
from sklearn.metrics import classification_report, confusion_matrix

logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self):
        pass

    def evaluate_predictions(self, y_true, y_pred, classes):
        """
        Generates precision, recall, F1-score, and confusion matrix from predictions.
        """
        try:
            # Map index back to class name if integer arrays are passed
            # Ensure inputs are clean numpy arrays
            y_t = np.array(y_true)
            y_p = np.array(y_pred)
            
            # Classification report
            report = classification_report(y_t, y_p, target_names=classes, output_dict=True, zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(y_t, y_p, labels=list(range(len(classes))))
            
            return {
                "accuracy": report.get("accuracy", 0.0),
                "macro_avg": report.get("macro avg", {}),
                "weighted_avg": report.get("weighted avg", {}),
                "class_wise": {cls: report[cls] for cls in classes if cls in report},
                "confusion_matrix": cm.tolist()
            }
        except Exception as e:
            logger.error(f"Error in model evaluation: {e}")
            return {}

    def get_dummy_evaluation_report(self, classes):
        """
        Creates a valid empty report when no test set is evaluated.
        """
        return {
            "accuracy": 0.0,
            "macro_avg": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0},
            "weighted_avg": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0},
            "class_wise": {cls: {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0} for cls in classes},
            "confusion_matrix": np.zeros((len(classes), len(classes)), dtype=int).tolist()
        }
