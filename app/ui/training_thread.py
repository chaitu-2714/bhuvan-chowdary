import time
import logging
from PySide6.QtCore import QThread, Signal
from app.training.trainer import get_training_status

logger = logging.getLogger(__name__)

class TrainingThread(QThread):
    # Signals to communicate with the GUI
    progress_changed = Signal(int, str)
    training_finished = Signal(dict)

    def __init__(self, trainer, model_type="static", epochs=20):
        super().__init__()
        self.trainer = trainer
        self.model_type = model_type
        self.epochs = epochs

    def run(self):
        logger.info(f"Starting {self.model_type} training inside QThread.")
        
        # Start training via the trainer class
        if self.model_type == "static":
            success = self.trainer.train_static_alphabet_async()
        else:
            success = self.trainer.train_dynamic_signs_async(epochs=self.epochs)
            
        if not success:
            self.training_finished.emit({"success": False, "error": "Training thread could not start."})
            return
            
        # Poll training progress state periodically
        while True:
            state = get_training_status()
            
            # Emit progress update
            self.progress_changed.emit(state["progress"], state["status"])
            
            # Check if training has terminated
            if not state["is_training"]:
                if state["progress"] >= 100:
                    logger.info("QThread training completed successfully.")
                    self.training_finished.emit({"success": True, "metrics": state["metrics"]})
                else:
                    logger.error(f"QThread training failed: {state['status']}")
                    self.training_finished.emit({"success": False, "error": state["status"]})
                break
                
            time.sleep(0.5)
