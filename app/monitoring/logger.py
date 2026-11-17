import logging
import os
import collections
import time
import sys

# Circular queue to hold the last 100 log statements in memory for web UI display
_log_buffer = collections.deque(maxlen=100)

class BufferHandler(logging.Handler):
    """
    Custom logging handler that stores formatted logs in an in-memory queue.
    """
    def emit(self, record):
        try:
            msg = self.format(record)
            _log_buffer.append({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created)),
                "level": record.levelname,
                "module": record.module,
                "message": record.getMessage()
            })
        except Exception:
            self.handleError(record)

def setup_logger(log_dir="logs", log_level=logging.INFO):
    """
    Configures the root logger to output to console, a log file, and the web UI memory buffer.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_path = os.path.join(log_dir, "app.log")
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clean existing handlers to avoid double logging
    root_logger.handlers = []
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # File Handler
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # UI Buffer Handler
    buffer_handler = BufferHandler()
    buffer_handler.setFormatter(formatter)
    root_logger.addHandler(buffer_handler)
    
    logging.getLogger("werkzeug").setLevel(logging.WARNING)  # Mute Flask server logs in console
    
    logging.info("Logger setup complete.")

def get_recent_logs():
    """
    Returns the log statements stored in the in-memory circular buffer.
    """
    return list(_log_buffer)

def get_system_health(camera_manager=None, landmark_detector=None):
    """
    Returns basic system performance metrics.
    """
    import psutil
    
    cpu_percent = psutil.cpu_percent()
    ram = psutil.virtual_memory()
    ram_percent = ram.percent
    
    camera_status = "Inactive"
    camera_fps = 0.0
    if camera_manager:
        camera_status = "Active" if camera_manager.is_opened() else "Closed"
        camera_fps = round(camera_manager.get_fps(), 1)
        
    mediapipe_status = "Not Initialized"
    if landmark_detector:
        mediapipe_status = "Active" if landmark_detector.hands is not None else "Closed"
        
    return {
        "cpu_usage": cpu_percent,
        "ram_usage": ram_percent,
        "camera_status": camera_status,
        "camera_fps": camera_fps,
        "mediapipe_status": mediapipe_status,
        "timestamp": time.time()
    }
