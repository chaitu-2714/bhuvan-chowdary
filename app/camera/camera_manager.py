import cv2
import time
import logging

logger = logging.getLogger(__name__)

class CameraManager:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.fps = 0.0
        self.prev_time = 0.0
        self.frame_count = 0
        self.fps_calc_interval = 1.0  # Calculate FPS every 1 second
        self.fps_timer = 0.0

    def start(self):
        """
        Starts the camera stream. Returns True if successful, False otherwise.
        """
        if self.cap is not None and self.cap.isOpened():
            return True
            
        try:
            # On Windows, using DSHOW often speeds up startup and reduces warnings
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                # Fallback to default backend if DSHOW fails
                self.cap = cv2.VideoCapture(self.camera_index)
                
            if self.cap.isOpened():
                # Set default resolution to 640x480 for real-time performance
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.prev_time = time.time()
                self.fps_timer = time.time()
                self.frame_count = 0
                logger.info(f"Camera {self.camera_index} started successfully.")
                return True
            else:
                logger.error(f"Failed to open camera index {self.camera_index}.")
                self.cap = None
                return False
        except Exception as e:
            logger.error(f"Error starting camera: {e}")
            self.cap = None
            return False

    def read_frame(self):
        """
        Reads a frame from the camera.
        Returns (success, frame).
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None
            
        try:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Camera read failed to return a frame.")
                return False, None
                
            # Calculate FPS
            self.frame_count += 1
            current_time = time.time()
            elapsed = current_time - self.fps_timer
            if elapsed >= self.fps_calc_interval:
                self.fps = self.frame_count / elapsed
                self.frame_count = 0
                self.fps_timer = current_time
                
            return True, frame
        except Exception as e:
            logger.error(f"Error reading frame: {e}")
            return False, None

    def stop(self):
        """
        Stops the camera stream.
        """
        if self.cap is not None:
            try:
                self.cap.release()
                logger.info("Camera released.")
            except Exception as e:
                logger.error(f"Error releasing camera: {e}")
            finally:
                self.cap = None
                self.fps = 0.0

    def is_opened(self):
        return self.cap is not None and self.cap.isOpened()

    def get_fps(self):
        return self.fps
