import threading
import queue
import logging
import time

logger = logging.getLogger(__name__)

class TextToSpeechEngine:
    def __init__(self):
        self.speech_queue = queue.Queue()
        self.worker_thread = None
        self.running = False
        self.engine = None
        self.auto_speak = False
        self.lock = threading.Lock()
        
        # Start worker thread
        self.start_worker()

    def start_worker(self):
        self.running = True
        self.worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.worker_thread.start()
        logger.info("TTS background thread started.")

    def _speech_worker(self):
        """
        Background worker that processes TTS requests.
        Crucial: pyttsx3 must be initialized within the thread it runs on.
        """
        import pyttsx3
        
        # Initialize pyttsx3 in this thread
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)  # Moderate speed
            self.engine.setProperty('volume', 1.0)
            logger.info("pyttsx3 initialized successfully in worker thread.")
        except Exception as e:
            logger.error(f"Failed to initialize pyttsx3 engine: {e}")
            self.engine = None
            
        while self.running:
            try:
                # Wait for speech item
                try:
                    text, lang = self.speech_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                    
                if self.engine is None:
                    logger.warning("TTS Engine not available. Simulating speech.")
                    self.speech_queue.task_done()
                    continue
                    
                with self.lock:
                    self._set_voice_by_lang(lang)
                    logger.info(f"Speaking: '{text}' in lang '{lang}'")
                    self.engine.say(text)
                    self.engine.runAndWait()
                    
                self.speech_queue.task_done()
            except Exception as e:
                logger.error(f"Error in TTS execution loop: {e}")
                # Re-initialize engine if crashed
                try:
                    self.engine = pyttsx3.init()
                except:
                    pass
                time.sleep(1.0)

    def _set_voice_by_lang(self, lang):
        """
        Maps standard language ISO codes to available system voices.
        """
        if self.engine is None:
            return
            
        try:
            voices = self.engine.getProperty('voices')
            target_voice = None
            
            # Search system voices for keywords
            lang = lang.lower()
            if lang == 'hi':  # Hindi
                for voice in voices:
                    if 'hindi' in voice.name.lower() or 'india' in voice.name.lower() or 'hi_IN' in voice.id:
                        target_voice = voice.id
                        break
            elif lang == 'te':  # Telugu
                for voice in voices:
                    if 'telugu' in voice.name.lower() or 'te_IN' in voice.id:
                        target_voice = voice.id
                        break
            
            # Default to English or first voice if specific is not found
            if not target_voice:
                for voice in voices:
                    if 'english' in voice.name.lower() or 'en_' in voice.id or 'en-US' in voice.id:
                        target_voice = voice.id
                        break
                        
            if not target_voice and voices:
                target_voice = voices[0].id
                
            if target_voice:
                self.engine.setProperty('voice', target_voice)
        except Exception as e:
            logger.error(f"Error selecting TTS voice: {e}")

    def speak(self, text, lang='en'):
        """
        Enqueues a string to be spoken asynchronously.
        """
        if not text or not text.strip():
            return
        self.speech_queue.put((text, lang))

    def stop(self):
        """
        Stops the speech engine.
        """
        # Clear queue
        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
                self.speech_queue.task_done()
            except queue.Empty:
                break
                
        # Call stop on engine if active
        if self.engine is not None:
            try:
                # pyttsx3 engine.stop() resets speech
                self.engine.stop()
                logger.info("TTS stopped.")
            except Exception as e:
                logger.error(f"Error stopping TTS engine: {e}")

    def set_auto_speak(self, enabled):
        self.auto_speak = bool(enabled)

    def is_auto_speak(self):
        return self.auto_speak

    def shutdown(self):
        self.running = False
        self.stop()
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)
            logger.info("TTS shutdown complete.")
