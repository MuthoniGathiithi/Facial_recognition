import cv2
import threading
import time

class VideoCapture:
    """
    A threaded video capture class to handle camera operations.
    """
    
    def __init__(self, src=0, width=1280, height=720, fps=30):
        """
        Initialize the video capture.
        
        Args:
            src: Camera source (default is 0 for default camera)
            width: Desired frame width
            height: Desired frame height
            fps: Desired frames per second
        """
        self.src = src
        self.width = width
        self.height = height
        self.fps = fps
        self.frame = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.cap = None

    def start(self):
        """Start the video capture thread."""
        if self.running:
            return False
            
        # Initialize the video capture
        self.cap = cv2.VideoCapture(self.src)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open video source")
            
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return True

    def _update(self):
        """Thread function to continuously grab frames."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                break
                
            with self.lock:
                self.frame = frame.copy()
                
            # Control frame rate
            time.sleep(1.0 / self.fps)

    def read(self):
        """
        Read the most recent frame.
        
        Returns:
            tuple: (frame, success) where success is a boolean indicating
                   if the frame was successfully read
        """
        if not self.running:
            return None, False
            
        with self.lock:
            if self.frame is None:
                return None, False
            return self.frame.copy(), True

    def stop(self):
        """Stop the video capture and release resources."""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None
            
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __del__(self):
        """Ensure resources are released when the object is destroyed."""
        self.stop()