import cv2
import numpy as np

class MotionGating:
    def __init__(self, threshold=25):
        self.threshold = threshold
        self.prev_gray = None
        self.frame_size = None  # Track expected frame size
    
    def process(self, frame):
        """
        Robust motion detection with size validation.
        Returns motion magnitude (0-100).
        """
        if frame is None or frame.size == 0:
            return 0.0
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        current_size = gray.shape
        
        # Initialize or handle size change
        if self.prev_gray is None or self.frame_size != current_size:
            self.prev_gray = gray.copy()
            self.frame_size = current_size
            return 0.0  # No motion on first frame or size change
        
        # Ensure sizes match (safety check)
        if gray.shape != self.prev_gray.shape:
            # Resize prev_gray to match current frame
            self.prev_gray = cv2.resize(self.prev_gray, (gray.shape[1], gray.shape[0]))
            self.frame_size = current_size
        
        try:
            # Calculate frame difference
            diff = cv2.absdiff(gray, self.prev_gray)
            
            # Threshold to get motion mask
            _, thresh = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
            
            # Calculate motion percentage
            motion_pixels = cv2.countNonZero(thresh)
            total_pixels = thresh.shape[0] * thresh.shape[1]
            motion_magnitude = (motion_pixels / total_pixels) * 100.0
            
            # Update previous frame
            self.prev_gray = gray.copy()
            
            return motion_magnitude
            
        except cv2.error as e:
            # Handle any OpenCV errors gracefully
            print(f"[MotionGating] OpenCV error: {e}, resetting...")
            self.prev_gray = gray.copy()
            return 0.0
    
    def reset(self):
        """Reset motion detection state."""
        self.prev_gray = None
        self.frame_size = None
