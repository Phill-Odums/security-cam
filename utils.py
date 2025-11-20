import cv2
import numpy as np
from PIL import Image
import io
import time
from collections import deque


def pil_to_cv2(pil_image):
    """Convert PIL Image to OpenCV format"""
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_image):
    """Convert OpenCV image to PIL format"""
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))



def to_jpg_bytes(cv2_image, quality=95):
    """Convert OpenCV BGR image to JPEG bytes"""
    try:
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_image)
        # Convert to bytes
        buf = io.BytesIO()
        pil_image.save(buf, format='JPEG', quality=quality)
        return buf.getvalue()
    except Exception as e:
        print(f"Error converting to JPEG: {e}")
        return None

def calculate_fps(frame_times, max_length=10):
    """Calculate FPS from frame times"""
    if not frame_times:
        return 0
    avg_time = sum(frame_times) / len(frame_times)
    return 1.0 / avg_time if avg_time > 0 else 0

class FPSCounter:
    """FPS counter utility"""
    def __init__(self, max_samples=10):
        self.times = deque(maxlen=max_samples)
        self.last_time = time.time()
    
    def update(self):
        """Update FPS calculation"""
        current_time = time.time()
        elapsed = current_time - self.last_time
        self.times.append(elapsed)
        self.last_time = current_time
        return self.get_fps()
    
    def get_fps(self):
        """Get current FPS"""
        if not self.times:
            return 0
        avg_time = sum(self.times) / len(self.times)
        return 1.0 / avg_time if avg_time > 0 else 0

def resize_frame(frame, width=None, height=None):
    """Resize frame while maintaining aspect ratio"""
    if width is None and height is None:
        return frame
    
    h, w = frame.shape[:2]
    
    if width is not None and height is not None:
        return cv2.resize(frame, (width, height))
    elif width is not None:
        ratio = width / w
        height = int(h * ratio)
        return cv2.resize(frame, (width, height))
    else:  # height is not None
        ratio = height / h
        width = int(w * ratio)
        return cv2.resize(frame, (width, height))