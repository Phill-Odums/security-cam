import cv2
import threading
import time
import logging
from typing import Optional, Tuple
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoStream:
    def __init__(self, src=0):
        self.src = src
        self.stream = None
        self.grabbed = False
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        self.initialized = False
        self.last_error = None
        self.frame_counter = 0
        self.last_frame_time = time.time()
        # Restart/backoff config
        self.max_restart_attempts = 3
        self.restart_backoff = 0.5
        
        
    def start(self):
        """Start camera with robust initialization - avoid DSHOW which crashes on indexed access"""
        try:
            print(f"🚀 Initializing camera {self.src}...")
            # Try to open camera safely
            ok = self._open_stream()
            if not ok:
                return self

            print(f"✅ Camera {self.src} opened successfully")
            # Set camera properties (non-blocking)
            self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.stream.set(cv2.CAP_PROP_FPS, 20)
            self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Start thread immediately - warmup happens in background
            self.stopped = False
            self.thread = threading.Thread(target=self.update, daemon=True)
            self.thread.start()
            # Mark as initialized after starting thread
            self.initialized = True
            print(f"✅ Camera {self.src} stream started!")

        except Exception as e:
            self.last_error = f"Start error: {str(e)}"
            print(f"❌ {self.last_error}")
            if self.stream:
                self.stream.release()
                self.stream = None

        return self

    def _open_stream(self) -> bool:
        """Attempt to open VideoCapture safely and set properties."""
        # Try a small set of backends to avoid DSHOW issues on some Windows systems.
        backends = [None]
        try:
            import sys
            if sys.platform == "win32":
                # Prefer MSMF (Microsoft Media Foundation) on Windows when available
                if hasattr(cv2, 'CAP_MSMF'):
                    backends.insert(0, cv2.CAP_MSMF)
                # Keep DSHOW as a fallback but after MSMF
                if hasattr(cv2, 'CAP_DSHOW'):
                    backends.append(cv2.CAP_DSHOW)
        except Exception:
            pass

        last_exc = None
        for backend in backends:
            try:
                if backend is None:
                    cap = cv2.VideoCapture(self.src)
                else:
                    cap = cv2.VideoCapture(self.src, backend)
            except Exception as e:
                last_exc = e
                print(f"❌ VideoCapture open raised exception for backend {backend}: {e}")
                cap = None

            if cap is None:
                continue

            try:
                if not cap.isOpened():
                    # release and continue trying
                    try:
                        cap.release()
                    except Exception:
                        pass
                    continue
            except Exception as e:
                last_exc = e
                try:
                    cap.release()
                except Exception:
                    pass
                continue

            # success
            self.stream = cap
            # Try setting some properties, ignore failures
            try:
                self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.stream.set(cv2.CAP_PROP_FPS, 20)
                self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            return True

        # All backends failed
        if last_exc is not None:
            self.last_error = f"VideoCapture open failed, last exception: {last_exc}"
            print(f"❌ {self.last_error}")
        else:
            self.last_error = f"Failed to open camera {self.src} with available backends"
            print(f"❌ {self.last_error}")
        return False
    
    def update(self):
        """Frame update thread with continuous monitoring"""
        # Simple continuous read loop without health/restart logic
        while not self.stopped:
            try:
                if self.stream and self.stream.isOpened():
                    grabbed, frame = self.stream.read()
                    if grabbed and frame is not None and getattr(frame, 'size', 0) > 0:
                        with self.lock:
                            self.grabbed = True
                            self.frame = frame
                            self.frame_counter += 1
                            self.last_frame_time = time.time()
                    else:
                        # No frame this iteration; mark not grabbed and keep trying
                        with self.lock:
                            self.grabbed = False
                        time.sleep(0.01)
                        continue
                else:
                    # Stream not opened; wait and retry
                    time.sleep(0.1)
                    continue

            except Exception:
                # Ignore transient errors and continue reading
                time.sleep(0.05)
                continue

            # Control frame rate (throttle)
            time.sleep(0.033)  # ~30 FPS max
            
    def read(self):
        """Get the latest frame with validation"""
        with self.lock:
            if (self.frame is not None and 
                self.grabbed and 
                self.frame.size > 0 and 
                len(self.frame.shape) == 3):
                return True, self.frame.copy()
            return False, None
        
    def stop(self):
        """Stop the camera stream"""
        self.stopped = True
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.stream and self.stream.isOpened():
            self.stream.release()
        self.initialized = False
        print(f"🛑 Camera {self.src} stopped")
        print(f"   Total frames processed: {self.frame_counter}")

    def _restart_stream(self) -> bool:
        """Attempt to restart the underlying VideoCapture up to max attempts."""
        # Release existing stream
        try:
            if self.stream:
                try:
                    self.stream.release()
                except:
                    pass
                self.stream = None
        except Exception:
            pass
        self.restart_attempts = 0
        while self.restart_attempts < getattr(self, 'max_restart_attempts', 3) and not self.stopped:
            self.restart_attempts += 1
            try:
                print(f"🔁 Restart attempt {self.restart_attempts} for camera {self.src}")
                ok = self._open_stream()
                if ok:
                    print(f"✅ Restarted camera {self.src} successfully")
                    self.initialized = True
                    return True
            except Exception as e:
                print(f"❌ Exception during restart attempt: {e}")
            time.sleep(getattr(self, 'restart_backoff', 0.5) * self.restart_attempts)

        return False

class CameraManager:
    def __init__(self):
        self.stream = None
        self.available_cameras = [0]  # Your camera is at index 0
    
    def start_stream(self, camera_id=0):
        """Start camera stream with extended initialization"""
        if self.stream is not None:
            self.stop_stream()
        
        print(f"🎬 Starting camera {camera_id}...")
        self.stream = VideoStream(camera_id).start()
        
        # Extended wait for full initialization
        time.sleep(3.0)
        
        # Verify stream is actually working
        if self.stream and self.stream.initialized:
            # Test that we can actually get frames
            ret, frame = self.stream.read()
            if ret and frame is not None:
                print(f"✅ Camera {camera_id} fully operational!")
                print(f"   Ready to stream frames")
                return True
            else:
                print(f"⚠️ Camera {camera_id} initialized but no frames available")
        
        error_msg = self.stream.last_error if self.stream else "Unknown error"
        print(f"❌ Camera {camera_id} failed: {error_msg}")
        
        if self.stream:
            self.stream.stop()
            self.stream = None
            
        return False
    
    def stop_stream(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream = None
    
    def get_frame(self):
        if self.stream is not None and self.stream.initialized:
            return self.stream.read()
        return False, None
    
    def is_running(self):
        return self.stream is not None and self.stream.initialized
    
    def get_available_cameras(self):
        return self.available_cameras.copy()
    
    def get_camera_info(self):
        if self.stream and self.stream.initialized:
            return {
                "status": "running",
                "camera_index": self.stream.src,
                "frames_processed": self.stream.frame_counter,
                "available_cameras": self.available_cameras
            }
        else:
            return {
                "status": "stopped",
                "available_cameras": self.available_cameras
            }