import cv2
import numpy as np
from PIL import Image
import io
from ultralytics import YOLO
import os

class DetectionSystem:
    def __init__(self, model_path='C:/Users/phill/.spyder-py3/projects/security_system/yolov8n.pt', confidence=0.25, device='cpu'):
        self.model_path = model_path
        self.confidence = confidence
        self.device = device
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the YOLO model"""
        try:
            self.model = YOLO(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def set_confidence(self, confidence):
        """Update confidence threshold"""
        self.confidence = confidence
    
    def set_device(self, device):
        """Update inference device"""
        self.device = device
    
    def process_image(self, image):
        """
        Process a single image
        Args:
            image: PIL Image or numpy array
        Returns:
            annotated_image: numpy array with detections
            detections: list of detection results
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)[:, :, ::-1].copy()  # RGB to BGR
        
        # Run inference
        results = self.model(image, conf=self.confidence, device=self.device)
        annotated = results[0].plot()
        
        # Extract detections
        detections = self._extract_detections(results)
        
        return annotated, detections
    
    def process_frame(self, frame):
        """
        Process a video frame
        Args:
            frame: numpy array (BGR format)
        Returns:
            annotated_frame: numpy array with detections
            detections: list of detection results
            alert_triggered: bool if target detected
        """
        if self.model is None:
            return frame, [], False
        
        results = self.model(frame, conf=self.confidence, device=self.device)
        annotated = results[0].plot()
        detections = self._extract_detections(results)
        
        # Check for target objects
        alert_triggered = self._check_target_detection(results)
        
        return annotated, detections, alert_triggered
    
    def _extract_detections(self, results):
        """Extract detection information from results"""
        detections = []
        boxes = getattr(results[0], 'boxes', None)
        
        if boxes is not None and len(boxes) > 0:
            for i, (xy, cls, confv) in enumerate(zip(boxes.xyxy, boxes.cls, boxes.conf)):
                coords = xy.cpu().numpy() if hasattr(xy, 'cpu') else np.array(xy)
                cls_id = int(cls.cpu().numpy()) if hasattr(cls, 'cpu') else int(cls)
                label = self.model.names.get(cls_id, str(cls_id))
                
                detections.append({
                    'label': label,
                    'confidence': float(confv),
                    'x1': int(coords[0]), 'y1': int(coords[1]),
                    'x2': int(coords[2]), 'y2': int(coords[3])
                })
        
        return detections
    
    def _check_target_detection(self, results):
        """Check if target objects are detected"""
        target_labels = {'person', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
                        'elephant', 'bear', 'zebra', 'giraffe'}
        
        boxes = getattr(results[0], 'boxes', None)
        if boxes is not None and hasattr(boxes, 'cls'):
            for cls_item in boxes.cls:
                try:
                    cls_id = int(cls_item.cpu().numpy()) if hasattr(cls_item, 'cpu') else int(cls_item)
                    label = self.model.names.get(cls_id, str(cls_id))
                    if label in target_labels:
                        return True
                except Exception:
                    continue
        return False

# Example usage for quick testing
if __name__ == "__main__":
    detector = DetectionSystem()
    print("Detection module loaded successfully")