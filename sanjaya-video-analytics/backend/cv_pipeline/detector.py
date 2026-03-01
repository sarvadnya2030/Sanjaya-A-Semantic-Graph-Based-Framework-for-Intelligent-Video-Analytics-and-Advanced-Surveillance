import cv2
import numpy as np
from ultralytics import YOLO
import logging

log = logging.getLogger("detector")

class YOLODetector:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.5):
        """
        Initialize YOLO detector.
        """
        try:
            self.model = YOLO(model_path)
            self.conf_threshold = conf_threshold
            log.info(f"[Detector] ✅ Loaded YOLO model: {model_path}")
        except Exception as e:
            log.error(f"[Detector] Failed to load model: {e}")
            raise
    
    def detect(self, frame):
        """
        Run detection on frame.
        Returns: list of detections [{'class': str, 'confidence': float, 'bbox': [x1,y1,x2,y2]}]
        """
        if frame is None or frame.size == 0:
            return []
        
        try:
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                    
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    class_name = self.model.names[cls_id]
                    
                    detections.append({
                        'class': class_name,
                        'class_id': cls_id,
                        'confidence': conf,
                        'bbox': [float(x1), float(y1), float(x2), float(y2)]
                    })
            
            return detections
            
        except Exception as e:
            log.error(f"[Detector] Error during detection: {e}")
            return []

