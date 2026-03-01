from ultralytics import YOLO
import torch

class ObjectDetector:
    def __init__(self, model_path="yolov8s.pt", conf_thresh=0.5, imgsz=640):
        self.model = YOLO(model_path)
        self.classes = [0, 2, 24, 26, 28]  # person, car, backpack, handbag, suitcase
        self.conf_thresh = conf_thresh
        self.imgsz = imgsz
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model.to(self.device)
        except Exception:
            pass

    def detect(self, frame):
        results = self.model.predict(
            frame, conf=self.conf_thresh, classes=self.classes,
            imgsz=self.imgsz, verbose=False, device=self.device
        )
        dets = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                dets.append({
                    "class": r.names[cls],
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": conf
                })
        return dets
