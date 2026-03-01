from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path="yolov8s.pt", conf_thresh=0.5):
        self.model = YOLO(model_path)
        self.classes = [0, 2, 24, 26, 28]  # person, car, backpack, handbag, suitcase
        self.conf_thresh = conf_thresh

    def detect(self, frame):
        res = self.model(frame, conf=self.conf_thresh, classes=self.classes, verbose=False)
        out = []
        for r in res:
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                cls = int(b.cls[0]); conf = float(b.conf[0])
                out.append({"class": self.model.names[cls], "bbox": [float(x1), float(y1), float(x2), float(y2)], "confidence": conf})
        return out