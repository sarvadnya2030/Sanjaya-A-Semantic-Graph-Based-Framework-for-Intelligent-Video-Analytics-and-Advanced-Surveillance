from ultralytics import YOLO

class Detector:
    def __init__(self, model_path="yolov8n.pt", conf=0.25):
        self.model = YOLO(model_path)
        self.conf = conf
    def infer(self, frame):
        res = self.model.predict(frame, conf=self.conf, verbose=False)[0]
        dets = []
        for b in res.boxes:
            cls = int(b.cls.item()); conf = float(b.conf.item())
            x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
            dets.append({"cls": res.names[cls], "conf": conf, "bbox": [x1,y1,x2-x1,y2-y1]})
        return dets