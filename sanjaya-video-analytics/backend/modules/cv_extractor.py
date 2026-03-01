import cv2, numpy as np
from ultralytics import YOLO

class RichCVExtractor:
    def __init__(self, model_path="yolov8n.pt", conf=0.3):
        self.model = YOLO(model_path)
        self.conf = conf

    def extract_zones(self, h, w):
        zones = {}
        row_h, col_w = h // 3, w // 3
        names = [["top_left","top_center","top_right"],
                 ["mid_left","mid_center","mid_right"],
                 ["bot_left","bot_center","bot_right"]]
        for i in range(3):
            for j in range(3):
                zones[f"Z{i*3+j+1}"] = {"name": names[i][j], "bbox": [j*col_w, i*row_h, col_w, row_h]}
        return zones

    def point_in_zone(self, x, y, z):
        zx, zy, zw, zh = z
        return zx <= x <= zx+zw and zy <= y <= zy+zh

    def estimate_clothing_color(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1: return "unknown"
        upper = frame[y1:y1+(y2-y1)//2, x1:x2]
        if upper.size == 0: return "unknown"
        hsv = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0], None, [180], [0,180])
        hue = int(np.argmax(h))
        if hue < 10 or hue >= 170: return "red"
        if hue < 25: return "orange"
        if hue < 35: return "yellow"
        if hue < 77: return "green"
        if hue < 99: return "cyan"
        if hue < 130: return "blue"
        if hue < 160: return "purple"
        return "gray"

    def infer(self, frame):
        h, w = frame.shape[:2]
        zones = self.extract_zones(h, w)
        res = self.model.predict(frame, conf=self.conf, verbose=False)[0]

        persons, objects, interactions = [], [], []

        for b in res.boxes:
            cls_id = int(b.cls.item())
            cls_name = res.names[cls_id]
            conf = float(b.conf.item())
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            cx, cy = (x1+x2)//2, (y1+y2)//2
            bw, bh = x2-x1, y2-y1

            zone_id = "Z5"
            for zid, zi in zones.items():
                if self.point_in_zone(cx, cy, zi["bbox"]):
                    zone_id = zid; break

            if cls_name == "person":
                aspect = bw / max(bh, 1)
                posture = "upright" if 0.3 < aspect < 0.7 else ("seated" if aspect > 0.7 else "unknown")
                persons.append({
                    "id": f"H{len(persons)+1}",
                    "class": "person",
                    "confidence": round(conf, 2),
                    "bbox": [x1, y1, bw, bh],
                    "center": [cx, cy],
                    "zone_id": zone_id,
                    "posture": posture,
                    "clothing_color": self.estimate_clothing_color(frame, [x1,y1,x2,y2])
                })
            else:
                objects.append({
                    "id": f"O{len(objects)+1}",
                    "class": cls_name,
                    "confidence": round(conf, 2),
                    "bbox": [x1, y1, bw, bh],
                    "center": [cx, cy],
                    "zone_id": zone_id,
                    "portable": cls_name in ["laptop","phone","bag","cup","book"]
                })

        for p in persons:
            for o in objects:
                dist = np.hypot(p["center"][0]-o["center"][0], p["center"][1]-o["center"][1])
                if dist < max(p["bbox"][2], p["bbox"][3]) * 2:
                    interactions.append({"person_id": p["id"], "object_id": o["id"], "type": "near", "distance": round(dist,1)})

        return {"persons": persons, "objects": objects, "zones": zones, "interactions": interactions, "frame_size": [w, h]}
