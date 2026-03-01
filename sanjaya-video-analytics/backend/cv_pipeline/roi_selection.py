import cv2
import os

class ROISelector:
    def __init__(self, out_dir="static/frames/roi"):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def select(self, frame, track_id, bbox, quality):
        if quality["blur"] < 50:
            return None
        x1, y1, x2, y2 = map(int, bbox)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        path = os.path.join(self.out_dir, f"track_{track_id}.jpg")
        cv2.imwrite(path, crop)
        return path
