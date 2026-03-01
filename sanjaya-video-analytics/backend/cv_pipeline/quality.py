import cv2
import numpy as np

class QualityAssessor:
    def assess(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return {"blur": 0, "lighting": "low", "occlusion_ratio": 1.0}
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        mean_intensity = np.mean(gray)
        lighting = "good" if mean_intensity > 80 else "low"
        return {"blur": float(blur), "lighting": lighting, "occlusion_ratio": 0.0}
