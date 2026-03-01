import math

class KinematicsAnalyzer:
    def __init__(self, fps=30):
        self.fps = fps
        self.prev_centers = {}  # tid -> (cx, cy)

    def _center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def analyze(self, tracks):
        out = []
        for tr in tracks:
            tid = tr["track_id"]
            cx, cy = self._center(tr["bbox"])
            pc = self.prev_centers.get(tid)
            if pc:
                dx, dy = cx - pc[0], cy - pc[1]
                speed_px_s = math.hypot(dx, dy) * self.fps
                direction_deg = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
            else:
                speed_px_s, direction_deg = 0.0, 0.0
            self.prev_centers[tid] = (cx, cy)
            out.append({
                "track_id": tid,
                "class": tr.get("class", "unknown"),
                "center": (cx, cy),
                "speed_px_s": float(speed_px_s),
                "direction_deg": float(direction_deg),
                "bbox": tr["bbox"]
            })
        return out
