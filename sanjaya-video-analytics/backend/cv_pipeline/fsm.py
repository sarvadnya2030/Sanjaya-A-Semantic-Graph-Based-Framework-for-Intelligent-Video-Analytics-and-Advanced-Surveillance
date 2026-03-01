from enum import Enum

class TrackState(Enum):
    NEW = "NEW"
    MOVING = "MOVING"
    STOPPED = "STOPPED"
    EXIT = "EXIT"

class FSMTracker:
    def __init__(self, fps=30, stop_thresh=8.0, move_thresh=14.0, min_state_frames=5):
        self.fps = fps
        self.stop_thresh = stop_thresh
        self.move_thresh = move_thresh
        self.min_state_frames = min_state_frames
        self.state = {}  # tid -> {"state": str, "frames": int}

    def update(self, kin_list):
        results = []
        for k in kin_list:
            tid = k["track_id"]
            speed = k["speed_px_s"]
            prev = self.state.get(tid, {"state": "INIT", "frames": 0})
            st = prev["state"]

            if st in ("INIT", "MOVING"):
                new_state = "STOPPED" if speed < self.stop_thresh else "MOVING"
            else:
                new_state = "MOVING" if speed > self.move_thresh else "STOPPED"

            if new_state == st:
                frames = prev["frames"] + 1
            else:
                st, frames = new_state, 1

            self.state[tid] = {"state": st, "frames": frames}
            results.append({
                "track_id": tid,
                "state": st if frames >= self.min_state_frames else "TRANSIENT",
                "state_frames": frames,
                "state_duration_sec": frames / self.fps,
                "center": k["center"],
                "bbox": k["bbox"],
                "class": k.get("class", "unknown")
            })
        return results
