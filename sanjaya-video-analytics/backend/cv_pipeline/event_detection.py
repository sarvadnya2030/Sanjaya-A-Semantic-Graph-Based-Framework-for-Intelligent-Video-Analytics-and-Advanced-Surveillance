from collections import defaultdict
import math
import logging
import numpy as np

log = logging.getLogger("events")

class EventDetector:
    def __init__(self, fps=30, loiter_sec=3, proximity_px=80, prox_frames=10):
        self.fps = fps
        self.loiter_frames = int(loiter_sec * fps)
        self.stop_counts = defaultdict(int)  # tid -> frames stopped
        self.proximity_px = proximity_px
        self.prox_frames = prox_frames
        self.pair_close = defaultdict(int)  # frozenset(t1,t2) -> frames

    def detect(self, fsm_states, zone_events):
        events = []

        for st in fsm_states:
            tid = st["track_id"]
            if st["state"] == "STOPPED":
                self.stop_counts[tid] += 1
                if self.stop_counts[tid] == self.loiter_frames:
                    events.append({
                        "event_type": "loitering",
                        "track_id": tid,
                        "duration_sec": st["state_duration_sec"]
                    })
            else:
                self.stop_counts[tid] = 0

        for ze in zone_events:
            events.append({
                "event_type": ze["event_type"],
                "track_id": ze["track_id"],
                "zone": ze.get("zone", "unknown")
            })

        centers = [(s["track_id"], s.get("center")) for s in fsm_states if s.get("center")]
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                (t1, c1), (t2, c2) = centers[i], centers[j]
                if not c1 or not c2:
                    continue
                d = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
                key = frozenset((t1, t2))
                if d <= self.proximity_px:
                    self.pair_close[key] += 1
                    if self.pair_close[key] == self.prox_frames:
                        events.append({
                            "event_type": "interaction",
                            "track_id": list(key)[0],
                            "peer_id": list(key)[1],
                            "distance_px": d
                        })
                else:
                    self.pair_close[key] = 0

        return events

class EventGenerator:
    def __init__(self):
        """
        Generate surveillance events from tracks.
        """
        self.track_history = {}  # {track_id: {'positions': [], 'zone_history': []}}
        log.info("[EventGenerator] ✅ Initialized")
    
    def generate_events(self, tracks, frame_id, timestamp):
        """
        Generate events from tracks.
        Returns: list of event dicts
        """
        if not tracks:
            return []
        
        events = []
        
        for track in tracks:
            track_id = track['track_id']
            
            # Initialize track history
            if track_id not in self.track_history:
                self.track_history[track_id] = {
                    'positions': [],
                    'zone_history': [],
                    'first_seen': frame_id
                }
            
            bbox = track['bbox']
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            zone = track.get('zone', 'unknown')
            
            self.track_history[track_id]['positions'].append((cx, cy, frame_id))
            self.track_history[track_id]['zone_history'].append(zone)
            
            # Keep last 30 positions
            if len(self.track_history[track_id]['positions']) > 30:
                self.track_history[track_id]['positions'].pop(0)
                self.track_history[track_id]['zone_history'].pop(0)
            
            # Detect events
            positions = self.track_history[track_id]['positions']
            
            if len(positions) >= 10:
                movement = self._calculate_movement(positions)
                speed = movement / len(positions)
                
                motion_state = 'STATIONARY'
                priority = 'low'
                
                if speed > 5:  # Fast movement
                    motion_state = 'MOVING'
                    priority = 'medium'
                elif speed > 2:
                    motion_state = 'WALKING'
                    priority = 'low'
                elif len(positions) >= 20 and speed < 1:  # Loitering
                    motion_state = 'LOITERING'
                    priority = 'high'
                
                events.append({
                    'type': f'person_{motion_state.lower()}',
                    'track_id': track_id,
                    'frame_id': frame_id,
                    'timestamp': timestamp,
                    'motion_state': motion_state,
                    'speed_px_s': speed,
                    'zone': zone,
                    'priority': priority,
                    'bbox': bbox
                })
        
        return events
    
    def _calculate_movement(self, positions):
        """Calculate total movement distance."""
        if len(positions) < 2:
            return 0.0
        
        total_dist = 0.0
        for i in range(1, len(positions)):
            x1, y1, _ = positions[i-1]
            x2, y2, _ = positions[i]
            dist = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
            total_dist += dist
        
        return total_dist
