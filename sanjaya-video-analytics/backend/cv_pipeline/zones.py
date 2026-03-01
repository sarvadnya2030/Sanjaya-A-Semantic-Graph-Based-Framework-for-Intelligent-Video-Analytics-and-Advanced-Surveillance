import numpy as np
import logging

log = logging.getLogger("zones")

class ZoneAnalyzer:
    def __init__(self):
        """
        Analyze spatial zones in frame (grid-based).
        """
        self.zones = {
            'Zone1': {'x': 0, 'y': 0, 'w': 640, 'h': 360, 'name': 'Top-Left'},
            'Zone2': {'x': 640, 'y': 0, 'w': 640, 'h': 360, 'name': 'Top-Right'},
            'Zone3': {'x': 0, 'y': 360, 'w': 640, 'h': 360, 'name': 'Bottom-Left'},
            'Zone4': {'x': 640, 'y': 360, 'w': 640, 'h': 360, 'name': 'Bottom-Right'}
        }
        log.info("[ZoneAnalyzer] ✅ Initialized with 4 zones")
    
    def analyze(self, tracks, frame):
        """
        Assign zones to tracks.
        Returns: dict of zone data
        """
        if not tracks:
            return {zone: {'count': 0, 'track_ids': []} for zone in self.zones}
        
        zone_data = {zone: {'count': 0, 'track_ids': []} for zone in self.zones}
        
        for track in tracks:
            bbox = track['bbox']
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            
            # Assign to zone
            assigned = False
            for zone_name, zone_rect in self.zones.items():
                if (zone_rect['x'] <= cx < zone_rect['x'] + zone_rect['w'] and
                    zone_rect['y'] <= cy < zone_rect['y'] + zone_rect['h']):
                    zone_data[zone_name]['count'] += 1
                    zone_data[zone_name]['track_ids'].append(track['track_id'])
                    track['zone'] = zone_name
                    assigned = True
                    break
            
            if not assigned:
                track['zone'] = 'unknown'
        
        return zone_data

class ZoneManager:
    def __init__(self, zones=None):
        self.h, self.w = None, None
        self.last_zone = {}   # tid -> zone_id
        self.zone_counts = {} # zone_id -> count

    def set_frame_shape(self, shape):
        self.h, self.w = shape[:2]

    def _zone_of(self, center):
        if self.h is None or self.w is None:
            return "Z?"
        cx, cy = center
        col = min(int(cx / (self.w / 3)), 2)
        row = min(int(cy / (self.h / 3)), 2)
        return f"Z{row*3 + col + 1}"

    def check(self, tracks):
        events = []
        for tr in tracks:
            tid = tr["track_id"]
            x1, y1, x2, y2 = tr["bbox"]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            zid = self._zone_of((cx, cy))
            tr["zone"] = zid  # <-- store zone on track
            prev = self.last_zone.get(tid)
            if zid != prev:
                if prev:
                    events.append({"event_type": "zone_exit", "track_id": tid, "zone": prev})
                events.append({"event_type": "zone_entry", "track_id": tid, "zone": zid})
                self.last_zone[tid] = zid
            self.zone_counts[zid] = self.zone_counts.get(zid, 0) + 1
        return events
