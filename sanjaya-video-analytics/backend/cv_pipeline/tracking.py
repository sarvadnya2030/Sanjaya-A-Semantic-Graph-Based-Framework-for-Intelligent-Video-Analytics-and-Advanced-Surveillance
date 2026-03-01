import numpy as np
from collections import defaultdict
import logging

log = logging.getLogger("tracker")

class DeepSORTTracker:
    def __init__(self):
        """
        Simple IoU-based tracker.
        """
        self.tracks = {}  # {track_id: {'bbox': [], 'class': str, 'last_seen': int, 'positions': []}}
        self.next_id = 1
        self.max_age = 30  # frames
        log.info("[Tracker] ✅ Initialized")
    
    def update(self, detections, frame_id):
        """
        Update tracks with new detections.
        Returns: list of tracks [{'track_id': int, 'class': str, 'bbox': [x1,y1,x2,y2]}]
        """
        if not detections:
            # Age out old tracks
            self._remove_old_tracks(frame_id)
            return []
        
        # Filter for persons only
        person_detections = [d for d in detections if d['class'] == 'person']
        
        matched_tracks = []
        unmatched_detections = []
        
        # Match detections to existing tracks
        for det in person_detections:
            best_iou = 0.0
            best_track_id = None
            
            for track_id, track_data in self.tracks.items():
                iou = self._iou(det['bbox'], track_data['bbox'])
                if iou > best_iou and iou > 0.3:
                    best_iou = iou
                    best_track_id = track_id
            
            if best_track_id:
                # Update existing track
                self.tracks[best_track_id]['bbox'] = det['bbox']
                self.tracks[best_track_id]['last_seen'] = frame_id
                self.tracks[best_track_id]['confidence'] = det['confidence']
                
                # Store position history
                cx = (det['bbox'][0] + det['bbox'][2]) / 2
                cy = (det['bbox'][1] + det['bbox'][3]) / 2
                self.tracks[best_track_id]['positions'].append((cx, cy, frame_id))
                
                # Keep last 30 positions
                if len(self.tracks[best_track_id]['positions']) > 30:
                    self.tracks[best_track_id]['positions'].pop(0)
                
                matched_tracks.append({
                    'track_id': best_track_id,
                    'class': 'person',
                    'bbox': det['bbox'],
                    'confidence': det['confidence'],
                    'positions': self.tracks[best_track_id]['positions']
                })
            else:
                # Create new track
                new_id = self.next_id
                self.next_id += 1
                
                cx = (det['bbox'][0] + det['bbox'][2]) / 2
                cy = (det['bbox'][1] + det['bbox'][3]) / 2
                
                self.tracks[new_id] = {
                    'bbox': det['bbox'],
                    'class': 'person',
                    'last_seen': frame_id,
                    'confidence': det['confidence'],
                    'positions': [(cx, cy, frame_id)]
                }
                
                matched_tracks.append({
                    'track_id': new_id,
                    'class': 'person',
                    'bbox': det['bbox'],
                    'confidence': det['confidence'],
                    'positions': [(cx, cy, frame_id)]
                })
        
        # Remove old tracks
        self._remove_old_tracks(frame_id)
        
        return matched_tracks
    
    def _remove_old_tracks(self, frame_id):
        """Remove tracks that haven't been seen recently."""
        to_remove = [tid for tid, data in self.tracks.items() 
                    if frame_id - data['last_seen'] > self.max_age]
        for tid in to_remove:
            del self.tracks[tid]
    
    def _iou(self, box1, box2):
        """Calculate IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        inter = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0.0
