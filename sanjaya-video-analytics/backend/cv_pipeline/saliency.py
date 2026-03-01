class SaliencyScorer:
    """Compute event saliency for frame prioritization"""
    
    def __init__(self):
        self.weights = {
            "zone_entry": 0.3,
            "zone_exit": 0.25,
            "loitering": 0.8,
            "interaction": 0.9,
            "running": 0.95,
            "stopped": 0.6,
            "moving": 0.4
        }
    
    def score_frame(self, events, tracks, fsm_states):
        """
        Returns saliency score 0.0–1.0
        Higher = more interesting for VLM analysis
        """
        score = 0.0
        
        # Event diversity bonus
        event_types = {e.get("event") for e in events}
        score += len(event_types) * 0.15
        
        # Weighted event scores
        for e in events:
            etype = e.get("event", "")
            score += self.weights.get(etype, 0.2)
        
        # Multi-person interactions
        person_count = sum(1 for t in tracks if t.get("class") == "person")
        if person_count >= 3:
            score += 0.4
        elif person_count == 2:
            score += 0.2
        
        # Motion state diversity
        states = {s.get("state") for s in fsm_states}
        if "STOPPED" in states and "MOVING" in states:
            score += 0.3  # mixed activity
        
        # High-speed motion
        max_speed = max([s.get("speed_px_s", 0) for s in fsm_states] + [0])
        if max_speed > 50:
            score += 0.5  # running or rapid movement
        
        return min(1.0, score)