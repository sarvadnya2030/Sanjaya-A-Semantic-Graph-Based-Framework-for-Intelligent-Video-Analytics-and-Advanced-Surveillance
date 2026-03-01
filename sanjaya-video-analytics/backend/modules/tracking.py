# Minimal ByteTrack-like placeholder (swap with real lib if needed)
def attach_tracks(dets, prev_tracks=None):
    # naive ID assignment by IoU; replace with ByteTrack for production
    tracks = []
    for i, d in enumerate(dets):
        tracks.append({"id": f"T{i+1}", "bbox": d["bbox"], "cls": d["cls"], "conf": d["conf"]})
    return tracks