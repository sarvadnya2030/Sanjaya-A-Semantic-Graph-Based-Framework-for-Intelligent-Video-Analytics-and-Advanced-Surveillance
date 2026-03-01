import os, cv2
from config import FRAMES_FOLDER

def extract_3_frames(video_path, video_name):
    """Extract start, middle, end frames from video"""
    os.makedirs(FRAMES_FOLDER, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    
    if total <= 0:
        raise RuntimeError(f"Video has no frames: {video_path}")
    
    idxs = [0, max(0, total//2), max(0, total-1)]
    labels = ["start", "middle", "end"]
    frames = []
    
    for fid, lab in zip(idxs, labels):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, frame = cap.read()
        
        if not ok or frame is None:
            print(f"   ⚠️ Failed to read frame {fid} ({lab})")
            continue
        
        fname = f"{video_name}_{lab}_frame{fid}.jpg"
        fpath = os.path.join(FRAMES_FOLDER, fname)
        cv2.imwrite(fpath, frame)
        
        frames.append({
            "frame_id": fid, 
            "label": lab, 
            "path": fpath, 
            "filename": fname
        })
        
        print(f"   ✅ {lab.upper()}: frame {fid} → {fname}")
    
    cap.release()
    
    if not frames:
        raise RuntimeError("Failed to extract any frames")
    
    return frames, total
