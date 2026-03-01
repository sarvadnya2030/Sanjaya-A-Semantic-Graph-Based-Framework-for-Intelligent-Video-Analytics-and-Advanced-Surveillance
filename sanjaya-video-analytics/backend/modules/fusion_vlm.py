import json
from modules.ollama_vlm import analyze_frame_with_vlm

def run_vlm_with_grounding(frame_path, video_name, frame_id, cv_grounding):
    return analyze_frame_with_vlm(frame_path, video_name, frame_id, grounding=cv_grounding)