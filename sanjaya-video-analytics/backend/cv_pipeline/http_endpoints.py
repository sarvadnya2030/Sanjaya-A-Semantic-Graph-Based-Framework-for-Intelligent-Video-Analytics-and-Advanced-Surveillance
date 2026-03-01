import os
import uuid
import tempfile
import logging
from typing import Any
from .pipeline import CVPipeline

log = logging.getLogger("cv.http")

def register_cv_routes(app: Any) -> None:
    """Register CV pipeline routes (Flask/FastAPI compatible)"""
    
    # Flask
    if hasattr(app, "add_url_rule"):
        from flask import request, jsonify
        
        def upload_cv():
            file = request.files.get("video")
            if not file:
                return jsonify({"error": "video required"}), 400
            
            tmpdir = tempfile.mkdtemp(prefix="cv_")
            fname = file.filename or f"upload_{uuid.uuid4().hex}.mp4"
            path = os.path.join(tmpdir, fname)
            file.save(path)
            
            log.info(f"Processing: {path}")
            cvp = CVPipeline()
            events = cvp.process_video(path)
            log.info(f"Events detected: {len(events)}")
            
            return jsonify({"video_name": os.path.basename(path), "events": events})
        
        app.add_url_rule("/pipeline/upload_cv", view_func=upload_cv, methods=["POST"])
        log.info("Registered /pipeline/upload_cv (Flask)")
        return
    
    raise RuntimeError("Unsupported app type")
