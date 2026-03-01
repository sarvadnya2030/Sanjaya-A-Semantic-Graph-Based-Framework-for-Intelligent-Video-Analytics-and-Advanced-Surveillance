from flask import Flask, request, jsonify, send_from_directory, render_template, url_for
from flask_cors import CORS
import os
import logging
import cv2
import json
import shutil
import sys

sys.path.insert(0, os.path.dirname(__file__))

# Import modules
from cv_pipeline.pipeline import CVPipeline
from modules.vlm_analyzer import analyze_salient_frame
from rag.json_rag import JsonRAG
from rag.graph_rag import GraphRAG
from modules.telegram_notifier import TelegramNotifier  # NEW
from modules.neo4j_kg import export_surveillance_graph, push_vlm_kg_to_neo4j, push_vlm_analysis_summary
from modules.neo4j_manager import ensure_neo4j

class Neo4jKG:
    def __init__(self):
        self.uri = "bolt://localhost:7687"
        self.auth = ("neo4j", "neo4j123")
        ensure_neo4j()
    
    def push_cv_events(self, events):
        events_path = "json_outputs/cv_events_temp.json"
        with open(events_path, 'w') as f:
            json.dump(events, f)
        return export_surveillance_graph(self.uri, self.auth, events_path, "current_video")
    
    def push_vlm_kg(self, frame_id, kg_data):
        return push_vlm_kg_to_neo4j(self.uri, self.auth, kg_data, "current_video", frame_id)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger("__main__")

app = Flask(__name__)
CORS(app)

UPLOAD_DIR = "uploads"
FRAMES_DIR = "static/frames"
JSON_DIR = "json_outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)

log.info("[INIT] Initializing pipeline components...")
cv_pipeline = CVPipeline()  # FIXED: No parameters needed
rag_engine = JsonRAG(json_dirs=[JSON_DIR])
graph_rag_engine = GraphRAG()
neo4j_kg = Neo4jKG()
log.info("[INIT] ✅ All components initialized")

# Initialize Telegram (add your credentials)
telegram_notifier = TelegramNotifier(
    bot_token=os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN"),
    chat_id=os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID")
)

def draw_yolo_annotations(frame, persons, objects_list):
    """Draw YOLO bounding boxes on frame with detailed labels."""
    annotated = frame.copy()
    
    # Draw person boxes in GREEN
    for p in persons:
        bbox = p.get('bbox', [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Label with track ID, speed, posture
            track_id = p.get('track_id', '?')
            speed = p.get('speed_px_s', 0)
            posture = p.get('posture', 'unknown')
            motion = p.get('motion_state', 'unknown')
            zone = p.get('zone', '?')
            
            label = f"P{track_id} Z{zone} {speed:.0f}px/s {motion} {posture}"
            
            # Background for text
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - h - 4), (x1 + w, y1), (0, 255, 0), -1)
            cv2.putText(annotated, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Draw object boxes in BLUE
    for o in objects_list:
        bbox = o.get('bbox', [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Label with class, confidence, zone
            obj_class = o.get('class', 'object')
            conf = o.get('confidence', 0)
            zone = o.get('zone', '?')
            
            label = f"{obj_class} Z{zone} {conf:.2f}"
            
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - h - 4), (x1 + w, y1), (255, 0, 0), -1)
            cv2.putText(annotated, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return annotated

@app.route("/")
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/test_events")
def test_events():
    return render_template("test_events.html")

@app.route("/api/videos")
def list_videos():
    try:
        videos = [f for f in os.listdir(UPLOAD_DIR) if f.endswith(('.mp4', '.avi', '.mov'))]
        return jsonify({"videos": videos})
    except:
        return jsonify({"videos": []})

@app.route("/pipeline/upload", methods=["POST"])
def pipeline_upload():
    try:
        # 1. SAVE VIDEO
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        video_path = os.path.join(UPLOAD_DIR, file.filename)
        file.save(video_path)
        log.info(f"[UPLOAD] ✅ Saved: {video_path}")
        
        # 2. CLEAR OLD DATA
        # CLEANUP OLD JSONs
        if os.path.exists(JSON_DIR):
            for f in os.listdir(JSON_DIR):
                if f.endswith('.json'):
                    os.remove(os.path.join(JSON_DIR, f))
        os.makedirs(JSON_DIR, exist_ok=True)
        
        if os.path.exists(FRAMES_DIR):
            for f in os.listdir(FRAMES_DIR):
                if f.endswith('.jpg'):
                    os.remove(os.path.join(FRAMES_DIR, f))
        os.makedirs(FRAMES_DIR, exist_ok=True)
        
        log.info("[CLEANUP] ✅ Cleared old data")

        # 3. RUN CV PIPELINE
        log.info("[CV] Starting CV pipeline...")
        cv_events, salient_frames_data = cv_pipeline.process_video(
            video_path=video_path,
            output_dir=JSON_DIR
        )
        log.info(f"[CV] ✅ Detected {len(cv_events)} events, {len(salient_frames_data)} salient frames")
        
        # FALLBACK: If no salient frames, use middle frame
        if len(salient_frames_data) == 0:
            log.warning("[CV] ⚠️ No salient frames detected! Using middle frame fallback...")
            
            # Read video to get middle frame
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            middle_idx = total_frames // 2
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_idx)
            ret, middle_frame = cap.read()
            cap.release()
            
            if ret and middle_frame is not None:
                # Create minimal salient frame data
                salient_frames_data = [(
                    middle_idx,
                    0.5,  # saliency score
                    middle_frame,
                    {
                        'frame_id': middle_idx,
                        'timestamp': middle_idx / fps if fps > 0 else 0,
                        'saliency': 0.5,
                        'persons': [],
                        'objects': [],
                        'all_objects': [],
                        'zones': {}
                    }
                )]
                log.info(f"[CV] ✅ Created fallback frame {middle_idx} at {middle_idx/fps:.1f}s")
            else:
                log.error("[CV] ❌ Could not read middle frame!")
        
        # 4. PREPARE TOP 3 FRAMES WITH YOLO ANNOTATIONS
        salient_frames = []
        top_3_frames = salient_frames_data[:3]  # LIMIT TO 3!
        
        for idx, (frame_id, saliency_score, frame_img, cv_metadata) in enumerate(top_3_frames):
            frame_filename = f"salient_{idx}_frame{frame_id}.jpg"
            frame_path = os.path.join(FRAMES_DIR, frame_filename)
            
            # Draw YOLO annotations on frame
            persons = cv_metadata.get('persons', [])
            objects_list = cv_metadata.get('objects', [])
            
            if frame_img is not None:
                annotated_frame = draw_yolo_annotations(frame_img, persons, objects_list)
                cv2.imwrite(frame_path, annotated_frame)
                log.info(f"[CV] ✅ Annotated frame {idx+1}/3: frame_id={frame_id}, {len(persons)} persons, {len(objects_list)} objects")
            
            cv_metadata["image_path"] = os.path.abspath(frame_path)
            cv_metadata["image_url"] = url_for('static', filename=f'frames/{frame_filename}')
            
            salient_frames.append(cv_metadata)
        
        # 5. VLM ANALYSIS ON TOP 3 FRAMES
        log.info("[VLM] Starting VLM analysis on 3 frames...")
        vlm_results = []
        
        for idx, frame_meta in enumerate(salient_frames[:3]):
            frame_id = frame_meta['frame_id']
            frame_path = frame_meta['image_path']
            timestamp = frame_meta.get('timestamp', 0)
            
            log.info(f"[VLM] Analyzing frame {idx+1}/3 (frame_id={frame_id})")
            
            try:
                # PREPARE CV DETECTIONS FOR VLM CONTEXT
                cv_detections = {
                    "persons": frame_meta.get("persons", []),
                    "objects": frame_meta.get("objects", [])
                }
                
                # GET VLM DESCRIPTION WITH CV CONTEXT
                vlm_result = analyze_salient_frame(
                    image_path=frame_path,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    cv_detections=cv_detections,
                    ollama_url="http://localhost:11434"
                )
                
                vlm_results.append(vlm_result)
                
                # SAVE 1: VLM-ONLY JSON
                vlm_path = os.path.join(JSON_DIR, f"frame_{frame_id}_vlm.json")
                with open(vlm_path, 'w') as f:
                    json.dump(vlm_result, f, indent=2)
                log.info(f"[VLM] 💾 Saved: {vlm_path}")
                
                # SAVE 2: CV-ONLY JSON
                cv_data = {
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "image_path": frame_path,
                    "persons": frame_meta.get("persons", []),
                    "objects": frame_meta.get("objects", []),
                    "zones": frame_meta.get("zones", {}),
                    "motion_magnitude": frame_meta.get("motion_magnitude", 0)
                }
                cv_path = os.path.join(JSON_DIR, f"frame_{frame_id}_cv.json")
                with open(cv_path, 'w') as f:
                    json.dump(cv_data, f, indent=2)
                log.info(f"[CV] 💾 Saved: {cv_path}")
                
            except Exception as e:
                log.error(f"[VLM] Failed frame {frame_id}: {e}")
        
        log.info(f"[VLM] ✅ Completed {len(vlm_results)}/3 frames")
        
        # PUSH CV EVENTS TO NEO4J
        if cv_events:
            try:
                neo4j_kg.push_cv_events(cv_events)
                log.info(f"[Neo4j] ✅ Pushed {len(cv_events)} events")
            except Exception as e:
                log.error(f"[Neo4j] CV events push failed: {e}")
        
        # PUSH VLM DATA TO NEO4J (Research-grade interconnected KG)
        for vlm_result in vlm_results:
            try:
                frame_id = vlm_result.get('frame_id', 'unknown')
                
                # 1. Push knowledge graph nodes & relationships
                if 'knowledge_graph' in vlm_result and vlm_result['knowledge_graph']:
                    kg_data = vlm_result['knowledge_graph']
                    neo4j_kg.push_vlm_kg(frame_id, kg_data)
                    log.info(f"[Neo4j] ✅ Pushed KG for frame {frame_id}")
                
                # 2. Push analysis summary with risks, anomalies, and interactions
                push_vlm_analysis_summary(neo4j_kg.uri, neo4j_kg.auth, vlm_result, "current_video", frame_id)
                log.info(f"[Neo4j] ✅ Pushed analysis summary for frame {frame_id}")
                
            except Exception as e:
                log.error(f"[Neo4j] VLM push failed for frame {vlm_result.get('frame_id')}: {e}")
        
        # BUILD RAG INDEX
        log.info("[RAG] Building index...")
        rag_engine.build_index()
        log.info(f"[RAG] ✅ Indexed {len(rag_engine.documents)} documents")

        # AGGREGATE INSIGHTS FOR DASHBOARD
        all_risks = []
        all_anomalies = []
        all_objects = set()
        
        for vlm in vlm_results:
            risks = vlm.get("risks", [])
            all_risks.extend(risks)
            
            anomalies = vlm.get("anomalies", [])
            all_anomalies.extend(anomalies)
        
        # Get unique objects from CV detections
        for frame in salient_frames[:3]:
            for obj in frame.get("objects", []):
                obj_class = obj.get("class", "")
                if obj_class:
                    all_objects.add(obj_class)
        
        # Calculate overall risk level
        risk_levels = [r.get("severity", "low") for r in all_risks]
        high_risk_count = risk_levels.count("high")
        medium_risk_count = risk_levels.count("medium")
        
        if high_risk_count > 0:
            overall_risk = "high"
        elif medium_risk_count > 0:
            overall_risk = "medium"
        else:
            overall_risk = "low"

        # 8. TELEGRAM NOTIFICATIONS (DISABLED)
        # log.info("[Telegram] Sending event summary...")
        # telegram_notifier.send_event_summary(
        #     events=cv_events[:10],
        #     salient_frames=[
        #         frame_meta['image_path'] for frame_meta in salient_frames[:3]
        #     ]
        # )
        # log.info("[Telegram] ✅ Event summary sent")
        log.info("[Telegram] DISABLED - skipping notifications")

        return jsonify({
            "status": "success",
            "salient_frames": salient_frames,
            "vlm_results": vlm_results,
            "cv_events": len(cv_events),
            "insights": {
                "risks": all_risks,
                "anomalies": all_anomalies,
                "overall_risk": overall_risk,
                "detected_objects": list(all_objects)
            },
            "telegram_sent": True
        })
        
    except Exception as e:
        log.error(f"[PIPELINE] Error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/rag/search")
def rag_search():
    """JSON RAG search (existing)."""
    q = request.args.get("q", "").strip()
    k = int(request.args.get("k", 5))
    
    if not q:
        return jsonify({"error": "empty query"}), 400
    
    try:
        result = rag_engine.ask(q, k=k)
        return jsonify({
            "answer": result["answer"],
            "evidence": result["evidence"],
            "confidence": result["confidence"],
            "sources": result["sources"],
            "type": "json_rag"
        })
    except Exception as e:
        log.error(f"[RAG] Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/rag/graph")
def graph_rag_search():
    """GraphRAG with chain-of-thought reasoning."""
    q = request.args.get("q", "").strip()
    
    if not q:
        return jsonify({"error": "empty query"}), 400
    
    try:
        log.info(f"[GraphRAG] Query: {q}")
        result = graph_rag_engine.ask(q)
        
        return jsonify({
            "answer": result.get("answer"),
            "chain_of_thought": result.get("chain_of_thought", []),
            "evidence": result.get("evidence", []),
            "confidence": result.get("confidence", 0.0),
            "reasoning_path": result.get("reasoning_path", ""),
            "graph_facts": result.get("graph_facts", []),
            "type": "graph_rag"
        })
    except Exception as e:
        log.error(f"[GraphRAG] Error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/rag/hybrid")
def hybrid_rag_search():
    """Hybrid: GraphRAG + JSON RAG combined."""
    q = request.args.get("q", "").strip()
    k = int(request.args.get("k", 3))
    
    if not q:
        return jsonify({"error": "empty query"}), 400
    
    try:
        # Get both results
        graph_result = graph_rag_engine.ask(q)
        json_result = rag_engine.ask(q, k=k)
        
        # Combine insights
        combined_answer = f"""
**Graph Reasoning:** {graph_result.get('answer')}

**JSON Evidence:** {json_result["answer"]}
"""
        
        # Parse confidence
        json_conf_str = json_result["confidence"].rstrip('%')
        json_conf = float(json_conf_str) / 100 if json_conf_str else 0.0
        
        return jsonify({
            "answer": combined_answer.strip(),
            "graph_reasoning": {
                "chain_of_thought": graph_result.get("chain_of_thought", []),
                "reasoning_path": graph_result.get("reasoning_path", "")
            },
            "json_evidence": {
                "evidence": json_result["evidence"],
                "sources": json_result["sources"]
            },
            "confidence": (graph_result.get("confidence", 0) + json_conf) / 2,
            "type": "hybrid_rag"
        })
    except Exception as e:
        log.error(f"[Hybrid RAG] Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/static/<path:path>")
def serve_static(path):
    return send_from_directory("static", path)

@app.route("/json_outputs/<path:filename>")
def serve_json_outputs(filename):
    """Serve JSON output files for frontend analysis."""
    return send_from_directory("json_outputs", filename)

@app.route("/api/telegram/status")
def telegram_status():
    """Telegram notifications disabled."""
    return jsonify({"active": False, "message": "Telegram disabled"})

@app.route("/rag/status")
def rag_status():
    """Check RAG index status."""
    status = {
        "index_built": rag_engine.index is not None,
        "document_count": len(rag_engine.documents),
        "json_dirs": rag_engine.json_dirs,
        "json_files": []
    }
    
    # Check what files exist
    for json_dir in rag_engine.json_dirs:
        if os.path.exists(json_dir):
            files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
            status["json_files"].extend(files)
    
    return jsonify(status)

@app.route("/rag/rebuild", methods=["POST"])
def rebuild_rag():
    """Manually rebuild RAG index."""
    try:
        log.info("[RAG] 🔨 Manual rebuild requested...")
        rag_engine.build_index()
        
        return jsonify({
            "status": "success",
            "document_count": len(rag_engine.documents),
            "index_ready": rag_engine.index is not None
        })
    except Exception as e:
        log.error(f"[RAG] Rebuild failed: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
