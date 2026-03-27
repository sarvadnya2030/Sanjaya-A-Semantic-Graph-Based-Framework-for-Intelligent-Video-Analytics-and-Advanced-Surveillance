"""
app_demo.py — Lightweight demo server for UI testing (Option A).
Serves the real dashboard.html with mock pipeline data.
No Neo4j, no Ollama, no YOLO, no MediaPipe required.

Run:
    python app_demo.py
Then open: http://localhost:5000
"""
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_socketio import SocketIO
import os, json, time, threading
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'sanjaya-demo'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ── Directories (created if missing) ─────────────────────────────────────────
for d in ["uploads", "static/frames", "json_outputs"]:
    os.makedirs(d, exist_ok=True)

# ── Mock data ─────────────────────────────────────────────────────────────────
MOCK_FRAMES = [
    {
        "frame_id": "frame_042",
        "timestamp": 1.4,
        "score": 0.91,
        "url": "/static/demo/frame_042.jpg",
        "zones": {"Z1": 3, "Z2": 1, "Z5": 2},
        "persons": [
            {"track_id": 1, "bbox": [120, 80, 280, 420], "posture": "standing",
             "motion_state": "walking", "speed_px_s": 38.2, "zone": "Z2"}
        ],
        "objects": [
            {"class": "backpack", "confidence": 0.87, "bbox": [130, 300, 220, 410], "zone": "Z2"}
        ]
    },
    {
        "frame_id": "frame_107",
        "timestamp": 3.6,
        "score": 0.78,
        "url": "/static/demo/frame_107.jpg",
        "zones": {"Z4": 4, "Z5": 2},
        "persons": [
            {"track_id": 2, "bbox": [340, 100, 480, 390], "posture": "crouching",
             "motion_state": "stationary", "speed_px_s": 2.1, "zone": "Z5"}
        ],
        "objects": []
    },
    {
        "frame_id": "frame_198",
        "timestamp": 6.6,
        "score": 0.85,
        "url": "/static/demo/frame_198.jpg",
        "zones": {"Z2": 2, "Z3": 5},
        "persons": [
            {"track_id": 1, "bbox": [200, 60, 360, 400], "posture": "standing",
             "motion_state": "running", "speed_px_s": 112.5, "zone": "Z3"},
            {"track_id": 3, "bbox": [420, 90, 560, 380], "posture": "standing",
             "motion_state": "walking", "speed_px_s": 42.0, "zone": "Z3"}
        ],
        "objects": [
            {"class": "laptop", "confidence": 0.79, "bbox": [430, 250, 550, 340], "zone": "Z3"}
        ]
    }
]

MOCK_VLM = [
    {
        "frame_id": "frame_042",
        "timestamp": 1.4,
        "description": "A person carrying a backpack walks through the corridor near the entrance zone.",
        "persons": [{"id": "P1", "action": "walking", "posture": "upright"}],
        "objects": ["backpack", "door"],
        "knowledge_graph": {
            "nodes": [
                {"id": "P1", "label": "Person", "type": "person"},
                {"id": "O1", "label": "Backpack", "type": "object"},
                {"id": "A1", "label": "Walking", "type": "action"}
            ],
            "edges": [
                {"source": "P1", "target": "A1", "label": "performs"},
                {"source": "P1", "target": "O1", "label": "carries"}
            ]
        }
    },
    {
        "frame_id": "frame_107",
        "timestamp": 3.6,
        "description": "Individual crouching near restricted zone — possible loitering behaviour.",
        "persons": [{"id": "P2", "action": "crouching", "posture": "crouched"}],
        "objects": [],
        "knowledge_graph": {
            "nodes": [
                {"id": "P2", "label": "Person", "type": "person"},
                {"id": "Z5", "label": "Restricted Zone", "type": "zone"},
                {"id": "A2", "label": "Loitering", "type": "action"}
            ],
            "edges": [
                {"source": "P2", "target": "A2", "label": "suspected"},
                {"source": "P2", "target": "Z5", "label": "near"}
            ]
        }
    },
    {
        "frame_id": "frame_198",
        "timestamp": 6.6,
        "description": "Two individuals — one running, one walking — converging near the server room entrance.",
        "persons": [
            {"id": "P1", "action": "running", "posture": "upright"},
            {"id": "P3", "action": "walking", "posture": "upright"}
        ],
        "objects": ["laptop"],
        "knowledge_graph": {
            "nodes": [
                {"id": "P1", "label": "Person 1", "type": "person"},
                {"id": "P3", "label": "Person 3", "type": "person"},
                {"id": "O2", "label": "Laptop",   "type": "object"},
                {"id": "A3", "label": "Running",  "type": "action"},
                {"id": "A4", "label": "Walking",  "type": "action"},
                {"id": "L1", "label": "Server Room Entrance", "type": "location"}
            ],
            "edges": [
                {"source": "P1", "target": "A3", "label": "performs"},
                {"source": "P3", "target": "A4", "label": "performs"},
                {"source": "P3", "target": "O2", "label": "carries"},
                {"source": "P1", "target": "L1", "label": "heading_to"},
                {"source": "P3", "target": "L1", "label": "heading_to"}
            ]
        }
    }
]

MOCK_INSIGHTS = {
    "overall_risk": "MEDIUM",
    "detected_objects": ["backpack", "laptop"],
    "posture_summary": {"standing": 4, "crouching": 1},
    "risks": [
        {"type": "LOITERING", "description": "Person P2 stationary in restricted zone for 45s", "severity": "HIGH"},
        {"type": "ANOMALY",   "description": "Rapid speed change detected — P1 accelerating to 112px/s", "severity": "MEDIUM"}
    ],
    "anomalies": [
        {"type": "TAILGATING", "description": "Two persons entering server room in close succession", "severity": "LOW"}
    ]
}

MOCK_SESSION = {
    "video_id": "demo_session_001",
    "filename": "demo_video.mp4",
    "timestamp": datetime.now().isoformat(),
    "frame_count": len(MOCK_FRAMES),
    "event_count": 7,
    "risk_level": "medium",
    "detected_objects": ["backpack", "laptop"],
    "posture_summary": {"standing": 4, "crouching": 1}
}

# Pre-populate sessions.json with one demo session
SESSIONS_FILE = "json_outputs/sessions.json"
if not os.path.exists(SESSIONS_FILE):
    with open(SESSIONS_FILE, 'w') as f:
        json.dump([MOCK_SESSION], f, indent=2)

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/api/videos")
def list_videos():
    videos = [f for f in os.listdir("uploads") if f.endswith(('.mp4', '.avi', '.mov'))]
    return jsonify({"videos": videos})

@app.route("/api/sessions")
def list_sessions():
    try:
        with open(SESSIONS_FILE) as f:
            sessions = json.load(f)
    except Exception:
        sessions = [MOCK_SESSION]
    return jsonify({"sessions": sessions})

@app.route("/api/session/<path:video_id>")
def get_session(video_id):
    try:
        with open(SESSIONS_FILE) as f:
            sessions = json.load(f)
        for s in sessions:
            if s.get("video_id") == video_id or s.get("filename") == video_id:
                return jsonify(s)
    except Exception:
        pass
    return jsonify(MOCK_SESSION)

@app.route("/api/export/report")
def export_report():
    fmt = request.args.get("format", "json")
    if fmt == "json":
        return jsonify({
            "sessions": [MOCK_SESSION],
            "frames": MOCK_FRAMES,
            "insights": MOCK_INSIGHTS
        })
    # CSV stub
    from flask import Response
    csv_data = "frame_id,timestamp,score\n"
    for f in MOCK_FRAMES:
        csv_data += f"{f['frame_id']},{f['timestamp']},{f['score']}\n"
    return Response(csv_data, mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=report.csv"})

# ── Mock pipeline (streams SocketIO progress, returns mock data) ──────────────
def _run_mock_pipeline(sid):
    stages = [
        ("upload",      5,  "Saved demo_video.mp4"),
        ("cleanup",     8,  "Cleared previous outputs"),
        ("cv_pipeline", 15, "Starting motion gating + detection…"),
        ("cv_pipeline", 30, "YOLO detection running…"),
        ("cv_pipeline", 45, "3 events · 3 salient frames"),
        ("vlm",         55, "Analysing frame_042 with VLM…"),
        ("vlm",         65, "Analysing frame_107 with VLM…"),
        ("vlm",         75, "Analysing frame_198 with VLM…"),
        ("neo4j",       82, "Pushing knowledge graph to Neo4j…"),
        ("rag",         90, "Rebuilding RAG index…"),
        ("done",       100, "Pipeline complete ✓"),
    ]
    for stage, pct, msg in stages:
        time.sleep(0.6)
        socketio.emit('progress', {'stage': stage, 'pct': pct, 'msg': msg}, room=sid)

@app.route("/pipeline/upload", methods=["POST"])
def pipeline_upload():
    sid = request.args.get("sid") or request.headers.get("X-Socket-ID", "")
    # Stream fake progress in background
    t = threading.Thread(target=_run_mock_pipeline, args=(sid,), daemon=True)
    t.start()
    t.join()  # wait so response comes after progress finishes

    return jsonify({
        "salient_frames": MOCK_FRAMES,
        "vlm_results":    MOCK_VLM,
        "cv_events":      7,
        "insights":       MOCK_INSIGHTS,
        "video_id":       f"demo_{datetime.now().strftime('%H%M%S')}",
        "filename":       request.files['file'].filename if 'file' in request.files else "demo.mp4"
    })

# ── RAG endpoints ─────────────────────────────────────────────────────────────
@app.route("/rag/ask", methods=["POST"])
def rag_ask():
    body     = request.get_json(force=True, silent=True) or {}
    question = body.get("question", "").strip()
    if not question:
        return jsonify({"error": "empty question"}), 400
    return jsonify({
        "answer":     f"[DEMO MODE] Based on the surveillance footage, {question.lower().rstrip('?')} "
                      f"shows 3 persons detected — one loitering in Zone 5 and two converging near the server room entrance.",
        "evidence":   ["frame_042: Person walking with backpack",
                       "frame_107: Crouching near restricted zone",
                       "frame_198: Two persons running toward server room"],
        "confidence": 0.82,
        "insights":   ["Loitering detected in Z5", "Rapid speed change P1", "Tailgating risk at entrance"]
    })

@app.route("/rag/search")
def rag_search():
    q = request.args.get("q", "")
    return jsonify({"results": [{"text": f"[DEMO] Result for: {q}", "score": 0.9}]})

@app.route("/rag/graph")
def rag_graph():
    q = request.args.get("q", "")
    return jsonify({"answer": f"[DEMO] Graph RAG result for: {q}", "graph_facts": []})

@app.route("/rag/hybrid")
def rag_hybrid():
    q = request.args.get("q", "")
    return jsonify({"answer": f"[DEMO] Hybrid RAG result for: {q}", "evidence": []})

@app.route("/rag/status")
def rag_status():
    return jsonify({"status": "ok", "mode": "demo"})

@app.route("/rag/rebuild", methods=["POST"])
def rag_rebuild():
    return jsonify({"status": "ok", "message": "RAG rebuild skipped in demo mode"})

# ── SocketIO events ───────────────────────────────────────────────────────────
@socketio.on('connect')
def on_connect():
    print(f"[SocketIO] Client connected: {request.sid}")

@socketio.on('disconnect')
def on_disconnect():
    print(f"[SocketIO] Client disconnected: {request.sid}")

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  SANJAYA — Demo Mode (UI test, no heavy deps)")
    print("  Dashboard: http://localhost:5000")
    print("  No Neo4j / Ollama / YOLO required")
    print("=" * 60)
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
