from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import requests
import os
import json
import glob

app = Flask(__name__)
CORS(app)

BACKEND_URL = "http://localhost:5000"
JSON_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend/json_outputs"))
UPLOADS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend/uploads"))
FRAMES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend/static/frames"))


def _load_vlm_data():
    """Load all VLM JSON files from json_outputs."""
    results = []
    for path in sorted(glob.glob(os.path.join(JSON_DIR, "*_vlm.json"))):
        try:
            with open(path) as f:
                results.append(json.load(f))
        except Exception:
            pass
    return results


def _load_events():
    """Load events.json; returns empty list on failure."""
    try:
        path = os.path.join(JSON_DIR, "events.json")
        with open(path) as f:
            return json.load(f)
    except Exception:
        return []


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.route('/api/videos')
def list_videos():
    """List all uploaded videos."""
    try:
        if os.path.exists(UPLOADS_DIR):
            videos = sorted([
                f for f in os.listdir(UPLOADS_DIR)
                if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
            ])
            return jsonify({'videos': videos})
    except Exception as e:
        print(f"Error listing videos: {e}")
    return jsonify({'videos': []})


@app.route('/api/stats')
def get_stats():
    """Get stats — fuses CV events + VLM data."""
    events = _load_events()
    vlm_data = _load_vlm_data()

    # Persons: from CV events first, then fall back to VLM
    cv_persons = len(set(e.get('track_id') for e in events if e.get('object') == 'person'))
    vlm_persons = sum(len(v.get('persons', [])) for v in vlm_data)
    total_persons = cv_persons or vlm_persons

    # Objects: aggregate across VLM frames
    obj_counts = {}
    for v in vlm_data:
        for o in v.get('objects', []):
            t = o.get('type', o.get('class', 'unknown'))
            obj_counts[t] = obj_counts.get(t, 0) + 1
    # Also from CV events
    for e in events:
        obj = e.get('object', '')
        if obj and obj != 'person':
            obj_counts[obj] = obj_counts.get(obj, 0) + 1

    # Activities from VLM persons
    activities = set()
    for v in vlm_data:
        for p in v.get('persons', []):
            act = p.get('action', '')
            if act:
                activities.add(act[:40])
    for e in events:
        ev = e.get('event', '')
        if ev:
            activities.add(ev)

    top_objects = sorted(
        [{'class': k, 'count': c} for k, c in obj_counts.items()],
        key=lambda x: x['count'], reverse=True
    )
    top_obj = top_objects[0]['class'] if top_objects else '–'

    return jsonify({
        'totals': {
            'persons': total_persons,
            'objects': sum(obj_counts.values()),
            'object_classes': len(obj_counts),
            'activities': len(activities),
            'frames': len(vlm_data) or len(set(e.get('frame_id') for e in events))
        },
        'top_objects': top_objects[:5],
        'top_object': top_obj,
        'activities': list(activities)[:10]
    })


@app.route('/api/frames')
def get_frames():
    """Return frame image URLs pointing to backend (port 5000)."""
    try:
        if os.path.exists(FRAMES_DIR):
            imgs = sorted([
                f for f in os.listdir(FRAMES_DIR)
                if f.lower().endswith(('.jpg', '.png'))
            ])[:3]

            # Load per-frame CV metadata for bbox overlay
            frames = []
            for img in imgs:
                # Parse frame number from filename e.g. salient_0_frame82.jpg
                frame_id = None
                parts = img.replace('.jpg', '').replace('.png', '').split('_')
                for p in parts:
                    if p.isdigit():
                        frame_id = int(p)

                persons, objects = [], []
                if frame_id is not None:
                    cv_path = os.path.join(JSON_DIR, f"frame_{frame_id}_cv.json")
                    vlm_path = os.path.join(JSON_DIR, f"frame_{frame_id}_vlm.json")
                    if os.path.exists(cv_path):
                        try:
                            with open(cv_path) as f:
                                cv = json.load(f)
                            for p in cv.get('persons', []):
                                bbox = p.get('bbox', [])
                                persons.append({
                                    'id': f"T{p.get('track_id','?')}",
                                    'bbox': bbox,
                                    'posture': p.get('posture', '')
                                })
                            for o in cv.get('objects', []):
                                objects.append({
                                    'class': o.get('class', '?'),
                                    'bbox': o.get('bbox', [])
                                })
                        except Exception:
                            pass
                    if not persons and os.path.exists(vlm_path):
                        try:
                            with open(vlm_path) as f:
                                vlm = json.load(f)
                            for i, p in enumerate(vlm.get('persons', [])[:6]):
                                persons.append({
                                    'id': p.get('id', f'P{i+1}'),
                                    'bbox': [],   # VLM has no pixel coords
                                    'posture': p.get('posture', '')
                                })
                        except Exception:
                            pass

                frames.append({
                    'image': f'{BACKEND_URL}/static/frames/{img}',
                    'objects': objects,
                    'persons': persons
                })
            return jsonify({'frames': frames})
    except Exception as e:
        print(f"Error getting frames: {e}")
    return jsonify({'frames': []})


@app.route('/api/heatmap')
def get_heatmap():
    """Zone activity heatmap from events or VLM location text."""
    events = _load_events()
    zones = {}
    for e in events:
        z = e.get('zone', '')
        if z:
            zones[z] = zones.get(z, 0) + 1

    if not zones:
        # Synthesise from VLM person locations
        zone_keywords = {
            'Z1': ['top-left', 'left corner', 'upper left'],
            'Z2': ['top', 'upper center', 'top center'],
            'Z3': ['top-right', 'right corner', 'upper right'],
            'Z4': ['left', 'left side', 'mid-left'],
            'Z5': ['center', 'middle', 'booth', 'table'],
            'Z6': ['right', 'right side', 'mid-right'],
            'Z7': ['bottom-left', 'lower left'],
            'Z8': ['bottom', 'lower center', 'foreground'],
            'Z9': ['bottom-right', 'lower right', 'background'],
        }
        for v in _load_vlm_data():
            for p in v.get('persons', []):
                loc = (p.get('location', '') + ' ' + p.get('action', '')).lower()
                for z, kws in zone_keywords.items():
                    if any(kw in loc for kw in kws):
                        zones[z] = zones.get(z, 0) + 1
                        break
                else:
                    zones['Z5'] = zones.get('Z5', 0) + 1  # default to center

    return jsonify({'zones': zones})


@app.route('/api/graph')
def get_graph():
    """Knowledge graph — uses VLM data for rich nodes/edges."""
    vlm_data = _load_vlm_data()
    events = _load_events()
    nodes = []
    edges = []
    seen = set()

    if vlm_data:
        for vi, v in enumerate(vlm_data):
            frame_id = v.get('_metadata', {}).get('frame_id', f'f{vi}')

            for p in v.get('persons', []):
                pid = p.get('id', f'P{vi}')
                nid = f'person_{pid}'
                if nid not in seen:
                    nodes.append({'data': {'id': nid, 'label': pid, 'type': 'Human'}})
                    seen.add(nid)
                action = p.get('action', '')
                if action:
                    aid = f'act_{nid}_{vi}'
                    nodes.append({'data': {'id': aid, 'label': action[:25], 'type': 'Event'}})
                    edges.append({'data': {'source': nid, 'target': aid, 'label': 'performs'}})

            for oi, o in enumerate(v.get('objects', [])[:6]):
                otype = o.get('type', o.get('class', 'object'))
                oid = f'obj_{otype}_{vi}_{oi}'
                if oid not in seen:
                    nodes.append({'data': {'id': oid, 'label': otype, 'type': 'Object'}})
                    seen.add(oid)
                owner = o.get('owner', '')
                if owner and owner != 'not visible':
                    src = f'person_{owner}'
                    if src in seen:
                        edges.append({'data': {'source': src, 'target': oid, 'label': 'carries'}})

        return jsonify({'nodes': nodes[:30], 'edges': edges[:30]})

    # Fallback: CV events
    for e in events:
        tid = e.get('track_id')
        obj = e.get('object', 'unknown')
        event = e.get('event', 'activity')
        if tid and f'track_{tid}' not in seen:
            nodes.append({'data': {'id': f'track_{tid}', 'label': f'{obj} #{tid}',
                                   'type': 'Human' if obj == 'person' else 'Object'}})
            seen.add(f'track_{tid}')
        eid = f'event_{len(edges)}'
        nodes.append({'data': {'id': eid, 'label': event, 'type': 'Event'}})
        if tid:
            edges.append({'data': {'source': f'track_{tid}', 'target': eid, 'label': 'performed'}})

    return jsonify({'nodes': nodes[:20], 'edges': edges[:20]})


@app.route('/rag/ask', methods=['POST'])
def rag_ask():
    """Proxy RAG questions to backend."""
    data = request.json or {}
    if not data.get('question'):
        return jsonify({'error': 'Question required'}), 400
    try:
        resp = requests.post(f"{BACKEND_URL}/rag/ask", json=data, timeout=60)
        return jsonify(resp.json())
    except Exception as e:
        return jsonify({'answer': f'Error: {e}', 'insights': [], 'evidence': []})


@app.route('/rag/status')
def rag_status():
    """Proxy RAG status from backend."""
    try:
        resp = requests.get(f"{BACKEND_URL}/rag/status", timeout=5)
        return jsonify(resp.json())
    except Exception:
        return jsonify({'index_built': False, 'document_count': 0})


@app.route("/api/telegram/status")
def telegram_status():
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    return jsonify({"active": bool(token) and token != "YOUR_BOT_TOKEN"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
