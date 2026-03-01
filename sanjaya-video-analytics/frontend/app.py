from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import requests
import os
import json

app = Flask(__name__)
CORS(app)

BACKEND_URL = "http://localhost:5000"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/videos')
def list_videos():
    """List all uploaded videos from backend"""
    try:
        uploads_dir = "../backend/uploads"
        if os.path.exists(uploads_dir):
            videos = [f for f in os.listdir(uploads_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
            return jsonify({'videos': videos})
    except Exception as e:
        print(f"Error listing videos: {e}")
    return jsonify({'videos': []})

@app.route('/api/stats')
def get_stats():
    """Get stats from backend"""
    video = request.args.get('video', '')
    if not video:
        return jsonify({'error': 'Video required'}), 400
    
    try:
        # Try to get from backend events.json
        events_path = f"../backend/json_outputs/events.json"
        if os.path.exists(events_path):
            with open(events_path, 'r') as f:
                events = json.load(f)
            
            # Calculate stats
            persons = len(set(e.get('track_id') for e in events if e.get('object') == 'person'))
            objects = len(events)
            object_classes = len(set(e.get('object') for e in events))
            activities = len(set(e.get('event') for e in events))
            
            top_objects = {}
            for e in events:
                obj = e.get('object', 'unknown')
                top_objects[obj] = top_objects.get(obj, 0) + 1
            
            top_objects_list = sorted([{'class': k, 'count': v} for k, v in top_objects.items()], key=lambda x: x['count'], reverse=True)
            
            return jsonify({
                'totals': {
                    'persons': persons,
                    'objects': objects,
                    'object_classes': object_classes,
                    'activities': activities,
                    'frames': len(events)
                },
                'top_objects': top_objects_list[:5],
                'activities': list(set(e.get('event') for e in events if e.get('event')))
            })
    except Exception as e:
        print(f"Error getting stats: {e}")
    
    return jsonify({
        'totals': {'persons': 0, 'objects': 0, 'object_classes': 0, 'activities': 0, 'frames': 0},
        'top_objects': [],
        'activities': []
    })

@app.route('/api/frames')
def get_frames():
    """Get sample frames from backend"""
    try:
        frames_dir = "../backend/static/frames"
        if os.path.exists(frames_dir):
            frames = []
            for img in sorted(os.listdir(frames_dir))[:3]:
                if img.endswith(('.jpg', '.png')):
                    frames.append({
                        'image': f'/static/frames/{img}',
                        'objects': [],
                        'persons': []
                    })
            return jsonify({'frames': frames})
    except Exception as e:
        print(f"Error getting frames: {e}")
    
    return jsonify({'frames': []})

@app.route('/api/heatmap')
def get_heatmap():
    """Get heatmap data from events"""
    try:
        events_path = "../backend/json_outputs/events.json"
        if os.path.exists(events_path):
            with open(events_path, 'r') as f:
                events = json.load(f)
            
            zones = {}
            for e in events:
                zone = e.get('zone', 'unknown')
                zones[zone] = zones.get(zone, 0) + 1
            
            # Map to 3x3 grid
            zone_map = {}
            zone_list = sorted(zones.keys())
            for i, z in enumerate(zone_list[:9]):
                zone_map[f'Z{i+1}'] = zones[z]
            
            return jsonify({'zones': zone_map})
    except Exception as e:
        print(f"Error getting heatmap: {e}")
    
    return jsonify({'zones': {}})

@app.route('/api/graph')
def get_graph():
    """Get knowledge graph from backend"""
    try:
        events_path = "../backend/json_outputs/events.json"
        if os.path.exists(events_path):
            with open(events_path, 'r') as f:
                events = json.load(f)
            
            nodes = []
            edges = []
            seen_ids = set()
            
            for e in events:
                track_id = e.get('track_id')
                obj = e.get('object', 'unknown')
                event = e.get('event', 'activity')
                
                if track_id and track_id not in seen_ids:
                    nodes.append({
                        'data': {
                            'id': f'track_{track_id}',
                            'label': f'{obj} #{track_id}',
                            'type': 'Human' if obj == 'person' else 'Object'
                        }
                    })
                    seen_ids.add(track_id)
                
                # Add event node
                event_id = f'event_{len(edges)}'
                nodes.append({
                    'data': {'id': event_id, 'label': event, 'type': 'Event'}
                })
                
                if track_id:
                    edges.append({
                        'data': {'source': f'track_{track_id}', 'target': event_id, 'label': 'performed'}
                    })
            
            return jsonify({'nodes': nodes[:20], 'edges': edges[:20]})  # Limit for performance
    except Exception as e:
        print(f"Error getting graph: {e}")
    
    return jsonify({'nodes': [], 'edges': []})

@app.route('/rag/ask', methods=['POST'])
def rag_ask():
    """Ask RAG question - forward to backend"""
    data = request.json
    question = data.get('question', '')
    video_name = data.get('video_name', '')
    mode = data.get('mode', 'graph')
    
    if not question:
        return jsonify({'error': 'Question required'}), 400
    
    try:
        # Forward to backend RAG endpoint
        response = requests.post(f"{BACKEND_URL}/rag/ask", json=data, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            return jsonify({'answer': 'Backend error', 'insights': [], 'evidence': []})
    except Exception as e:
        print(f"RAG error: {e}")
        return jsonify({'answer': f'Error: {str(e)}', 'insights': [], 'evidence': []})

@app.route("/api/telegram/status")
def telegram_status():
    """Check if Telegram bot is configured."""
    is_active = telegram_notifier.bot_token != "8505865943:AAHkhf9i2rCBD250TbzeL3RW3JlwksR7J40"
    return jsonify({"active": is_active})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
