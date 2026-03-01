import base64, json, requests, cv2, re
from config import OLLAMA_URL, VISION_MODEL

def _b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def _resize_image(img_path, max_size=640):
    """Resize image to max_size for faster VLM processing"""
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, (int(w*scale), int(h*scale)))
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode("utf-8")

def _aggressive_json_clean(s):
    """Aggressively clean and repair JSON"""
    # Remove control chars
    s = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', s)
    # Remove trailing commas before } or ]
    s = re.sub(r',\s*([}\]])', r'\1', s)
    # Fix unescaped quotes inside strings (heuristic)
    s = re.sub(r'("description":\s*")([^"]*)"([^,}\]]*)"', r'\1\2\3"', s)
    # Remove markdown
    s = s.strip()
    if s.startswith("```"):
        lines = s.split('\n')
        s = '\n'.join(lines[1:-1]) if len(lines) > 2 else s
    if s.startswith("json"):
        s = s[4:].strip()
    return s

def _extract_json_multi_strategy(resp):
    """Try multiple JSON extraction strategies"""
    if isinstance(resp, dict):
        return resp
    
    resp = str(resp).strip()
    
    # Strategy 1: Direct parse after cleanup
    try:
        cleaned = _aggressive_json_clean(resp)
        s = cleaned.find("{")
        e = cleaned.rfind("}")
        if s != -1 and e != -1 and e > s:
            return json.loads(cleaned[s:e+1])
    except:
        pass
    
    # Strategy 2: Extract largest JSON object
    try:
        import regex
        matches = regex.findall(r'\{(?:[^{}]|(?R))*\}', resp, regex.DOTALL)
        for match in sorted(matches, key=len, reverse=True):
            try:
                return json.loads(_aggressive_json_clean(match))
            except:
                continue
    except:
        pass
    
    # Strategy 3: Line-by-line reconstruction
    try:
        lines = resp.split('\n')
        json_lines = []
        in_json = False
        for line in lines:
            if '{' in line:
                in_json = True
            if in_json:
                json_lines.append(line)
            if '}' in line and in_json:
                break
        reconstructed = '\n'.join(json_lines)
        return json.loads(_aggressive_json_clean(reconstructed))
    except:
        pass
    
    raise ValueError("All JSON extraction strategies failed")

def _build_cv_graph_fallback(cv_grounding, video_name, frame_id):
    """Build minimal graph from CV data only"""
    nodes = []
    edges = []
    
    persons = cv_grounding.get("persons", [])
    objects = cv_grounding.get("objects", [])
    interactions = cv_grounding.get("interactions", [])
    groups = cv_grounding.get("groups", [])
    
    # Map CV person IDs to surveillance IDs
    person_map = {p["id"]: f"PERSON_{i+1:03d}" for i, p in enumerate(persons)}
    object_map = {o["id"]: f"OBJECT_{i+1:03d}" for i, o in enumerate(objects)}
    
    # Person nodes
    for p in persons:
        nodes.append({
            "node_id": person_map[p["id"]],
            "node_type": "Human",
            "cv_detection": f"{p['id']} (conf: {p['confidence']})",
            "properties": {
                "posture": p.get("posture", "unknown"),
                "clothing_color": p.get("clothing_color", "unknown"),
                "zone": p.get("zone_id", "unknown"),
                "face_detected": p.get("face_detected", False)
            }
        })
    
    # Object nodes
    for o in objects:
        nodes.append({
            "node_id": object_map[o["id"]],
            "node_type": "Object",
            "cv_detection": f"{o['id']}: {o['class']} (conf: {o['confidence']})",
            "properties": {
                "object_class": o["class"],
                "portable": o.get("portable", False),
                "zone": o.get("zone_id", "unknown")
            }
        })
    
    # Activity node
    if persons:
        nodes.append({
            "node_id": "ACTIVITY_001",
            "node_type": "Activity",
            "properties": {
                "activity_type": "collaborative_work" if len(persons) > 1 else "individual_activity",
                "participants": list(person_map.values()),
                "confidence": 0.7
            }
        })
        
        # Edges: persons perform activity
        for pid in person_map.values():
            edges.append({
                "source": pid,
                "relation": "PERFORMS",
                "target": "ACTIVITY_001",
                "confidence": 0.75
            })
    
    # Interaction edges
    for inter in interactions:
        pid = person_map.get(inter.get("person_id"))
        oid = object_map.get(inter.get("object_id"))
        if pid and oid:
            edges.append({
                "source": pid,
                "relation": "INTERACTS_WITH",
                "target": oid,
                "confidence": 0.8,
                "metadata": {"type": inter.get("type", "near")}
            })
    
    # Group nodes and edges
    for g in groups:
        group_id = g.get("group_id", "GROUP_001")
        nodes.append({
            "node_id": group_id,
            "node_type": "Group",
            "properties": {
                "members": [person_map.get(m, m) for m in g.get("members", [])],
                "group_size": g.get("size", 0),
                "cohesion": g.get("cohesion", "unknown")
            }
        })
        for member in g.get("members", []):
            pid = person_map.get(member)
            if pid:
                edges.append({
                    "source": pid,
                    "relation": "MEMBER_OF",
                    "target": group_id,
                    "confidence": 0.9
                })
    
    narrative = f"{len(persons)} person(s) detected"
    if objects:
        narrative += f" with {len(objects)} object(s)"
    if groups:
        narrative += f" forming {len(groups)} group(s)"
    narrative += ". CV-based analysis."
    
    return {
        "surveillance_narrative": narrative,
        "detailed_scene_analysis": {
            "environment": "unknown",
            "lighting_condition": "unknown",
            "activity_level": "moderate" if len(persons) > 1 else "low"
        },
        "nodes": nodes,
        "edges": edges,
        "video_name": video_name,
        "frame_id": str(frame_id)
    }

SIMPLE_PROMPT = """Analyze image with CV data. Return valid JSON only.

CV: {cv_summary}

JSON format:
{{
  "surveillance_narrative": "Short description",
  "nodes": [{{"node_id": "PERSON_001", "node_type": "Human", "properties": {{"posture": "standing"}}}}],
  "edges": [{{"source": "PERSON_001", "relation": "PERFORMS", "target": "ACTIVITY_001", "confidence": 0.9}}]
}}
"""

def analyze_frame_with_vlm(frame_path, video_name, frame_id, cv_grounding=None):
    """Robust VLM analysis with multiple fallbacks"""
    
    if cv_grounding is None:
        cv_grounding = {}
    
    # Fallback 1: If CV data exists, use it directly
    if cv_grounding.get("persons") or cv_grounding.get("objects"):
        cv_fallback = _build_cv_graph_fallback(cv_grounding, video_name, frame_id)
    else:
        cv_fallback = {
            "surveillance_narrative": "No entities detected",
            "detailed_scene_analysis": {},
            "nodes": [],
            "edges": [],
            "video_name": video_name,
            "frame_id": str(frame_id)
        }
    
    img_b64 = _resize_image(frame_path, max_size=640)
    
    # Simplified CV summary for prompt
    cv_summary = f"{len(cv_grounding.get('persons', []))} persons, {len(cv_grounding.get('objects', []))} objects"
    
    prompt = SIMPLE_PROMPT.format(cv_summary=cv_summary)
    options = {"temperature": 0.0, "num_predict": 512, "top_p": 0.1}
    
    payload = {
        "model": VISION_MODEL,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "format": "json",
        "options": options,
    }
    
    # Try VLM with timeout and fallback
    try:
        print(f"   📊 VLM analyzing...")
        r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=60)
        r.raise_for_status()
        resp = r.json().get("response", "")
        
        data = _extract_json_multi_strategy(resp)
        
        # Merge with CV fallback (enrich)
        if not data.get("nodes"):
            data["nodes"] = cv_fallback["nodes"]
        if not data.get("edges"):
            data["edges"] = cv_fallback["edges"]
        if not data.get("surveillance_narrative"):
            data["surveillance_narrative"] = cv_fallback["surveillance_narrative"]
        
        data["video_name"] = video_name
        data["frame_id"] = str(frame_id)
        
        print(f"   ✅ VLM success: {len(data.get('nodes', []))} nodes, {len(data.get('edges', []))} edges")
        return data
        
    except Exception as e:
        print(f"   ⚠️ VLM failed, using CV fallback: {str(e)[:60]}")
        return cv_fallback