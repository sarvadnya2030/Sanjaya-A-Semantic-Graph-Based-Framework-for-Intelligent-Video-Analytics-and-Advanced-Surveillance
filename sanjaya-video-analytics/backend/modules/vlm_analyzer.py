import re
import json
import requests
import base64
import logging
import os
import time

log = logging.getLogger("vlm")

def analyze_salient_frame(image_path: str, frame_id: int, timestamp: float, cv_detections=None, ollama_url="http://localhost:11434", max_retries=2):
    """VLM analysis with CV detection context."""
    
    if cv_detections is None:
        cv_detections = {"persons": [], "objects": []}
    
    for attempt in range(max_retries):
        try:
            log.info(f"[VLM] Analyzing frame {frame_id} @ {timestamp:.1f}s (attempt {attempt+1})")
            
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            # CHECK IMAGE SIZE
            file_size = os.path.getsize(image_path)
            log.info(f"[VLM] Image file: {file_size} bytes")
            
            if file_size == 0:
                raise ValueError("Image file is empty!")
            
            # Read and encode image
            with open(image_path, "rb") as f:
                img_bytes = f.read()
            
            img_data = base64.b64encode(img_bytes).decode("utf-8")
            log.info(f"[VLM] Image encoded: {len(img_data)} chars")
            
            # PREPARE CV DETECTION SUMMARY
            person_count = len(cv_detections.get("persons", []))
            object_list = cv_detections.get("objects", [])
            object_classes = [obj.get("class", "unknown") for obj in object_list]
            object_summary = ", ".join(set(object_classes)) if object_classes else "none"
            
            cv_context = f"""YOLO Detections (green bounding boxes in image):
- {person_count} person(s) detected
- Objects: {object_summary}
- Total detections: {person_count + len(object_list)}"""
            
            # DETAILED SURVEILLANCE PROMPT FOR KNOWLEDGE GRAPH
            prompt = f"""{cv_context}

Analyze this surveillance image with YOLO detection overlays (green boxes with labels).

Return detailed JSON for knowledge graph construction:

{{
  "scene": {{
    "type": "indoor/outdoor/retail/office/street/parking",
    "lighting": "bright/dim/normal",
    "time_of_day": "morning/afternoon/evening/night",
    "weather": "clear/rainy/foggy" (if outdoor)
  }},
  "persons": [
    {{
      "id": "P1",
      "appearance": "detailed clothing, age group, gender",
      "posture": "standing/walking/running/sitting/crouching",
      "action": "what they are doing",
      "location": "specific zone or area",
      "gaze_direction": "where looking",
      "carrying": "bag/phone/nothing/unknown"
    }}
  ],
  "objects": [
    {{
      "type": "car/motorcycle/bag/laptop/cart/etc",
      "location": "where in scene",
      "state": "moving/stationary/abandoned",
      "owner": "which person ID or unknown"
    }}
  ],
  "interactions": [
    {{
      "type": "person-person/person-object/person-vehicle",
      "participants": ["P1", "P2"] or ["P1", "car"],
      "description": "detailed interaction",
      "duration": "brief/ongoing/prolonged"
    }}
  ],
  "risks": [
    {{
      "type": "loitering/suspicious_behavior/unauthorized_access/abandoned_object/crowd_formation/vehicle_violation",
      "severity": "low/medium/high",
      "location": "where",
      "involved": ["P1", "object1"],
      "description": "detailed risk explanation",
      "rating": 1-10 (numeric risk score)
    }}
  ],
  "anomalies": [
    {{
      "type": "unusual_path/restricted_area/after_hours/suspicious_package/aggressive_behavior",
      "description": "what is anomalous and why",
      "rating": 1-10 (numeric anomaly score)
    }}
  ],
  "relationships": [
    {{
      "source": "P1",
      "relation": "CARRIES/NEAR/INTERACTS_WITH/APPROACHES/AVOIDS/USES/TOUCHES",
      "target": "bag1",
      "confidence": 0.9
    }}
  ],
  "overall_risk_rating": 1-10,
  "overall_anomaly_rating": 1-10
}}

**CRITICAL REQUIREMENTS:**
1. Create AT LEAST 3-5 relationships showing person-object or person-person connections
2. Every person MUST have at least 1 relationship (CARRIES, USES, NEAR, or INTERACTS_WITH)
3. Include spatial relationships (NEAR) for people/objects in same area
4. Add numeric ratings (1-10) for ALL risks and anomalies

Focus on: person-object interactions, movement patterns, suspicious behaviors, security risks, spatial relationships."""
            
            log.info("[VLM] Sending request to Ollama...")
            
            # CHECK OLLAMA HEALTH
            try:
                health = requests.get(f"{ollama_url}/api/tags", timeout=5)
                if health.status_code != 200:
                    raise Exception("Ollama not responding")
            except:
                raise Exception("Cannot connect to Ollama")
            
            # SEND REQUEST
            response = requests.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": "qwen3-vl:2b-instruct-q4_K_M",
                    "prompt": prompt,
                    "images": [img_data],
                    "stream": False
                },
                timeout=180
            )
            
            log.info(f"[VLM] Response status: {response.status_code}")
            
            if response.status_code != 200:
                log.error(f"[VLM] HTTP {response.status_code}: {response.text[:500]}")
                raise Exception(f"Ollama returned {response.status_code}")
            
            response_data = response.json()
            raw = response_data.get("response", "")
            
            log.info(f"[VLM] Response: {len(raw)} chars")
            
            if len(raw) == 0:
                log.error(f"[VLM] EMPTY RESPONSE! Full data: {response_data}")
                
                # Check if model is still loading
                if response_data.get("done_reason") == "load":
                    log.warning("[VLM] Model still loading, retrying...")
                    time.sleep(5)
                    continue
                
                raise ValueError("Empty Ollama response")
            
            # LOG FIRST 300 CHARS
            log.info(f"[VLM] Raw: {raw[:300]}...")
            
            # EXTRACT JSON
            json_str = raw.strip()
            
            # Remove markdown
            json_str = re.sub(r'```json\s*', '', json_str, flags=re.IGNORECASE)
            json_str = re.sub(r'```\s*', '', json_str)
            json_str = json_str.strip()
            
            # Find first complete JSON
            start = json_str.find('{')
            if start == -1:
                log.error(f"[VLM] No JSON found in: {json_str[:200]}")
                raise ValueError("No JSON in response")
            
            json_str = json_str[start:]
            
            # Find matching closing brace
            brace_count = 0
            end = -1
            for i, char in enumerate(json_str):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = i + 1
                        break
            
            if end == -1:
                log.error("[VLM] JSON incomplete, trying repair...")
                # Add missing closing braces
                missing = json_str.count('{') - json_str.count('}')
                json_str += '}' * missing
            else:
                json_str = json_str[:end]
            
            log.info(f"[VLM] Extracted JSON: {len(json_str)} chars")
            
            # PARSE
            try:
                vlm_data = json.loads(json_str)
                log.info("[VLM] ✅ JSON parsed successfully")
            except json.JSONDecodeError as e:
                log.error(f"[VLM] Parse error: {e}")
                log.error(f"[VLM] JSON string: {json_str[:500]}")
                raise
            
            # VALIDATE
            if not isinstance(vlm_data, dict):
                raise ValueError("VLM response is not a dictionary")
            
            # Add metadata
            vlm_data["_metadata"] = {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "image_path": image_path,
                "attempt": attempt + 1,
                "response_length": len(raw)
            }
            
            log.info(f"[VLM] ✅ Frame {frame_id} analyzed successfully")
            
            return vlm_data
            
        except Exception as e:
            log.error(f"[VLM] Attempt {attempt+1} failed: {e}")
            
            if attempt < max_retries - 1:
                log.info(f"[VLM] Retrying in 3 seconds...")
                time.sleep(3)
            else:
                log.error(f"[VLM] All {max_retries} attempts failed!")
                
                # FALLBACK
                return {
                    "surveillance_type": "error",
                    "description": f"VLM analysis failed after {max_retries} attempts: {str(e)}",
                    "details": {
                        "type": "error",
                        "reason": str(e)
                    },
                    "_metadata": {
                        "frame_id": frame_id,
                        "timestamp": timestamp,
                        "image_path": image_path,
                        "error": str(e)
                    }
                }