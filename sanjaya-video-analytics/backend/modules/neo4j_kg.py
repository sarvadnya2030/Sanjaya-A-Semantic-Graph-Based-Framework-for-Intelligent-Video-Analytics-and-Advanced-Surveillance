import cv2
import base64
import requests
import json
import re
import logging
from datetime import datetime
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError, ConstraintError

log = logging.getLogger("vlm")

def repair_truncated_json(json_str):
    """Repair incomplete JSON from VLM."""
    open_brackets = json_str.count('[') - json_str.count(']')
    json_str += ']' * open_brackets
    
    open_braces = json_str.count('{') - json_str.count('}')
    json_str += '}' * open_braces
    
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
    
    return json_str

def analyze_salient_frame(image_path: str, cv_metadata: dict, ollama_url="http://localhost:11434"):
    """
    RESEARCH-GRADE GRAPHRAG KNOWLEDGE GRAPH GENERATION
    """
    try:
        log.info(f"[VLM] Loading image: {image_path}")
        
        # Read and encode image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        _, buffer = cv2.imencode('.jpg', image)
        b64_img = base64.b64encode(buffer).decode('utf-8')
        
        log.info(f"[VLM] Image encoded, size: {len(b64_img)} chars")
        
        # Build rich CV context
        persons = cv_metadata.get('persons', [])
        objects_list = cv_metadata.get('objects', [])
        events = cv_metadata.get('events', [])
        frame_id = cv_metadata.get('frame_id')
        timestamp = cv_metadata.get('timestamp', 0)
        
        cv_context = f"""SURVEILLANCE FRAME ANALYSIS:
Frame: {frame_id} | Time: {timestamp:.2f}s
Detected: {len(persons)} persons, {len(objects_list)} objects
"""
        
        # Detailed person analysis
        if persons:
            cv_context += "\nPERSON BEHAVIORAL DATA:\n"
            for p in persons:
                cv_context += f"""  P{p.get('track_id')}:
    - Motion: {p.get('motion_state')} at {p.get('speed_px_s', 0):.0f}px/s
    - Posture: {p.get('posture')}
    - Zone: {p.get('zone')}
    - Direction: {p.get('direction_deg', 0):.0f}°
    - Dwell time: {p.get('dwell_time_s', 0):.1f}s
"""
        
        # GRAPHRAG PROMPT - Multi-entity reasoning with PERSON-OBJECT RELATIONSHIPS
        prompt = f"""You are a surveillance intelligence analyst. Create a RESEARCH-GRADE knowledge graph focusing on PERSON-OBJECT INTERACTIONS and SPATIAL RELATIONSHIPS.

{cv_context}

BUILD A RICH KNOWLEDGE GRAPH WITH:

1. **ENTITIES**:
   - Persons (P1, P2, etc.) with clothing, posture, behavior
   - Objects (laptop, bag, phone, etc.) with description, position
   - Actions (walking, carrying, placing, grabbing, using)
   - Locations (zones, areas, positions)
   - Interactions (meeting, exchange, surveillance)

2. **PERSON-OBJECT RELATIONSHIPS** (PRIMARY FOCUS):
   - CARRIES (person → object: "P1 carries red backpack")
   - USES (person → object: "P2 uses laptop")
   - TOUCHES (person → object: "P1 touches door handle")
   - PLACES (person → object → location: "P1 places bag on table")
   - GRABS (person → object: "P2 grabs phone")
   - INTERACTS_WITH (person → object: proximity, usage)
   - OWNS (person → object: possession indication)
   - EXCHANGES (person → object → person: "P1 hands document to P2")

3. **PERSON-PERSON RELATIONSHIPS**:
   - MEETS_WITH (person → person: social interaction)
   - FOLLOWS (person → person: spatial tracking)
   - TALKS_TO (person → person: communication)
   - STANDS_NEAR (person → person: proximity)
   - WALKS_WITH (person → person: group movement)

4. **SPATIAL RELATIONSHIPS**:
   - LOCATED_IN (entity → zone)
   - NEAR (entity → entity: distance < 2m)
   - BETWEEN (person → person/object)
   - APPROACHES (person → object/person)
   - MOVES_AWAY_FROM (person → object/person)

5. **TEMPORAL RELATIONSHIPS**:
   - BEFORE/AFTER (action → action)
   - DURING (action occurs during event)
   - SIMULTANEOUS (actions happen together)

6. **CONTEXTUAL ATTRIBUTES**:
   - distance: "close" | "medium" | "far"
   - duration: seconds
   - confidence: 0-1
   - intent_score: 0-1 (suspicious behavior)
   - risk_factor: "low" | "medium" | "high"

Return ONLY this JSON:

{{
  "scene_description": "detailed scene: persons, objects, their positions, what they're doing",
  "entities": [
    {{"id": "P1", "type": "Person", "label": "person in blue shirt", "attributes": {{"clothing": "blue shirt, jeans", "posture": "standing", "carrying": ["laptop bag"], "zone": "Z1", "activity": "working on laptop"}}}},
    {{"id": "Obj_Laptop_1", "type": "Object", "label": "silver laptop", "attributes": {{"class": "laptop", "color": "silver", "position": "on table", "zone": "Z1", "owner": "P1"}}}},
    {{"id": "Obj_Bag_1", "type": "Object", "label": "red backpack", "attributes": {{"class": "backpack", "color": "red", "size": "large", "zone": "Z1", "status": "on floor"}}}},
    {{"id": "Action_Typing_1", "type": "Action", "label": "typing on laptop", "attributes": {{"actor": "P1", "target": "Obj_Laptop_1", "duration": "ongoing", "intensity": "focused"}}}}
  ],
  "relationships": [
    {{"source": "P1", "target": "Obj_Laptop_1", "type": "USES", "attributes": {{"distance": "touching", "duration": 10, "confidence": 0.95, "context": "working"}}}},
    {{"source": "P1", "target": "Obj_Bag_1", "type": "CARRIES", "attributes": {{"manner": "on shoulder", "duration": 5, "confidence": 0.9}}}},
    {{"source": "P1", "target": "Action_Typing_1", "type": "PERFORMS", "attributes": {{"timestamp": {timestamp}}}}},
    {{"source": "P1", "target": "Z1", "type": "LOCATED_IN", "attributes": {{"position": "center"}}}},
    {{"source": "Obj_Laptop_1", "target": "Z1", "type": "PLACED_IN", "attributes": {{"surface": "table"}}}},
    {{"source": "P1", "target": "P2", "type": "NEAR", "attributes": {{"distance": "2m", "context": "same room"}}}}
  ],
  "interactions": [
    {{"type": "person_object", "person": "P1", "object": "Obj_Laptop_1", "action": "using", "description": "P1 actively using laptop for work", "risk_level": "low"}},
    {{"type": "person_object", "person": "P1", "object": "Obj_Bag_1", "action": "carrying", "description": "P1 carries personal belongings", "risk_level": "low"}},
    {{"type": "person_person", "person1": "P1", "person2": "P2", "action": "proximity", "description": "P1 and P2 in same area", "risk_level": "low"}}
  ],
  "scene_intelligence": {{
    "primary_activity": "office work/meeting/suspicious activity",
    "risk_assessment": "low|medium|high",
    "suspicious_patterns": ["any anomalies in person-object interactions"],
    "temporal_sequence": ["P1 enters → P1 places bag → P1 opens laptop → P1 starts typing"],
    "object_ownership": {{"Obj_Laptop_1": "P1", "Obj_Bag_1": "P1"}},
    "social_groups": [["P1", "P2"]]
  }}
}}

CRITICAL: 
- Map EVERY person to objects they interact with
- Create CARRIES/USES/TOUCHES relationships for ALL person-object interactions
- Include spatial proximity (NEAR) between persons
- Add confidence scores to all relationships
- Describe the PURPOSE of each interaction (work, theft, normal behavior)"""
        
        log.info(f"[VLM] Calling Qwen3-VL for GraphRAG analysis...")
        
        # CALL QWEN3-VL
        response = requests.post(
            f"{ollama_url}/api/generate",
            json={
                "model": "qwen3-vl:2b-instruct-q4_K_M",
                "prompt": prompt,
                "images": [b64_img],
                "stream": False,
                "temperature": 0.2,
                "options": {
                    "num_predict": 1000,
                    "top_p": 0.9
                }
            },
            timeout=120
        )
        
        if response.status_code != 200:
            raise Exception(f"Ollama API error: {response.status_code}")
        
        vlm_text = response.json().get("response", "").strip()
        
        log.info(f"[VLM] ✅ Got response ({len(vlm_text)} chars)")
        log.info(f"[VLM] Raw output: {vlm_text[:500]}")
        
        # Extract and repair JSON
        json_match = re.search(r'\{.*\}', vlm_text, re.DOTALL)
        if not json_match:
            log.error(f"[VLM] NO JSON FOUND!")
            raise ValueError("VLM did not return JSON")
        
        json_str = json_match.group()
        
        try:
            vlm_data = json.loads(json_str)
        except json.JSONDecodeError as e:
            log.warning(f"[VLM] JSON incomplete, repairing...")
            json_str = repair_truncated_json(json_str)
            vlm_data = json.loads(json_str)
        
        log.info(f"[VLM] ✅ Parsed GraphRAG KG")
        
        # Extract GraphRAG components with interactions
        entities = vlm_data.get("entities", [])
        relationships = vlm_data.get("relationships", [])
        interactions = vlm_data.get("interactions", [])
        scene_intel = vlm_data.get("scene_intelligence", {})
        
        log.info(f"[VLM] Entities: {len(entities)}, Relationships: {len(relationships)}, Interactions: {len(interactions)}")
        log.info(f"[VLM] Primary activity: {scene_intel.get('primary_activity')}")
        log.info(f"[VLM] Risk: {scene_intel.get('risk_assessment')}")
        
        # Build result with interactions
        result = {
            "image_description": vlm_data.get("scene_description", ""),
            "surveillance_description": f"Primary activity: {scene_intel.get('primary_activity', 'unknown')}. {vlm_data.get('scene_description', '')}",
            "behavioral_assessment": {
                "risk_level": scene_intel.get("risk_assessment", "unknown"),
                "inferred_intent": scene_intel.get("primary_activity", "unknown"),
                "primary_subjects": [f"P{p.get('track_id')}" for p in persons[:5]],
                "justification": scene_intel.get("suspicious_patterns", ["No specific concerns"]),
                "temporal_sequence": scene_intel.get("temporal_sequence", [])
            },
            "knowledge_graph": {
                "nodes": entities,
                "relationships": relationships,
                "interactions": interactions  # NEW: explicit interaction list
            },
            "scene_intelligence": scene_intel,
            "object_ownership": scene_intel.get("object_ownership", {}),
            "social_groups": scene_intel.get("social_groups", [])
        }
        
        # Validate KG richness - check for person-object relationships
        person_object_rels = [r for r in relationships if r.get('type') in ['CARRIES', 'USES', 'TOUCHES', 'PLACES', 'GRABS', 'INTERACTS_WITH']]
        if len(person_object_rels) < 1 and len(interactions) < 1:
            log.warning("[VLM] KG lacks person-object interactions, enriching with CV data...")
            result["knowledge_graph"] = _build_research_grade_kg(cv_metadata, frame_id, timestamp)
        else:
            # Even if VLM has some relationships, add CV-based spatial enrichment
            cv_kg = _build_research_grade_kg(cv_metadata, frame_id, timestamp)
            # Merge CV relationships into VLM KG
            result["knowledge_graph"]["nodes"].extend([n for n in cv_kg["nodes"] if n["id"] not in [x["id"] for x in result["knowledge_graph"]["nodes"]]])
            result["knowledge_graph"]["relationships"].extend(cv_kg["relationships"])
            result["knowledge_graph"]["interactions"].extend(cv_kg.get("interactions", []))
            log.info(f"[VLM] ✅ Enhanced VLM KG with CV spatial data: +{len(cv_kg['nodes'])} nodes, +{len(cv_kg['relationships'])} rels")
        
        log.info(f"[VLM] ✅ GraphRAG KG: {len(result['knowledge_graph']['nodes'])} nodes, {len(result['knowledge_graph']['relationships'])} rels")
        
        return result
        
    except Exception as e:
        log.error(f"[VLM] ERROR: {e}", exc_info=True)
        # Fallback to research-grade CV-based KG
        return {
            "image_description": "VLM analysis failed",
            "surveillance_description": f"CV-based analysis: {len(cv_metadata.get('persons', []))} persons detected",
            "behavioral_assessment": {
                "risk_level": "unknown",
                "inferred_intent": "unknown",
                "primary_subjects": [f"P{p.get('track_id')}" for p in cv_metadata.get('persons', [])[:5]],
                "justification": ["VLM failed, using CV data"],
                "temporal_sequence": []
            },
            "knowledge_graph": _build_research_grade_kg(cv_metadata, cv_metadata.get('frame_id'), cv_metadata.get('timestamp', 0)),
            "scene_intelligence": {"primary_activity": "unknown", "risk_assessment": "unknown"}
        }

def _build_research_grade_kg(cv_metadata, frame_id, timestamp):
    """
    Build RESEARCH-GRADE GraphRAG KG with PERSON-OBJECT INTERACTIONS from CV data.
    Focus on creating rich relationship mappings.
    """
    nodes = []
    relationships = []
    interactions = []
    
    persons = cv_metadata.get('persons', [])
    objects_list = cv_metadata.get('objects', [])
    events = cv_metadata.get('events', [])
    
    # Track object assignments for interaction analysis
    object_assignments = {}
    
    # Create PERSON ENTITIES with rich attributes
    for p in persons:
        pid = f"P{p.get('track_id')}"
        zone = p.get('zone', 'Z0')
        speed = p.get('speed_px_s', 0)
        motion = p.get('motion_state', 'unknown')
        posture = p.get('posture', 'unknown')
        
        nodes.append({
            "id": pid,
            "type": "Person",
            "label": f"Person {p.get('track_id')}",
            "attributes": {
                "zone": zone,
                "motion_state": motion,
                "posture": posture,
                "speed": speed,
                "risk_score": 1.0 if speed > 200 else (0.5 if speed > 100 else 0.1),
                "timestamp": timestamp,
                "bbox": p.get('bbox', [])
            }
        })
        
        # Action node
        action_id = f"Action_{motion}_{pid}_{frame_id}"
        nodes.append({
            "id": action_id,
            "type": "Action",
            "label": f"{motion.lower()} {posture}",
            "attributes": {
                "speed": speed,
                "intent_score": 0.9 if motion == "STOPPED" else (0.3 if speed > 200 else 0.5),
                "frame": frame_id
            }
        })
        
        # Zone node
        if not any(n['id'] == zone for n in nodes):
            nodes.append({
                "id": zone,
                "type": "Zone",
                "label": f"Zone {zone}",
                "attributes": {"occupancy": len([p for p in persons if p.get('zone') == zone])}
            })
        
        # RELATIONSHIPS
        relationships.append({"source": pid, "target": action_id, "type": "PERFORMS", "attributes": {"timestamp": timestamp}})
        relationships.append({"source": action_id, "target": zone, "type": "OCCURS_IN", "attributes": {"frame": frame_id}})
        relationships.append({"source": pid, "target": zone, "type": "LOCATED_IN", "attributes": {}})
    
    # Create OBJECT ENTITIES with position tracking
    for idx, o in enumerate(objects_list[:20]):
        oid = f"Obj_{o.get('class', 'object')}_{idx}_{frame_id}"
        zone = o.get('zone', 'Z0')
        obj_bbox = o.get('bbox', [])
        
        nodes.append({
            "id": oid,
            "type": "Object",
            "label": o.get('class', 'object'),
            "attributes": {
                "class": o.get('class'),
                "zone": zone,
                "confidence": o.get('confidence', 0),
                "bbox": obj_bbox,
                "position": "tracked"
            }
        })
        
        # Zone relationship
        if not any(n['id'] == zone for n in nodes):
            nodes.append({"id": zone, "type": "Zone", "label": f"Zone {zone}", "attributes": {}})
        
        relationships.append({"source": oid, "target": zone, "type": "PLACED_IN", "attributes": {}})
        
        # PERSON-OBJECT PROXIMITY ANALYSIS (spatial relationships)
        for p in persons:
            pid = f"P{p.get('track_id')}"
            person_bbox = p.get('bbox', [])
            
            # Same zone = potential interaction
            if p.get('zone') == zone:
                # Calculate bounding box distance
                if person_bbox and obj_bbox and len(person_bbox) >= 4 and len(obj_bbox) >= 4:
                    p_center_x = (person_bbox[0] + person_bbox[2]) / 2
                    p_center_y = (person_bbox[1] + person_bbox[3]) / 2
                    o_center_x = (obj_bbox[0] + obj_bbox[2]) / 2
                    o_center_y = (obj_bbox[1] + obj_bbox[3]) / 2
                    
                    distance = ((p_center_x - o_center_x)**2 + (p_center_y - o_center_y)**2)**0.5
                    
                    if distance < 100:  # Close proximity
                        rel_type = "USES" if p.get('motion_state') == "STOPPED" else "NEAR"
                        relationships.append({
                            "source": pid,
                            "target": oid,
                            "type": rel_type,
                            "attributes": {
                                "distance": round(distance, 2),
                                "proximity": "close",
                                "confidence": 0.8,
                                "context": "spatial_proximity"
                            }
                        })
                        
                        # Track interaction
                        interactions.append({
                            "type": "person_object",
                            "person": pid,
                            "object": oid,
                            "action": rel_type.lower(),
                            "description": f"{pid} {rel_type.lower()} {o.get('class')} at distance {round(distance, 2)}px",
                            "risk_level": "low",
                            "confidence": 0.8
                        })
                        
                        # Assign object to person
                        if oid not in object_assignments:
                            object_assignments[oid] = pid
                    
                    elif distance < 200:  # Medium proximity
                        relationships.append({
                            "source": pid,
                            "target": oid,
                            "type": "NEAR",
                            "attributes": {
                                "distance": round(distance, 2),
                                "proximity": "medium",
                                "confidence": 0.6
                            }
                        })
    
    # PERSON-PERSON RELATIONSHIPS (social groups, proximity)
    for i, p1 in enumerate(persons):
        for p2 in persons[i+1:]:
            pid1 = f"P{p1.get('track_id')}"
            pid2 = f"P{p2.get('track_id')}"
            
            # Same zone = potential interaction
            if p1.get('zone') == p2.get('zone'):
                bbox1 = p1.get('bbox', [])
                bbox2 = p2.get('bbox', [])
                
                if bbox1 and bbox2 and len(bbox1) >= 4 and len(bbox2) >= 4:
                    p1_x = (bbox1[0] + bbox1[2]) / 2
                    p1_y = (bbox1[1] + bbox1[3]) / 2
                    p2_x = (bbox2[0] + bbox2[2]) / 2
                    p2_y = (bbox2[1] + bbox2[3]) / 2
                    
                    distance = ((p1_x - p2_x)**2 + (p1_y - p2_y)**2)**0.5
                    
                    if distance < 150:  # Close proximity between persons
                        relationships.append({
                            "source": pid1,
                            "target": pid2,
                            "type": "NEAR",
                            "attributes": {
                                "distance": round(distance, 2),
                                "context": "social_proximity",
                                "confidence": 0.7
                            }
                        })
                        
                        # Track person-person interaction
                        interactions.append({
                            "type": "person_person",
                            "person1": pid1,
                            "person2": pid2,
                            "action": "proximity",
                            "description": f"{pid1} and {pid2} in close proximity ({round(distance, 2)}px)",
                            "risk_level": "low",
                            "confidence": 0.7
                        })
    
    # Event-based interactions
    for e in events:
        event_id = f"Event_{e.get('type')}_{e.get('track_id', 0)}_{frame_id}"
        nodes.append({
            "id": event_id,
            "type": "Event",
            "label": e.get('type', 'unknown'),
            "attributes": {"type": e.get('type'), "frame": frame_id, "timestamp": timestamp}
        })
        
        if 'track_id' in e:
            pid = f"P{e.get('track_id')}"
            relationships.append({"source": pid, "target": event_id, "type": "TRIGGERS", "attributes": {}})
    
    return {
        "nodes": nodes,
        "relationships": relationships,
        "interactions": interactions,
        "object_ownership": object_assignments
    }

def _ensure_constraints(tx):
    """Create constraints without conflicts."""
    # Remove old composite constraint if it exists (one-time cleanup)
    try:
        tx.run("DROP CONSTRAINT zone_name_video_unique IF EXISTS")
    except:
        pass
    
    # Use simple unique constraints per label
    tx.run("CREATE CONSTRAINT video_name_unique IF NOT EXISTS FOR (v:Video) REQUIRE v.name IS UNIQUE")
    tx.run("CREATE CONSTRAINT track_id_unique IF NOT EXISTS FOR (t:Track) REQUIRE t.id IS UNIQUE")
    tx.run("CREATE CONSTRAINT class_name_unique IF NOT EXISTS FOR (c:ObjectClass) REQUIRE c.name IS UNIQUE")
    # Zone constraint: per video to avoid global collisions
    tx.run("CREATE INDEX zone_name_video_idx IF NOT EXISTS FOR (z:Zone) ON (z.name, z.video)")

def export_surveillance_graph(uri, auth, events_path, video_name):
    """Export CV events to Neo4j."""
    try:
        with open(events_path, "r") as f:
            events = json.load(f)
        if not events:
            log.warning("[Neo4j] No events to export")
            return False

        driver = GraphDatabase.driver(uri, auth=auth, max_connection_lifetime=3600)
        with driver.session() as sess:
            try:
                sess.execute_write(_ensure_constraints)
            except Exception as ce:
                log.warning(f"[Neo4j] Constraint setup: {ce}")
            
            # Export events
            for e in events:
                try:
                    sess.execute_write(_merge_event, e, video_name)
                except ConstraintError as ce:
                    log.warning(f"[Neo4j] Constraint conflict on event {e.get('frame_id')}: {ce}. Skipping.")
                except Exception as e2:
                    log.error(f"[Neo4j] Error merging event: {e2}")
        
        driver.close()
        log.info(f"[Neo4j] ✅ Exported {len(events)} CV events to knowledge graph")
        return True
        
    except (ServiceUnavailable, ConnectionRefusedError) as e:
        log.warning(f"[Neo4j] ⚠️ Connection failed: {e}")
        return False
    except Exception as e:
        log.error(f"[Neo4j] ❌ Error: {e}", exc_info=True)
        return False

def _merge_event(tx, e, video_name):
    """Merge CV event into graph. Use MATCH + CREATE to avoid constraint conflicts."""
    zone = e.get("zone", "unknown")
    cls = e.get("object", "unknown")
    tid = str(e.get("track_id", "unknown"))
    etype = e.get("event", "unknown")
    fid = int(e.get("frame_id", 0))
    ts = float(e.get("timestamp", 0.0))
    conf = 1.0 if e.get("confidence") == "high" else 0.6
    dur = float(e.get("duration_sec", 0.0))

    # Step 1: Ensure Video node
    tx.run("""
MERGE (v:Video {name: $video})
ON CREATE SET v.createdAt = timestamp()
""", video=video_name)

    # Step 2: Ensure Zone node (unique per video)
    tx.run("""
MERGE (z:Zone {name: $zone, video: $video})
ON CREATE SET z.createdAt = timestamp()
""", zone=zone, video=video_name)

    # Step 3: Ensure ObjectClass
    tx.run("""
MERGE (c:ObjectClass {name: $cls})
ON CREATE SET c.createdAt = timestamp()
""", cls=cls)

    # Step 4: Ensure Track
    tx.run("""
MERGE (t:Track {id: $tid})
ON CREATE SET t.createdAt = timestamp()
""", tid=tid)

    # Step 5: Create/update Event
    tx.run("""
MERGE (ev:Event {video: $video, frame: $fid, type: $etype})
ON CREATE SET ev.ts = $ts, ev.confidence = $conf, ev.duration = $dur, ev.createdAt = timestamp()
ON MATCH SET ev.ts = $ts, ev.confidence = $conf, ev.duration = $dur
""", video=video_name, fid=fid, etype=etype, ts=ts, conf=conf, dur=dur)

    # Step 6: Create relationships
    tx.run("""
MATCH (v:Video {name: $video})
MATCH (ev:Event {video: $video, frame: $fid, type: $etype})
MERGE (v)-[:HAS_EVENT]->(ev)
""", video=video_name, fid=fid, etype=etype)

    tx.run("""
MATCH (t:Track {id: $tid})
MATCH (ev:Event {video: $video, frame: $fid, type: $etype})
MERGE (t)-[:PERFORMED]->(ev)
""", tid=tid, video=video_name, fid=fid, etype=etype)

    tx.run("""
MATCH (t:Track {id: $tid})
MATCH (c:ObjectClass {name: $cls})
MERGE (t)-[:OF_CLASS]->(c)
""", tid=tid, cls=cls)

    tx.run("""
MATCH (t:Track {id: $tid})
MATCH (z:Zone {name: $zone, video: $video})
MERGE (t)-[:IN_ZONE]->(z)
""", tid=tid, zone=zone, video=video_name)

def push_vlm_kg_to_neo4j(uri, auth, kg_data, video_name, frame_id):
    """
    Push RESEARCH-GRADE Knowledge Graph to Neo4j.
    GUARANTEES interconnected graph with FORCED relationships:
    - Person ↔ Object interactions (CARRIES, USES, NEAR)
    - Person ↔ Person proximity (NEAR, INTERACTS_WITH)
    - Person → Action (PERFORMS)
    - Action → Object (TARGETS)
    - Spatial fallback relationships if VLM fails
    """
    driver = GraphDatabase.driver(uri, auth=auth)
    
    try:
        with driver.session() as session:
            nodes = kg_data.get("nodes", [])
            relationships = kg_data.get("relationships", [])
            
            if not nodes and not relationships:
                log.warning(f"[Neo4j VLM] Empty KG for frame {frame_id}")
                return False
            
            # ===== CRITICAL: FORCE RELATIONSHIP CREATION BEFORE MERGING =====
            log.info(f"[Neo4j VLM] Initial data - Nodes: {len(nodes)}, Relationships: {len(relationships)}")
            
            # Extract entities by type
            person_nodes = [n for n in nodes if n.get('type') == 'Person']
            object_nodes = [n for n in nodes if n.get('type') == 'Object']
            action_nodes = [n for n in nodes if n.get('type') == 'Action']
            
            log.info(f"[Neo4j VLM] Entity breakdown - Persons: {len(person_nodes)}, Objects: {len(object_nodes)}, Actions: {len(action_nodes)}")
            
            # FORCE RELATIONSHIP GENERATION - Guarantee interconnected graph
            forced_relationships = []
            
            # 1. Person → Action (PERFORMS) - from actor attribute
            for action in action_nodes:
                actor = action.get('attributes', {}).get('actor', '')
                if actor:
                    forced_relationships.append({
                        'source': actor,
                        'target': action['id'],
                        'type': 'PERFORMS',
                        'attributes': {'confidence': 0.95, 'source': 'cv_actor'}
                    })
            
            # 2. Action → Object (TARGETS) - from target attribute
            for action in action_nodes:
                target = action.get('attributes', {}).get('target', '')
                if target:
                    forced_relationships.append({
                        'source': action['id'],
                        'target': target,
                        'type': 'TARGETS',
                        'attributes': {'confidence': 0.9, 'source': 'cv_target'}
                    })
            
            # 3. Person ↔ Object - SPATIAL PROXIMITY (distance-based)
            for person in person_nodes:
                p_bbox = person.get('attributes', {}).get('bbox', [])
                p_carrying = person.get('attributes', {}).get('carrying', [])
                
                if len(p_bbox) == 4:
                    p_cx = (p_bbox[0] + p_bbox[2]) / 2
                    p_cy = (p_bbox[1] + p_bbox[3]) / 2
                    
                    for obj in object_nodes:
                        o_bbox = obj.get('attributes', {}).get('bbox', [])
                        if len(o_bbox) == 4:
                            o_cx = (o_bbox[0] + o_bbox[2]) / 2
                            o_cy = (o_bbox[1] + o_bbox[3]) / 2
                            
                            dist = ((p_cx - o_cx)**2 + (p_cy - o_cy)**2)**0.5
                            
                            # Very close = CARRIES (< 120px)
                            if dist < 120:
                                forced_relationships.append({
                                    'source': person['id'],
                                    'target': obj['id'],
                                    'type': 'CARRIES',
                                    'attributes': {'distance_px': round(dist, 1), 'confidence': 0.92, 'source': 'spatial'}
                                })
                            # Close = USES (120-200px)
                            elif dist < 200:
                                forced_relationships.append({
                                    'source': person['id'],
                                    'target': obj['id'],
                                    'type': 'USES',
                                    'attributes': {'distance_px': round(dist, 1), 'confidence': 0.85, 'source': 'spatial'}
                                })
                            # Medium = NEAR (200-350px)
                            elif dist < 350:
                                forced_relationships.append({
                                    'source': person['id'],
                                    'target': obj['id'],
                                    'type': 'NEAR',
                                    'attributes': {'distance_px': round(dist, 1), 'confidence': 0.75, 'source': 'spatial'}
                                })
            
            # 4. Person ↔ Person - PROXIMITY
            for i, p1 in enumerate(person_nodes):
                p1_bbox = p1.get('attributes', {}).get('bbox', [])
                p1_zone = p1.get('attributes', {}).get('zone', '')
                
                for p2 in person_nodes[i+1:]:
                    p2_bbox = p2.get('attributes', {}).get('bbox', [])
                    p2_zone = p2.get('attributes', {}).get('zone', '')
                    
                    # Same zone = NEAR
                    if p1_zone and p2_zone and p1_zone == p2_zone:
                        forced_relationships.append({
                            'source': p1['id'],
                            'target': p2['id'],
                            'type': 'NEAR',
                            'attributes': {'zone': p1_zone, 'confidence': 0.8, 'source': 'zone'}
                        })
                    # Distance-based
                    elif len(p1_bbox) == 4 and len(p2_bbox) == 4:
                        p1_cx = (p1_bbox[0] + p1_bbox[2]) / 2
                        p1_cy = (p1_bbox[1] + p1_bbox[3]) / 2
                        p2_cx = (p2_bbox[0] + p2_bbox[2]) / 2
                        p2_cy = (p2_bbox[1] + p2_bbox[3]) / 2
                        
                        dist = ((p1_cx - p2_cx)**2 + (p1_cy - p2_cy)**2)**0.5
                        
                        if dist < 300:
                            forced_relationships.append({
                                'source': p1['id'],
                                'target': p2['id'],
                                'type': 'NEAR',
                                'attributes': {'distance_px': round(dist, 1), 'confidence': 0.82, 'source': 'spatial'}
                            })
            
            # 5. Object → Person (OWNED_BY) - reverse ownership
            for obj in object_nodes:
                owner = obj.get('attributes', {}).get('owner', '')
                if owner:
                    forced_relationships.append({
                        'source': obj['id'],
                        'target': owner,
                        'type': 'OWNED_BY',
                        'attributes': {'confidence': 0.85, 'source': 'cv_ownership'}
                    })
            
            # MERGE: Combine VLM relationships + Forced spatial relationships
            all_rels = relationships + forced_relationships
            
            # DEDUPLICATE by (source, target, type)
            seen = set()
            unique_rels = []
            for rel in all_rels:
                key = (rel['source'], rel['target'], rel['type'])
                if key not in seen:
                    seen.add(key)
                    unique_rels.append(rel)
            
            relationships = unique_rels
            
            log.info(f"[Neo4j VLM] 🔗 RELATIONSHIP SUMMARY:")
            log.info(f"  - VLM provided: {len(kg_data.get('relationships', []))}")
            log.info(f"  - Forced spatial: {len(forced_relationships)}")
            log.info(f"  - Total unique: {len(relationships)}")
            log.info(f"[Neo4j VLM] Building research-grade KG for frame {frame_id}: {len(nodes)} nodes, {len(relationships)} rels")
            
            # CREATE NODES with rich properties
            for node in nodes:
                node_id = node.get("id")
                node_type = node.get("type", "Entity")
                label = node.get("label", node_id)
                attributes = node.get("attributes", {})
                
                # Build property dictionary
                props = {
                    "id": node_id,
                    "label": label,
                    "video": video_name,
                    "frame": frame_id,
                    "lastUpdated": int(datetime.now().timestamp() * 1000)
                }
                
                # Add all attributes
                for k, v in attributes.items():
                    if v is not None and v != "":
                        props[k] = v
                
                # Create node with specific type
                cypher = f"""
                MERGE (n:{node_type} {{id: $id, video: $video, frame: $frame}})
                SET n += $props
                RETURN n
                """
                
                try:
                    session.run(cypher, id=node_id, video=video_name, frame=frame_id, props=props)
                    log.info(f"[Neo4j VLM] ✓ Created {node_type} node: {node_id}")
                except Exception as e:
                    log.warning(f"[Neo4j VLM] Node creation warning for {node_id}: {e}")
            
            # CREATE RELATIONSHIPS with properties
            for rel in relationships:
                source = rel.get("source")
                target = rel.get("target")
                rel_type = rel.get("type", "RELATED_TO")
                rel_attrs = rel.get("attributes", {})
                
                if not source or not target:
                    continue
                
                # Build relationship properties
                rel_props = {
                    "video": video_name,
                    "frame": frame_id,
                    "createdAt": int(datetime.now().timestamp() * 1000)
                }
                
                # Add relationship attributes
                for k, v in rel_attrs.items():
                    if v is not None:
                        rel_props[k] = v
                
                # Create relationship (match any node type with matching id)
                cypher = f"""
                MATCH (a {{id: $source, video: $video}})
                MATCH (b {{id: $target, video: $video}})
                MERGE (a)-[r:{rel_type}]->(b)
                SET r += $props
                RETURN r
                """
                
                try:
                    session.run(cypher, source=source, target=target, video=video_name, props=rel_props)
                    log.info(f"[Neo4j VLM] ✓ Created relationship: {source} -{rel_type}-> {target}")
                except Exception as e:
                    log.warning(f"[Neo4j VLM] Relationship warning {source}->{target}: {e}")
            
            log.info(f"[Neo4j VLM] ✅ Pushed KG for frame {frame_id}: {len(nodes)} nodes, {len(relationships)} rels")
            
            # CONNECTIVITY ENHANCEMENTS - Make graph research-grade
            
            # 1. Connect ALL entities to FrameSummary (central hub)
            summary_id = f"Summary_Frame{frame_id}"
            session.run("""
                MERGE (s:FrameSummary {id: $summary_id, video: $video, frame: $frame})
            """, summary_id=summary_id, video=video_name, frame=frame_id)
            
            for node in nodes:
                node_id = node.get("id")
                node_type = node.get("type", "Entity")
                
                # Link entity → FrameSummary
                session.run(f"""
                    MATCH (e:{node_type} {{id: $node_id, video: $video}})
                    MATCH (s:FrameSummary {{id: $summary_id}})
                    MERGE (e)-[:PART_OF]->(s)
                """, node_id=node_id, video=video_name, summary_id=summary_id)
            
            # 2. Connect to Video hub
            session.run("""
                MATCH (v:Video {name: $video})
                MATCH (s:FrameSummary {id: $summary_id})
                MERGE (v)-[:CONTAINS]->(s)
            """, video=video_name, summary_id=summary_id)
            
            # 3. Create inter-entity relationships based on attributes
            for node in nodes:
                node_id = node.get("id")
                node_type = node.get("type")
                attributes = node.get("attributes", {})
                
                # Link Person → Zone
                if node_type == "Person" and attributes.get("zone"):
                    zone_id = attributes["zone"]
                    session.run("""
                        MATCH (p:Person {id: $person_id, video: $video})
                        MERGE (z:Zone {id: $zone_id, video: $video})
                        MERGE (p)-[:LOCATED_IN {frame: $frame}]->(z)
                    """, person_id=node_id, video=video_name, zone_id=zone_id, frame=frame_id)
                
                # Link Object → Zone
                if node_type == "Object" and attributes.get("zone"):
                    zone_id = attributes["zone"]
                    session.run("""
                        MATCH (o:Object {id: $obj_id, video: $video})
                        MERGE (z:Zone {id: $zone_id, video: $video})
                        MERGE (o)-[:PLACED_IN {frame: $frame}]->(z)
                    """, obj_id=node_id, video=video_name, zone_id=zone_id, frame=frame_id)
                
                # Link Object → Person (ownership)
                if node_type == "Object" and attributes.get("owner"):
                    owner_id = attributes["owner"]
                    session.run("""
                        MATCH (o:Object {id: $obj_id, video: $video})
                        MATCH (p:Person {id: $owner_id, video: $video})
                        MERGE (p)-[:OWNS {frame: $frame, confidence: 0.8}]->(o)
                    """, obj_id=node_id, owner_id=owner_id, video=video_name, frame=frame_id)
                
                # Link Action → Person (actor)
                if node_type == "Action" and attributes.get("actor"):
                    actor_id = attributes["actor"]
                    session.run("""
                        MATCH (a:Action {id: $action_id, video: $video})
                        MATCH (p:Person {id: $actor_id, video: $video})
                        MERGE (p)-[:PERFORMS {frame: $frame}]->(a)
                    """, action_id=node_id, actor_id=actor_id, video=video_name, frame=frame_id)
                
                # Link Action → Object (target)
                if node_type == "Action" and attributes.get("target"):
                    target_id = attributes["target"]
                    session.run("""
                        MATCH (a:Action {id: $action_id, video: $video})
                        MATCH (o:Object {id: $target_id, video: $video})
                        MERGE (a)-[:TARGETS {frame: $frame}]->(o)
                    """, action_id=node_id, target_id=target_id, video=video_name, frame=frame_id)
            
            # 4. Create temporal frame sequence (Frame N → Frame N+1)
            if frame_id > 0:
                prev_frame_id = frame_id - 1
                session.run("""
                    MATCH (prev:FrameSummary {video: $video, frame: $prev_frame})
                    MATCH (curr:FrameSummary {id: $summary_id})
                    MERGE (prev)-[:NEXT_FRAME {time_delta: 1}]->(curr)
                """, video=video_name, prev_frame=prev_frame_id, summary_id=summary_id)
            
            # 5. Create proximity relationships (Person-Person in same zone)
            session.run("""
                MATCH (p1:Person {video: $video, frame: $frame})
                MATCH (p2:Person {video: $video, frame: $frame})
                WHERE p1.id < p2.id AND p1.zone = p2.zone
                MERGE (p1)-[:NEAR {distance: 'same_zone', frame: $frame, confidence: 0.7}]->(p2)
            """, video=video_name, frame=frame_id)
            
            # 6. Create Person-Object proximity in same zone
            session.run("""
                MATCH (p:Person {video: $video, frame: $frame})
                MATCH (o:Object {video: $video, frame: $frame})
                WHERE p.zone = o.zone
                  AND NOT (p)-[:USES|CARRIES|OWNS]->(o)
                MERGE (p)-[:NEAR {type: 'spatial_proximity', frame: $frame, confidence: 0.6}]->(o)
            """, video=video_name, frame=frame_id)
            
            # 7. FORCE RELATIONSHIP CREATION if VLM returned no relationships
            rel_count = len(relationships)
            if rel_count == 0:
                log.warning(f"[Neo4j VLM] VLM returned 0 relationships! Creating fallback connections...")
                
                # Force Person-Action relationships for all actions
                session.run("""
                    MATCH (a:Action {video: $video, frame: $frame})
                    MATCH (p:Person {video: $video, frame: $frame})
                    WHERE a.actor = p.id OR a.label CONTAINS p.id
                    MERGE (p)-[:PERFORMS {frame: $frame, source: 'fallback'}]->(a)
                """, video=video_name, frame=frame_id)
                
                # Force Person-Object relationships based on spatial proximity
                session.run("""
                    MATCH (p:Person {video: $video, frame: $frame})
                    MATCH (o:Object {video: $video, frame: $frame})
                    WHERE p.zone = o.zone
                    WITH p, o, point({x: toFloat(p.bbox[0] + p.bbox[2])/2, y: toFloat(p.bbox[1] + p.bbox[3])/2}) AS p_center,
                              point({x: toFloat(o.bbox[0] + o.bbox[2])/2, y: toFloat(o.bbox[1] + o.bbox[3])/2}) AS o_center
                    WHERE distance(p_center, o_center) < 200
                    MERGE (p)-[:INTERACTS_WITH {frame: $frame, distance: distance(p_center, o_center), source: 'spatial_fallback'}]->(o)
                """, video=video_name, frame=frame_id)
                
                # Force Person-Person proximity
                session.run("""
                    MATCH (p1:Person {video: $video, frame: $frame})
                    MATCH (p2:Person {video: $video, frame: $frame})
                    WHERE p1.id < p2.id AND p1.zone = p2.zone
                    MERGE (p1)-[:NEAR {frame: $frame, context: 'same_zone', source: 'fallback'}]->(p2)
                """, video=video_name, frame=frame_id)
                
                # Force Action-Object targeting
                session.run("""
                    MATCH (a:Action {video: $video, frame: $frame})
                    MATCH (o:Object {video: $video, frame: $frame})
                    WHERE a.target = o.id OR a.label CONTAINS o.class
                    MERGE (a)-[:TARGETS {frame: $frame, source: 'fallback'}]->(o)
                """, video=video_name, frame=frame_id)
                
                log.info(f"[Neo4j VLM] ✅ Created fallback relationships: Person↔Person, Person↔Object, Person↔Action, Action↔Object")
            
            log.info(f"[Neo4j VLM] ✅ Enhanced connectivity: All entities linked to hub + {rel_count} explicit + fallback inter-entity relationships + zones")
            
            return True
            
    except Exception as e:
        log.error(f"[Neo4j VLM] Error pushing KG: {e}", exc_info=True)
        return False
    finally:
        driver.close()


def push_vlm_analysis_summary(uri, auth, vlm_result, video_name, frame_id):
    """
    Create RESEARCH-GRADE interconnected analysis with:
    - Full VLM JSON data stored in nodes
    - Risk nodes linked to specific behaviors
    - Anomaly nodes with evidence chains
    - Summary node aggregating insights
    - Scene description and context
    - Multi-hop paths: Person→Action→Risk→Summary
    """
    driver = GraphDatabase.driver(uri, auth=auth)
    
    try:
        with driver.session() as session:
            scene_intel = vlm_result.get("scene_intelligence", {})
            behavioral = vlm_result.get("behavioral_assessment", {})
            image_desc = vlm_result.get("image_description", "")
            surveillance_desc = vlm_result.get("surveillance_description", "")
            scene_info = vlm_result.get("scene", {})
            
            # 0. CREATE VIDEO NODE (MASTER HUB) - Graph anchor
            video_cypher = """
            MERGE (v:Video {name: $video})
            SET v.lastUpdated = $timestamp
            RETURN v
            """
            session.run(video_cypher, video=video_name, timestamp=int(datetime.now().timestamp() * 1000))
            log.info(f"[Neo4j] ✓ Video hub node: {video_name}")
            
            # 1. CREATE FRAME SUMMARY NODE (central hub) with full VLM data
            summary_id = f"Summary_Frame{frame_id}"
            cypher = """
            MERGE (s:FrameSummary {id: $id, video: $video, frame: $frame})
            SET s.scene_description = $scene_desc,
                s.surveillance_description = $surv_desc,
                s.image_description = $image_desc,
                s.primary_activity = $activity,
                s.overall_risk = $risk,
                s.inferred_intent = $intent,
                s.scene_type = $scene_type,
                s.lighting = $lighting,
                s.time_of_day = $time_of_day,
                s.camera_angle = $camera_angle,
                s.environment = $environment,
                s.total_persons = $persons,
                s.total_objects = $objects,
                s.justification = $justification,
                s.temporal_sequence = $temporal_seq,
                s.timestamp = $timestamp,
                s.createdAt = $created,
                s.vlm_raw = $vlm_raw
            RETURN s
            """
            
            session.run(
                cypher,
                id=summary_id,
                video=video_name,
                frame=frame_id,
                scene_desc=vlm_result.get("scene_description", ""),
                surv_desc=surveillance_desc,
                image_desc=image_desc,
                activity=scene_intel.get("primary_activity", "unknown"),
                risk=behavioral.get("risk_level", "unknown"),
                intent=behavioral.get("inferred_intent", "unknown"),
                scene_type=scene_info.get("type", "unknown"),
                lighting=scene_info.get("lighting", "unknown"),
                time_of_day=scene_info.get("time_of_day", "unknown"),
                camera_angle=scene_info.get("camera_angle", "unknown"),
                environment=scene_info.get("environment", "unknown"),
                persons=len(behavioral.get("primary_subjects", [])),
                objects=0,
                justification=str(behavioral.get("justification", [])),
                temporal_seq=str(behavioral.get("temporal_sequence", [])),
                timestamp=vlm_result.get("timestamp", 0.0),
                created=int(datetime.now().timestamp() * 1000),
                vlm_raw=json.dumps(vlm_result)  # Store complete VLM JSON
            )
            log.info(f"[Neo4j] ✅ Created FrameSummary with full VLM data: {summary_id}")
            
            # Link Summary → Video
            session.run("""
                MATCH (s:FrameSummary {id: $summary_id})
                MATCH (v:Video {name: $video})
                MERGE (v)-[:HAS_FRAME {frame_number: $frame, timestamp: $ts}]->(s)
            """, summary_id=summary_id, video=video_name, frame=frame_id, ts=vlm_result.get("timestamp", 0.0))
            
            # 2. CREATE SCENE CONTEXT NODE (environment details)
            if scene_info:
                scene_id = f"Scene_Frame{frame_id}"
                scene_cypher = """
                MERGE (sc:Scene {id: $id, video: $video, frame: $frame})
                SET sc.type = $type,
                    sc.lighting = $lighting,
                    sc.time_of_day = $time,
                    sc.camera_angle = $angle,
                    sc.environment = $env,
                    sc.weather = $weather,
                    sc.visibility = $visibility,
                    sc.description = $desc,
                    sc.createdAt = $created
                RETURN sc
                """
                
                session.run(
                    scene_cypher,
                    id=scene_id,
                    video=video_name,
                    frame=frame_id,
                    type=scene_info.get("type", "unknown"),
                    lighting=scene_info.get("lighting", "normal"),
                    time=scene_info.get("time_of_day", "unknown"),
                    angle=scene_info.get("camera_angle", "unknown"),
                    env=scene_info.get("environment", "unknown"),
                    weather=scene_info.get("weather", "unknown"),
                    visibility=scene_info.get("visibility", "good"),
                    desc=scene_info.get("description", ""),
                    created=int(datetime.now().timestamp() * 1000)
                )
                
                # Link Scene → Summary
                session.run("""
                    MATCH (sc:Scene {id: $scene_id})
                    MATCH (s:FrameSummary {id: $summary_id})
                    MERGE (sc)-[:DESCRIBES]->(s)
                """, scene_id=scene_id, summary_id=summary_id)
                
                log.info(f"[Neo4j] ✓ Created Scene context: {scene_id}")
            
            # 3. CREATE RISK NODES with full evidence from VLM
            risks = vlm_result.get("risks", []) or []
            for idx, risk in enumerate(risks):
                risk_id = f"Risk_{risk.get('type', 'unknown').replace(' ', '_')}_F{frame_id}_{idx}"
                risk_cypher = """
                MERGE (r:Risk {id: $id, video: $video, frame: $frame})
                SET r.type = $type,
                    r.severity = $severity,
                    r.description = $description,
                    r.rating = $rating,
                    r.evidence = $evidence,
                    r.confidence = $confidence,
                    r.category = $category,
                    r.source = 'VLM',
                    r.createdAt = $created
                RETURN r
                """
                
                session.run(
                    risk_cypher,
                    id=risk_id,
                    video=video_name,
                    frame=frame_id,
                    type=risk.get("type", "unknown"),
                    severity=risk.get("severity", "low"),
                    description=risk.get("description", ""),
                    rating=risk.get("rating", 5),
                    evidence=str(risk.get("evidence", [])),
                    confidence=risk.get("confidence", 0.5),
                    category=risk.get("category", "general"),
                    created=int(datetime.now().timestamp() * 1000)
                )
                
                # Link Risk → Summary
                session.run("""
                    MATCH (r:Risk {id: $risk_id})
                    MATCH (s:FrameSummary {id: $summary_id})
                    MERGE (r)-[:CONTRIBUTES_TO {weight: $severity}]->(s)
                """, risk_id=risk_id, summary_id=summary_id, 
                     severity=3 if risk.get("severity") == "high" else (2 if risk.get("severity") == "medium" else 1))
                
                log.info(f"[Neo4j] ✓ Created Risk: {risk_id} [{risk.get('severity')}] - {risk.get('type')}")
            
            # 4. CREATE ANOMALY NODES with full VLM details
            anomalies = vlm_result.get("anomalies", []) or []
            for idx, anomaly in enumerate(anomalies):
                anomaly_id = f"Anomaly_{anomaly.get('type', 'unknown').replace(' ', '_')}_F{frame_id}_{idx}"
                anomaly_cypher = """
                MERGE (a:Anomaly {id: $id, video: $video, frame: $frame})
                SET a.type = $type,
                    a.description = $description,
                    a.rating = $rating,
                    a.confidence = $confidence,
                    a.severity = $severity,
                    a.details = $details,
                    a.source = 'VLM',
                    a.createdAt = $created
                RETURN a
                """
                
                session.run(
                    anomaly_cypher,
                    id=anomaly_id,
                    video=video_name,
                    frame=frame_id,
                    type=anomaly.get("type", "unknown"),
                    description=anomaly.get("description", ""),
                    rating=anomaly.get("rating", 5),
                    confidence=anomaly.get("confidence", 0.5),
                    severity=anomaly.get("severity", "low"),
                    details=json.dumps(anomaly),
                    created=int(datetime.now().timestamp() * 1000)
                )
                
                # Link Anomaly → Summary
                session.run("""
                    MATCH (a:Anomaly {id: $anomaly_id})
                    MATCH (s:FrameSummary {id: $summary_id})
                    MERGE (a)-[:DETECTED_IN {confidence: $conf}]->(s)
                """, anomaly_id=anomaly_id, summary_id=summary_id, conf=anomaly.get("confidence", 0.5))
                
                log.info(f"[Neo4j] ✓ Created Anomaly: {anomaly_id} - {anomaly.get('type')}")
            
            # 5. CREATE DETECTED OBJECTS from VLM with rich attributes
            detected_objects = vlm_result.get("detected_objects", []) or []
            for idx, obj in enumerate(detected_objects):
                obj_id = f"VLMObject_{obj.replace(' ', '_')}_F{frame_id}_{idx}"
                obj_cypher = """
                MERGE (o:DetectedObject {id: $id, video: $video, frame: $frame})
                SET o.class = $class_name,
                    o.type = $type,
                    o.description = $description,
                    o.source = 'VLM',
                    o.createdAt = $created
                RETURN o
                """
                
                session.run(
                    obj_cypher,
                    id=obj_id,
                    video=video_name,
                    frame=frame_id,
                    class_name=obj,
                    type=obj.split()[0] if obj else "unknown",
                    description=f"Detected: {obj}",
                    created=int(datetime.now().timestamp() * 1000)
                )
                
                # Link Object → Summary
                session.run("""
                    MATCH (o:DetectedObject {id: $obj_id})
                    MATCH (s:FrameSummary {id: $summary_id})
                    MERGE (o)-[:FOUND_IN]->(s)
                """, obj_id=obj_id, summary_id=summary_id)
            
            # 6. LINK PERSONS to SUMMARY and create interaction chains
            subjects = behavioral.get("primary_subjects", [])
            for subject_id in subjects:
                # Link Person → Summary
                session.run("""
                    MATCH (p:Person {video: $video, frame: $frame})
                    WHERE p.id STARTS WITH $subject_prefix
                    MATCH (s:FrameSummary {id: $summary_id})
                    MERGE (p)-[:APPEARS_IN]->(s)
                """, video=video_name, frame=frame_id, subject_prefix=subject_id, summary_id=summary_id)
                
                # Person → Risk chains (when high/medium severity)
                for idx, risk in enumerate(risks):
                    if risk.get("severity") in ["high", "medium"]:
                        risk_id = f"Risk_{risk.get('type', 'unknown').replace(' ', '_')}_F{frame_id}_{idx}"
                        session.run("""
                            MATCH (p:Person {video: $video, frame: $frame})
                            WHERE p.id STARTS WITH $subject_prefix
                            MATCH (r:Risk {id: $risk_id})
                            MERGE (p)-[:EXHIBITS {severity: $severity}]->(r)
                        """, video=video_name, frame=frame_id, subject_prefix=subject_id, 
                             risk_id=risk_id, severity=risk.get("severity"))
                
                # Person → Anomaly chains
                for idx, anomaly in enumerate(anomalies):
                    anomaly_id = f"Anomaly_{anomaly.get('type', 'unknown').replace(' ', '_')}_F{frame_id}_{idx}"
                    session.run("""
                        MATCH (p:Person {video: $video, frame: $frame})
                        WHERE p.id STARTS WITH $subject_prefix
                        MATCH (a:Anomaly {id: $anomaly_id})
                        MERGE (p)-[:SHOWS {confidence: $conf}]->(a)
                    """, video=video_name, frame=frame_id, subject_prefix=subject_id,
                         anomaly_id=anomaly_id, conf=anomaly.get("confidence", 0.5))
            
            # 7. CREATE PERSON ↔ OBJECT INTERACTIONS with context
            session.run("""
                MATCH (p:Person {video: $video, frame: $frame})
                MATCH (o:Object {video: $video, frame: $frame})
                WHERE p.zone = o.zone
                MERGE (p)-[r:INTERACTS_WITH]->(o)
                SET r.proximity = 'same_zone', 
                    r.frame = $frame,
                    r.context = 'spatial_proximity'
            """, video=video_name, frame=frame_id)
            
            # Link VLM detected objects to CV objects
            session.run("""
                MATCH (vo:DetectedObject {video: $video, frame: $frame, source: 'VLM'})
                MATCH (co:Object {video: $video, frame: $frame})
                WHERE vo.class CONTAINS co.class OR co.class CONTAINS vo.class
                MERGE (vo)-[:CORRESPONDS_TO]->(co)
            """, video=video_name, frame=frame_id)
            
            # 8. CREATE OBJECT → RISK chains (suspicious items)
            session.run("""
                MATCH (o:Object {video: $video, frame: $frame})
                MATCH (r:Risk {video: $video, frame: $frame})
                WHERE o.class IN ['knife', 'gun', 'weapon', 'suspicious_item', 'bag', 'backpack']
                   OR r.type CONTAINS 'weapon' OR r.type CONTAINS 'threat' OR r.type CONTAINS 'suspicious'
                MERGE (o)-[:INDICATES {reason: 'suspicious_object'}]->(r)
            """, video=video_name, frame=frame_id)
            
            # 9. CREATE TEMPORAL SEQUENCES (Action → Action chains)
            sequence = behavioral.get("temporal_sequence", [])
            for i in range(len(sequence) - 1):
                action1 = sequence[i]
                action2 = sequence[i + 1]
                session.run("""
                    MATCH (a1:Action {video: $video, frame: $frame})
                    WHERE a1.label CONTAINS $action1
                    MATCH (a2:Action {video: $video, frame: $frame})
                    WHERE a2.label CONTAINS $action2
                    MERGE (a1)-[:PRECEDES {order: $order}]->(a2)
                """, video=video_name, frame=frame_id, action1=action1, action2=action2, order=i)
            
            # 10. CREATE OVERALL RISK ASSESSMENT NODE
            if risks or anomalies:
                assessment_id = f"Assessment_Frame{frame_id}"
                total_risk_score = sum([
                    (3 if r.get("severity") == "high" else 2 if r.get("severity") == "medium" else 1) * r.get("rating", 5)
                    for r in risks
                ]) + sum([a.get("rating", 5) * a.get("confidence", 0.5) for a in anomalies])
                
                session.run("""
                    MERGE (a:RiskAssessment {id: $id, video: $video, frame: $frame})
                    SET a.overall_risk = $risk,
                        a.risk_score = $score,
                        a.total_risks = $num_risks,
                        a.total_anomalies = $num_anomalies,
                        a.recommendation = $recommendation,
                        a.createdAt = $created
                """, id=assessment_id, video=video_name, frame=frame_id,
                     risk=behavioral.get("risk_level", "unknown"),
                     score=total_risk_score,
                     num_risks=len(risks),
                     num_anomalies=len(anomalies),
                     recommendation="ALERT" if total_risk_score > 20 else ("REVIEW" if total_risk_score > 10 else "NORMAL"),
                     created=int(datetime.now().timestamp() * 1000))
                
                # Link Assessment → Summary
                session.run("""
                    MATCH (a:RiskAssessment {id: $assessment_id})
                    MATCH (s:FrameSummary {id: $summary_id})
                    MERGE (a)-[:EVALUATES]->(s)
                """, assessment_id=assessment_id, summary_id=summary_id)
                
                # Link Assessment → Video
                session.run("""
                    MATCH (a:RiskAssessment {id: $assessment_id})
                    MATCH (v:Video {name: $video})
                    MERGE (v)-[:HAS_ASSESSMENT]->(a)
                """, assessment_id=assessment_id, video=video_name)
            
            # 11. CREATE CROSS-FRAME TRACK PERSISTENCE (Person continuity)
            if frame_id > 0:
                session.run("""
                    MATCH (curr_p:Person {video: $video, frame: $frame})
                    MATCH (prev_p:Person {video: $video})
                    WHERE prev_p.frame < $frame 
                      AND prev_p.track_id = curr_p.track_id
                      AND NOT EXISTS((prev_p)-[:CONTINUES_AS]->(curr_p))
                    WITH curr_p, prev_p
                    ORDER BY prev_p.frame DESC
                    LIMIT 1
                    MERGE (prev_p)-[:CONTINUES_AS {frames_apart: $frame - prev_p.frame}]->(curr_p)
                """, video=video_name, frame=frame_id)
            
            # 12. Link Zones to Video hub
            session.run("""
                MATCH (z:Zone {video: $video})
                MATCH (v:Video {name: $video})
                WHERE NOT EXISTS((v)-[:HAS_ZONE]->(z))
                MERGE (v)-[:HAS_ZONE]->(z)
            """, video=video_name)
            
            # 13. Create Scene → Zone relationships
            session.run("""
                MATCH (sc:Scene {video: $video, frame: $frame})
                MATCH (z:Zone {video: $video})
                MATCH (p:Person {video: $video, frame: $frame})
                WHERE (p)-[:LOCATED_IN]->(z)
                MERGE (sc)-[:COVERS {frame: $frame}]->(z)
            """, video=video_name, frame=frame_id)
            
            log.info(f"[Neo4j] ✅ COMPLETE RESEARCH-GRADE KG: Video hub + Summary + {len(risks)} risks + {len(anomalies)} anomalies + {len(subjects)} persons + Scene + {len(detected_objects)} objects + Zones + Cross-frame tracks")
            
            return True
            
    except Exception as e:
        log.error(f"[Neo4j VLM] Error creating analysis summary: {e}", exc_info=True)
        return False
    finally:
        driver.close()