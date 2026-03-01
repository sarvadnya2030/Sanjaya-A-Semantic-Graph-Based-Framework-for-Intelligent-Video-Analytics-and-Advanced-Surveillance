import os
import json
import faiss
import numpy as np
import requests
import logging

log = logging.getLogger("rag.json_rag")

class JsonRAG:
    def __init__(self, json_dirs=None, ollama_url="http://localhost:11434"):
        self.json_dirs = json_dirs or ["json_outputs"]
        self.ollama_url = ollama_url
        self.index = None
        self.documents = []
        self.embed_model = "bge-m3:latest"
        self.llm_model = "gemma:2b"
        log.info(f"[RAG] Initialized with dirs: {self.json_dirs}")
    
    def _extract_text_from_event(self, event):
        """Extract text from a single event dict."""
        parts = []
        
        
        event_type = event.get("type", "")
        track_id = event.get("track_id", "")
        motion = event.get("motion_state", "")
        zone = event.get("zone", "")
        priority = event.get("priority", "")
        speed = event.get("speed_px_s", 0)
        frame_id = event.get("frame_id", 0)
        timestamp = event.get("timestamp", 0)
        
        if motion and zone:
            parts.append(
                f"At {timestamp:.1f}s: Person track-{track_id} was {motion} in {zone} "
                f"(speed: {speed:.0f}px/s, priority: {priority})"
            )
        
        return " ".join(parts)
    
    def _extract_text_from_enriched(self, data):
        """Extract rich text from enriched frame JSON."""
        parts = []
        
        # Frame metadata
        frame_id = data.get("frame_id", 0)
        timestamp = data.get("timestamp", 0)
        parts.append(f"Frame {frame_id} at {timestamp:.1f}s:")
        
        # VLM Surveillance summary
        surveillance = data.get("surveillance", {})
        if surveillance:
            scene = surveillance.get("scene_type", "")
            risk = surveillance.get("risk_level", "")
            summary = surveillance.get("summary", "")
            if summary and summary != "Analysis incomplete":
                parts.append(f"Scene: {scene}, Risk: {risk}. {summary}")
        
        # VLM Persons (rich descriptions)
        persons_vlm = data.get("persons", [])
        for person in persons_vlm:
            if isinstance(person, dict):
                pid = person.get("id", "")
                appearance = person.get("appearance", "")
                action = person.get("action", "")
                suspicious = person.get("suspicious", False)
                
                if appearance:  # VLM data
                    parts.append(
                        f"{pid}: {appearance}. Action: {action}. "
                        f"{'SUSPICIOUS' if suspicious else 'Normal behavior'}."
                    )
                else:  # CV data
                    track_id = person.get("track_id", "")
                    motion = person.get("motion_state", "")
                    zone = person.get("zone", "")
                    if motion:
                        parts.append(f"Track-{track_id}: {motion} in {zone}")
        
        # VLM Objects
        objects = data.get("objects", [])
        for obj in objects:
            if isinstance(obj, dict):
                obj_type = obj.get("type", obj.get("class", ""))
                location = obj.get("location", "")
                owner = obj.get("owner", "")
                
                if location:  # VLM data
                    parts.append(f"Object: {obj_type} at {location}, owner: {owner}")
                else:  # CV data
                    conf = obj.get("confidence", 0)
                    if conf > 0:
                        parts.append(f"Detected: {obj_type} ({conf:.0%} confidence)")
        
        # VLM Interactions
        interactions = data.get("interactions", [])
        for interaction in interactions:
            if isinstance(interaction, dict):
                itype = interaction.get("type", "")
                desc = interaction.get("description", "")
                participants = interaction.get("participants", [])
                if desc:
                    parts.append(f"Interaction: {desc} between {', '.join(participants)}")
        
        # CV Events
        events = data.get("events", [])
        for event in events:
            if isinstance(event, dict):
                event_text = self._extract_text_from_event(event)
                if event_text:
                    parts.append(event_text)
        
        # Knowledge Graph
        kg = data.get("knowledge_graph", {})
        if kg:
            nodes = kg.get("nodes", [])
            for node in nodes:
                if isinstance(node, dict):
                    nid = node.get("id", "")
                    ntype = node.get("type", "")
                    props = node.get("properties", {})
                    behavior = props.get("behavior", "")
                    if behavior:
                        parts.append(f"{ntype} {nid}: {behavior}")
            
            rels = kg.get("relationships", [])
            for rel in rels:
                if isinstance(rel, dict):
                    src = rel.get("source", "")
                    tgt = rel.get("target", "")
                    rtype = rel.get("type", "")
                    if src and tgt and rtype:
                        parts.append(f"{src} {rtype} {tgt}")
        
        return " ".join(parts)
    
    def build_index(self):
        """Build FAISS index from ALL JSONs in json_outputs."""
        log.info("[RAG] 🔨 Starting FAISS index build...")
        
        all_texts = []
        all_metadata = []
        
        for json_dir in self.json_dirs:
            if not os.path.exists(json_dir):
                log.warning(f"[RAG] Directory not found: {json_dir}")
                continue
            
            json_files = sorted([f for f in os.listdir(json_dir) if f.endswith('.json')])
            log.info(f"[RAG] Found {len(json_files)} JSON files in {json_dir}")
            
            for fname in json_files:
                fpath = os.path.join(json_dir, fname)
                
                # SKIP OLD ENRICHED FILES COMPLETELY
                if "enriched" in fname:
                    log.info(f"[RAG] ⏭️ Skipping old enriched file: {fname}")
                    continue
                
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    text = ""
                    doc_type = "unknown"
                    
                    # ONLY VLM FILES
                    if "_vlm.json" in fname:
                        text = self._extract_text_from_vlm(data)
                        doc_type = "vlm_analysis"
                        log.info(f"[RAG] ✅ VLM file: {fname}")
                    
                    # ONLY CV FILES
                    elif "_cv.json" in fname:
                        text = self._extract_text_from_cv(data)
                        doc_type = "cv_data"
                        log.info(f"[RAG] ✅ CV file: {fname}")
                    
                    # EVENT FILES
                    elif isinstance(data, list):
                        event_texts = []
                        for event in data[:100]:
                            event_text = self._extract_text_from_event(event)
                            if event_text:
                                event_texts.append(event_text)
                        text = " ".join(event_texts)
                        doc_type = "events"
                        log.info(f"[RAG] ✅ Events file: {fname}")
                    
                    # SKIP EVERYTHING ELSE
                    else:
                        log.info(f"[RAG] ⏭️ Skipping: {fname}")
                        continue

                    # Validate text
                    if not text or len(text.strip()) < 20:
                        log.warning(f"[RAG] ⚠️ Skipping {fname} (insufficient text: {len(text)} chars)")
                        continue
                    
                    all_texts.append(text)
                    all_metadata.append({
                        'filename': fname,
                        'path': fpath,
                        'type': doc_type,
                        'frame_id': data.get('frame_id', 0) if isinstance(data, dict) else 0,
                        'timestamp': data.get('timestamp', 0) if isinstance(data, dict) else 0,
                        'text_length': len(text)
                    })
                    
                    log.info(f"[RAG] ✅ Indexed {fname} ({doc_type}, {len(text)} chars)")
                
                except json.JSONDecodeError as e:
                    log.error(f"[RAG] ❌ JSON error in {fname}: {e}")
                except Exception as e:
                    log.error(f"[RAG] ❌ Failed {fname}: {e}")
        
        if not all_texts:
            log.error("[RAG] ❌ NO DOCUMENTS TO INDEX!")
            return
        
        log.info(f"[RAG] 🔄 Embedding {len(all_texts)} documents...")
        
        # Get embeddings
        embeddings = []
        for i, text in enumerate(all_texts):
            try:
                emb = self._get_embedding(text[:8000])
                embeddings.append(emb)
                if (i + 1) % 5 == 0:
                    log.info(f"[RAG] Embedded {i+1}/{len(all_texts)}...")
            except Exception as e:
                log.error(f"[RAG] Embedding failed for doc {i}: {e}")
                embeddings.append(np.zeros(1024))
        
        embeddings = np.array(embeddings).astype('float32')
        
        # Build FAISS
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
        self.documents = list(zip(all_texts, all_metadata))
        
        log.info(f"[RAG] ✅ INDEX READY: {len(self.documents)} docs, {dimension} dims")
        
        # Log sample
        if self.documents:
            sample_text = all_texts[0][:200]
            log.info(f"[RAG] Sample text: {sample_text}...")
    
    def _get_embedding(self, text):
        """Get embedding from Ollama."""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": self.embed_model, "prompt": text},
                timeout=30
            )
            response.raise_for_status()
            return np.array(response.json()["embedding"])
        except Exception as e:
            log.error(f"[RAG] Embedding error: {e}")
            return np.zeros(1024)
    
    def search(self, query, k=5):
        """Search for relevant documents."""
        if self.index is None or not self.documents:
            log.error("[RAG] ❌ Index not built! Call build_index() first.")
            return []
        
        log.info(f"[RAG] 🔍 Searching for: '{query}'")
        
        q_emb = self._get_embedding(query).reshape(1, -1).astype('float32')
        
        # Search more documents to allow VLM boosting
        k_search = min(k * 3, len(self.documents))
        distances, indices = self.index.search(q_emb, k_search)
        
        all_results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                text, metadata = self.documents[idx]
                all_results.append({
                    'text': text,
                    'metadata': metadata,
                    'distance': float(distances[0][i])
                })
        
        # Check if question prefers detailed VLM analysis
        query_lower = query.lower()
        prefers_vlm = any(keyword in query_lower for keyword in [
            'object', 'carrying', 'person', 'people', 'interaction', 
            'risk', 'anomaly', 'suspicious', 'behavior', 'wearing',
            'appearance', 'action', 'relationship', 'associated',
            'what', 'who', 'describe', 'scene'
        ])
        
        if prefers_vlm:
            # Boost VLM results significantly
            for r in all_results:
                if 'vlm' in r['metadata']['filename']:
                    r['distance'] *= 0.3  # Strong boost for VLM
            
            # Re-sort by adjusted distance
            all_results.sort(key=lambda x: x['distance'])
        
        # Take top k
        results = all_results[:k]
        
        for i, r in enumerate(results, 1):
            log.info(f"[RAG]   Match {i+1}: {r['metadata']['filename']} (dist: {r['distance']:.2f})")
        
        return results
    
    def ask(self, question, k=5):
        """Ask question using RAG with improved query routing."""
        log.info(f"[RAG] ❓ Question: {question}")
        
        # Expand question for better semantic matching
        expanded_question = question
        keywords = question.lower()
        
        # Add context keywords for better VLM retrieval
        if any(word in keywords for word in ['object', 'carrying', 'person', 'people', 'interaction', 'risk', 'anomaly', 'suspicious']):
            expanded_question = f"{question} (looking for VLM scene analysis with persons objects interactions risks anomalies)"
        
        # Search with expanded query
        results = self.search(expanded_question, k=k)
        
        if not results:
            return {
                "answer": "❌ No data indexed. Upload a video first.",
                "confidence": "0%",
                "evidence": [],
                "sources": []
            }
        
        # Build context - use MORE context from each result
        context_parts = []
        sources = []
        
        for i, result in enumerate(results, 1):
            # Use full text, not just first 500 chars
            context_parts.append(f"[Source {i}] {result['text']}")
            sources.append(result['metadata']['filename'])
        
        context = "\n\n".join(context_parts)
        
        # Generate answer with better prompt
        prompt = f"""Answer this surveillance question directly using the data below.

Question: {question}

Surveillance Data:
{context}

Instructions:
- Give a direct, focused answer
- List specific details: persons (P1, P2), objects, locations
- Include risks and anomalies if relevant
- Keep answer under 200 words
- No introductory phrases or explanations

Answer:"""
        
        try:
            log.info(f"[RAG] 🤖 Asking LLM ({self.llm_model})...")
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,  # Slightly higher for better answers
                        "num_predict": 500,  # Allow longer answers
                        "top_p": 0.9,
                        "repeat_penalty": 1.1
                    }
                },
                timeout=120
            )
            
            response.raise_for_status()
            resp_data = response.json()
            answer = resp_data.get("response", "").strip()
            
            # Clean up formatting aggressively
            answer = answer.replace('**', '')  # Remove bold
            answer = answer.replace('* ', '')  # Remove bullets
            
            # Remove common LLM prefixes with multiple passes
            prefixes_to_remove = [
                "sure, here's the answer to the question:",
                "here's the answer based on the provided data:",
                "here's the answer you requested:",
                "here's the answer to the surveillance question:",
                "here's the focused answer:",
                "here's the direct answer:",
                "here's the answer:",
                "here's a direct answer:",
                "sure,",
                "based on the data,",
                "based on the surveillance data,",
                "according to the data,",
                "the answer is:"
            ]
            
            # Clean multiple times to catch nested phrases
            for _ in range(3):
                answer_lower = answer.lower()
                for prefix in prefixes_to_remove:
                    if answer_lower.startswith(prefix):
                        answer = answer[len(prefix):].strip()
                        break
            
            # Remove excessive newlines but keep some structure
            answer = '\n'.join(line.strip() for line in answer.split('\n') if line.strip())
            
            if not answer:
                answer = "Unable to generate answer from the provided context."
            
            # Calculate confidence
            avg_dist = np.mean([r['distance'] for r in results])
            confidence = max(0, min(1, 1 - (avg_dist / 50)))
            
            log.info(f"[RAG] ✅ Answer generated (confidence: {confidence:.0%})")
            
            return {
                "answer": answer,
                "confidence": f"{confidence:.0%}",
                "evidence": context_parts,
                "sources": sources
            }
        
        except Exception as e:
            log.error(f"[RAG] ❌ LLM error: {e}")
            return {
                "answer": f"Error generating answer: {e}",
                "confidence": "0%",
                "evidence": context_parts if 'context_parts' in locals() else [],
                "sources": sources if 'sources' in locals() else []
            }
    
    def _extract_text_from_vlm(self, vlm_data):
        """Extract VLM description text for knowledge graph."""
        parts = []
        
        # Get metadata
        metadata = vlm_data.get("_metadata", {})
        frame_id = metadata.get("frame_id", vlm_data.get("frame_id", 0))
        timestamp = metadata.get("timestamp", vlm_data.get("timestamp", 0))
        
        parts.append(f"Frame {frame_id} at {timestamp:.1f}s:")
        
        # SCENE ANALYSIS
        scene = vlm_data.get("scene", {})
        if scene:
            scene_type = scene.get("type", "")
            lighting = scene.get("lighting", "")
            if scene_type:
                parts.append(f"Scene: {scene_type} environment, {lighting} lighting.")
        
        # PERSONS (DETAILED)
        persons = vlm_data.get("persons", [])
        for p in persons:
            if isinstance(p, dict):
                pid = p.get("id", "")
                appearance = p.get("appearance", "")
                posture = p.get("posture", "")
                action = p.get("action", "")
                location = p.get("location", "")
                carrying = p.get("carrying", "")
                
                person_desc = f"{pid}: {appearance}. " if appearance else f"{pid}: "
                if posture:
                    person_desc += f"Posture: {posture}. "
                if action:
                    person_desc += f"Action: {action}. "
                if location:
                    person_desc += f"Location: {location}. "
                if carrying and carrying != "nothing":
                    person_desc += f"Carrying: {carrying}."
                
                parts.append(person_desc)
        
        # OBJECTS
        objects = vlm_data.get("objects", [])
        for obj in objects:
            if isinstance(obj, dict):
                obj_type = obj.get("type", "")
                location = obj.get("location", "")
                state = obj.get("state", "")
                owner = obj.get("owner", "")
                
                if obj_type:
                    obj_desc = f"Object {obj_type}: {location}, {state}"
                    if owner and owner != "unknown":
                        obj_desc += f", belongs to {owner}"
                    parts.append(obj_desc + ".")
        
        # INTERACTIONS
        interactions = vlm_data.get("interactions", [])
        for inter in interactions:
            if isinstance(inter, dict):
                itype = inter.get("type", "")
                desc = inter.get("description", "")
                participants = inter.get("participants", [])
                if desc:
                    parts.append(f"Interaction ({itype}): {desc} involving {', '.join(participants)}.")
        
        # RISKS
        risks = vlm_data.get("risks", [])
        for risk in risks:
            if isinstance(risk, dict):
                rtype = risk.get("type", "")
                severity = risk.get("severity", "")
                desc = risk.get("description", "")
                if desc:
                    parts.append(f"⚠️ Risk ({severity} severity): {rtype} - {desc}")
        
        # ANOMALIES
        anomalies = vlm_data.get("anomalies", [])
        for anom in anomalies:
            if isinstance(anom, dict):
                atype = anom.get("type", "")
                desc = anom.get("description", "")
                if desc:
                    parts.append(f"🚨 Anomaly: {atype} - {desc}")
        
        # RELATIONSHIPS (for KG)
        relationships = vlm_data.get("relationships", [])
        for rel in relationships:
            if isinstance(rel, dict):
                src = rel.get("source", "")
                relation = rel.get("relation", "")
                tgt = rel.get("target", "")
                conf = rel.get("confidence", 0)
                if src and relation and tgt:
                    parts.append(f"{src} {relation} {tgt} (confidence: {conf:.0%})")
        
        # FALLBACK: Old format support
        if not parts or len(parts) == 1:
            surv_type = vlm_data.get("surveillance_type", "")
            description = vlm_data.get("description", "")
            if description and "failed" not in description.lower():
                parts.append(f"Surveillance: {surv_type}. {description}")
            
            features = vlm_data.get("features", [])
            if features:
                for feat in features:
                    if isinstance(feat, dict):
                        name = feat.get("name", "")
                        desc = feat.get("description", "")
                        if name and desc:
                            parts.append(f"{name}: {desc}")
        
        # Filter out non-strings and convert to string
        text_parts = []
        for part in parts:
            if isinstance(part, str):
                text_parts.append(part)
            elif part:
                text_parts.append(str(part))
        
        return " ".join(text_parts)
    
    def _extract_text_from_cv(self, cv_data):
        """Extract CV detection text."""
        parts = []
        
        frame_id = cv_data.get("frame_id", 0)
        timestamp = cv_data.get("timestamp", 0)
        
        parts.append(f"Frame {frame_id} at {timestamp:.1f}s:")
        
        # PERSONS
        persons = cv_data.get("persons", [])
        for p in persons:
            track_id = p.get("track_id", "")
            motion = p.get("motion_state", "")
            zone = p.get("zone", "")
            if motion:
                parts.append(f"Track {track_id}: {motion} in {zone}")
        
        # OBJECTS
        objects = cv_data.get("objects", [])
        for obj in objects:
            obj_class = obj.get("class", "")
            conf = obj.get("confidence", 0)
            parts.append(f"Detected {obj_class} ({conf:.0%})")
        
        return " ".join(parts)