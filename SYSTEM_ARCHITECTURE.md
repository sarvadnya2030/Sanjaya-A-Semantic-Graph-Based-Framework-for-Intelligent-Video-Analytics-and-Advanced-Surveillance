# SANJAYA - Video Intelligence Platform
## System Architecture Documentation

---

## 1. Executive Summary

**Sanjaya** is a research-grade, multi-modal video analytics platform combining Computer Vision (CV), Vision-Language Models (VLM), and Knowledge Graph (KG) technologies for real-time surveillance intelligence. The system processes surveillance video through a 6-stage pipeline, generating semantic events, behavioral assessments, and queryable knowledge graphs.

**Key Capabilities:**
- Real-time person tracking with motion state detection
- VLM-powered scene understanding and risk assessment
- Research-grade Neo4j knowledge graph with 15+ relationship types
- Hybrid RAG (Retrieval-Augmented Generation) for natural language queries
- Interactive dashboard with analytics visualizations

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SANJAYA VIDEO ANALYTICS                       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────┐      ┌──────────────────────────────────────────────┐
│   VIDEO     │      │           PROCESSING PIPELINE                 │
│   UPLOAD    │─────▶│  1. Motion Gating                             │
│  (Flask)    │      │  2. CV Detection (YOLOv8n)                    │
└─────────────┘      │  3. Multi-Object Tracking (DeepSORT)          │
                     │  4. Zone Analysis (9-zone grid)                │
                     │  5. Event Generation (semantic events)         │
                     │  6. Salient Frame Selection                    │
                     └────────────────┬───────────────────────────────┘
                                      │
                     ┌────────────────┴───────────────────────────────┐
                     │                                                 │
         ┌───────────▼────────────┐              ┌──────────▼──────────────┐
         │   VLM ANALYSIS         │              │   DATA PERSISTENCE       │
         │  (Qwen3-VL 4-bit)      │              │  - JSON outputs          │
         │  - Scene understanding │              │  - Frame images          │
         │  - Entity extraction   │              │  - Event logs            │
         │  - Risk assessment     │              │  - CV metadata           │
         │  - Relationship mining │              └──────────┬───────────────┘
         └───────────┬────────────┘                         │
                     │                                      │
         ┌───────────▼───────────────────────────┬─────────▼───────────────┐
         │    KNOWLEDGE GRAPH CONSTRUCTION       │                          │
         │         (Neo4j 5.x)                   │    RAG INDEXING          │
         │  - Video hub (anchor)                 │  - JsonRAG (FAISS)       │
         │  - FrameSummary nodes                 │  - GraphRAG (Cypher)     │
         │  - Person/Object/Action/Zone          │  - Hybrid fusion         │
         │  - Risk/Anomaly chains                │  - Sentence transformers │
         │  - 15+ relationship types             │                          │
         │  - Cross-frame tracking               │                          │
         └───────────┬───────────────────────────┴──────────┬───────────────┘
                     │                                      │
         ┌───────────▼──────────────────────────────────────▼───────────────┐
         │                    QUERY & ANALYTICS LAYER                        │
         │  - Natural language queries (3 RAG modes)                         │
         │  - Chain-of-thought reasoning                                     │
         │  - Dashboard visualizations (Chart.js)                            │
         │  - Event insights & analytics                                     │
         └───────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Architecture

### 3.1 Computer Vision Pipeline (`cv_pipeline/`)

**Purpose:** Extract quantitative metrics from video frames

**Components:**
```
┌────────────────────────────────────────────────────────────────┐
│  CVPipeline (pipeline.py)                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │ Motion       │─▶│   YOLO       │─▶│  DeepSORT    │        │
│  │ Gating       │  │  Detector    │  │  Tracker     │        │
│  │ (threshold   │  │ (yolov8n.pt) │  │ (IoU-based)  │        │
│  │  filtering)  │  │              │  │              │        │
│  └──────────────┘  └──────────────┘  └──────┬───────┘        │
│                                              │                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────▼───────┐        │
│  │   Salient    │◀─│    Event     │◀─│     Zone     │        │
│  │   Frame      │  │  Generator   │  │   Analyzer   │        │
│  │  Selection   │  │  (semantic)  │  │  (9-zone)    │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
└────────────────────────────────────────────────────────────────┘
```

**Key Technologies:**
- **YOLOv8n** (nano): Lightweight object detection (~3ms inference on GPU)
- **DeepSORT**: Multi-object tracking with ID persistence (IoU matching + Kalman filtering)
- **Motion Gating**: Frame differencing to skip low-activity frames (~50% reduction)
- **Zone Analyzer**: 9-zone spatial grid for location mapping
- **Event Generator**: FSM-based semantic event detection (loitering, moving, walking, stationary)

**Output:**
```json
{
  "events": [...],           // 496 events with motion states, speeds, zones
  "salient_frames": [...],   // Top 3-5 frames ranked by activity score
  "cv_stats": {...}          // Processing metrics
}
```

---

### 3.2 Vision-Language Model Integration (`modules/vlm_analyzer.py`)

**Purpose:** Semantic scene understanding and behavioral assessment

**Architecture:**
```
┌─────────────────────────────────────────────────────────────────┐
│  VLM Analyzer (Qwen3-VL 2b-instruct-q4_K_M)                     │
│                                                                  │
│  Input: Frame Image + CV Detection Context                      │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Prompt Engineering (Research-Grade KG Focus)          │    │
│  │  - Entity extraction (Person, Object, Action)          │    │
│  │  - Relationship mining (CARRIES, USES, NEAR, TOUCHES)  │    │
│  │  - Scene intelligence (risk, intent, anomalies)        │    │
│  │  - Behavioral assessment (posture, gaze, carrying)     │    │
│  │  - Spatial relationships (zones, proximity)            │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Ollama API (localhost:11434)                           │   │
│  │  - 4-bit quantization (edge deployment ready)           │   │
│  │  - Temperature: 0.2 (deterministic)                     │   │
│  │  - Timeout: 180s                                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Output: Structured JSON                                        │
│  {                                                               │
│    "entities": [Person, Object, Action],                        │
│    "relationships": [CARRIES, USES, NEAR, ...],                 │
│    "scene_intelligence": {risk, intent, anomalies},             │
│    "behavioral_assessment": {...},                              │
│    "knowledge_graph": {nodes, relationships, interactions}      │
│  }                                                               │
└──────────────────────────────────────────────────────────────────┘
```

**Key Features:**
- **Quantized Model**: 4-bit (q4_K_M) for 4x memory reduction, minimal accuracy loss
- **CV Context Injection**: YOLO detections guide VLM attention
- **Fallback Mechanisms**: JSON repair, retry logic, CV-based enrichment
- **Minimum Relationship Guarantee**: Forces 3-5 relationships per frame

---

### 3.3 Knowledge Graph Construction (`modules/neo4j_kg.py`)

**Purpose:** Build research-grade, interconnected semantic graph

**Graph Schema:**
```
┌─────────────────────────────────────────────────────────────────┐
│  Neo4j Knowledge Graph (bolt://localhost:7687)                  │
│                                                                  │
│  NODE TYPES (10):                                               │
│  ┌────────────┐  ┌──────────────┐  ┌─────────┐  ┌──────────┐  │
│  │   Video    │─▶│ FrameSummary │◀─│ Person  │  │  Object  │  │
│  │  (anchor)  │  │    (hub)     │  │         │  │          │  │
│  └────────────┘  └──────────────┘  └─────────┘  └──────────┘  │
│                                                                  │
│  ┌─────────┐  ┌──────────┐  ┌────────────┐  ┌──────────────┐  │
│  │  Scene  │  │  Action  │  │    Zone    │  │     Risk     │  │
│  └─────────┘  └──────────┘  └────────────┘  └──────────────┘  │
│                                                                  │
│  ┌───────────┐  ┌────────────────┐                             │
│  │  Anomaly  │  │ RiskAssessment │                             │
│  └───────────┘  └────────────────┘                             │
│                                                                  │
│  RELATIONSHIP TYPES (15+):                                      │
│  Person ↔ Object:   CARRIES, USES, TOUCHES, OWNS, NEAR         │
│  Person ↔ Person:   NEAR, MEETS_WITH, FOLLOWS                  │
│  Person ↔ Action:   PERFORMS                                   │
│  Person ↔ Zone:     LOCATED_IN                                 │
│  Object ↔ Zone:     PLACED_IN                                  │
│  Action ↔ Object:   TARGETS                                    │
│  Frame ↔ Frame:     NEXT_FRAME                                 │
│  Person ↔ Person:   CONTINUES_AS (cross-frame tracking)        │
│  Risk ↔ Summary:    CONTRIBUTES_TO                             │
│  Anomaly ↔ Summary: DETECTED_IN                                │
│  Video ↔ Frame:     HAS_FRAME, CONTAINS                        │
│  Video ↔ Zone:      HAS_ZONE                                   │
│  Scene ↔ Zone:      COVERS                                     │
│                                                                  │
│  CONNECTIVITY MECHANISMS (13):                                  │
│  1. Video hub (master anchor for all data)                     │
│  2. FrameSummary hub (per-frame aggregation)                   │
│  3. Person → Zone (spatial location)                           │
│  4. Object → Zone (spatial placement)                          │
│  5. Person → Object (ownership/interaction)                    │
│  6. Action → Person (actor linkage)                            │
│  7. Action → Object (action target)                            │
│  8. Temporal sequences (NEXT_FRAME)                            │
│  9. Person-Person proximity (NEAR in same zone)                │
│  10. Person-Object proximity (spatial distance <200px)         │
│  11. Cross-frame tracking (CONTINUES_AS for track_id)          │
│  12. Zone → Video hub (HAS_ZONE)                               │
│  13. Scene → Zone (coverage mapping)                           │
│                                                                  │
│  FALLBACK RELATIONSHIP CREATION:                                │
│  If VLM returns 0 relationships, automatically creates:         │
│  - Spatial proximity connections (zone-based)                   │
│  - Person-Action PERFORMS links                                 │
│  - Person-Object INTERACTS_WITH (distance-based)                │
│  - Person-Person NEAR (same zone)                               │
└──────────────────────────────────────────────────────────────────┘
```

**Key Functions:**
- `export_surveillance_graph()`: CV event import
- `push_vlm_kg_to_neo4j()`: VLM entity/relationship creation + fallback
- `push_vlm_analysis_summary()`: Risk/anomaly chains, assessment nodes
- `_build_research_grade_kg()`: CV-based spatial relationship inference

---

### 3.4 Hybrid RAG System (`rag/`)

**Purpose:** Natural language query interface over multi-modal data

**Architecture:**
```
┌─────────────────────────────────────────────────────────────────┐
│  Hybrid RAG Engine                                              │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  1. JsonRAG (Vector Similarity)                           │ │
│  │     - Index: FAISS (L2 distance)                          │ │
│  │     - Embeddings: bge-m3 (Ollama)                         │ │
│  │     - Documents: JSON files (events, frames, VLM outputs) │ │
│  │     - Retrieval: k-NN search (top-k=5)                    │ │
│  │     - LLM Synthesis: Gemma 2b                             │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  2. GraphRAG (Symbolic Reasoning)                         │ │
│  │     - Query: Natural language → Cypher translation        │ │
│  │     - Subgraph Extraction: Multi-hop traversal            │ │
│  │     - Reasoning: Chain-of-thought over graph facts        │ │
│  │     - Evidence: Node/relationship citations               │ │
│  │     - Confidence: Graph pattern matching score            │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  3. Hybrid Fusion                                         │ │
│  │     - Parallel execution: JsonRAG + GraphRAG              │ │
│  │     - Confidence blending: (graph_conf + json_conf) / 2   │ │
│  │     - Evidence merging: Graph reasoning + JSON snippets   │ │
│  │     - Answer synthesis: Combined insights                 │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Query Types Supported:                                         │
│  - Surveillance queries: "Who was running?"                     │
│  - Interaction queries: "Show person-object interactions"       │
│  - Risk queries: "What suspicious activities detected?"         │
│  - Temporal queries: "What happened before X?"                  │
│  - Spatial queries: "Activity in Zone1?"                        │
│  - Behavioral queries: "Who was loitering?"                     │
└──────────────────────────────────────────────────────────────────┘
```

**Endpoints:**
- `/rag/search`: JsonRAG (vector similarity)
- `/rag/graph`: GraphRAG (symbolic reasoning)
- `/rag/hybrid`: Combined approach

---

### 3.5 Backend Orchestration (`app.py`)

**Purpose:** Flask-based API for pipeline execution and data serving

**Pipeline Flow:**
```
POST /pipeline/upload
│
├─▶ 1. Save video file
│
├─▶ 2. Clear old data (JSON, frames)
│
├─▶ 3. Run CV Pipeline
│   └─▶ Output: events.json (496 events), salient frames
│
├─▶ 4. Select top 3 frames (activity-based)
│   └─▶ Draw YOLO annotations
│
├─▶ 5. VLM Analysis (per salient frame)
│   └─▶ Output: frame_X_vlm.json, frame_X_cv.json
│
├─▶ 6. Push to Neo4j
│   ├─▶ push_vlm_kg_to_neo4j() → Entities + Relationships
│   └─▶ push_vlm_analysis_summary() → Risk/Anomaly chains
│
├─▶ 7. Build RAG indices
│   ├─▶ JsonRAG: FAISS vector store
│   └─▶ GraphRAG: Neo4j connection ready
│
└─▶ 8. Return results to dashboard
    └─▶ JSON: {salient_frames, vlm_results, insights, cv_events}
```

**Key Endpoints:**
- `POST /pipeline/upload`: Video processing
- `GET /rag/search?q=...`: JsonRAG query
- `GET /rag/graph?q=...`: GraphRAG query
- `GET /rag/hybrid?q=...`: Hybrid query
- `GET /json_outputs/<file>`: Serve JSON data
- `GET /static/frames/<image>`: Serve frame images

---

### 3.6 Frontend Dashboard (`templates/dashboard.html` + `static/js/`)

**Purpose:** Interactive analytics and query interface

**Components:**
```
┌────────────────────────────────────────────────────────────────┐
│  Dashboard UI                                                   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Analytics Overview                                      │   │
│  │  - Total Frames, Persons, Objects, Events               │   │
│  │  - Risk Level (HIGH/MEDIUM/LOW)                         │   │
│  │  - Salient frame count                                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Salient Frames (Top 3)                                 │   │
│  │  - Annotated images with YOLO bounding boxes           │   │
│  │  - Frame metadata: ID, timestamp, person/object counts  │   │
│  │  - Detected objects list                                │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Event Insights & Analytics                             │   │
│  │  - Total Events: 496                                    │   │
│  │  - Motion State Distribution:                           │   │
│  │    * STATIONARY: 222 (47.3%)                            │   │
│  │    * WALKING: 79 (16.8%)                                │   │
│  │    * MOVING: 59 (12.6%)                                 │   │
│  │    * LOITERING: 109 (23.2%)                             │   │
│  │  - Priority Breakdown: High/Medium/Low                  │   │
│  │  - Speed Analysis: Avg/Max/Min                          │   │
│  │  - Zone Hotspots: Zone1 (469 events)                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  VLM Analysis (Tabs)                                    │   │
│  │  ├─ Scene: Environment, lighting, time of day           │   │
│  │  ├─ Behavioral: Risk level, anomalies (with ratings)    │   │
│  │  └─ Knowledge Graph: Nodes (25), Relationships (0→X)    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Natural Language Query Interface                        │   │
│  │  - Input: Free-text question                            │   │
│  │  - Mode selector: JsonRAG / GraphRAG / Hybrid           │   │
│  │  - Output: Answer + confidence + evidence               │   │
│  │  - Recent queries history                               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Charts (Chart.js 4.4.0)                                │   │
│  │  - Motion Timeline (line chart)                         │   │
│  │  - Event Distribution (doughnut)                        │   │
│  │  - Zone Activity (bar chart)                            │   │
│  │  - Event Timeline (scatter)                             │   │
│  └─────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────┘
```

**Key Features:**
- **Auto-refresh**: Fetches `events.json` on page load and after upload
- **Object display**: Shows actual class names (laptop, bag, phone)
- **Real-time updates**: Event insights refresh automatically
- **Error handling**: Graceful degradation if charts.js unavailable

---

## 4. Data Flow Diagram

```
┌───────────┐
│   VIDEO   │
│   FILE    │
└─────┬─────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1: MOTION GATING                                     │
│  Input: Raw video frames (1280x720 @ 25fps)                │
│  Process: Frame differencing → Activity score               │
│  Output: Filtered frames (~50% reduction)                   │
│  Efficiency: Skip low-activity frames (score < 2.0)         │
└─────┬───────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2: OBJECT DETECTION                                  │
│  Model: YOLOv8n (nano) - 80 COCO classes                    │
│  Input: Active frames                                       │
│  Output: Bounding boxes [x1,y1,x2,y2], class, confidence    │
│  Detections: Person, car, laptop, bag, phone, etc.          │
└─────┬───────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 3: MULTI-OBJECT TRACKING                             │
│  Algorithm: DeepSORT (IoU-based + Kalman filtering)         │
│  Input: Detections (persons only)                           │
│  Output: track_id, bbox, positions[], motion_state, speed   │
│  Features: ID persistence (max_age=30 frames)               │
└─────┬───────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 4: ZONE ANALYSIS                                     │
│  Grid: 9-zone spatial division (3x3)                        │
│  Input: Track positions                                     │
│  Output: Zone assignment per track/object                   │
│  Data: Zone occupancy, activity heatmap                     │
└─────┬───────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 5: EVENT GENERATION                                  │
│  FSM: Track motion states over time                         │
│  Events: person_moving, person_stationary,                  │
│          person_loitering (>10s), person_walking            │
│  Output: events.json (496 events)                           │
│  Metadata: type, track_id, frame_id, timestamp,             │
│           motion_state, speed_px_s, zone, priority, bbox    │
└─────┬───────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 6: SALIENT FRAME SELECTION                           │
│  Scoring: person_count*10 + object_count*5 +                │
│           event_count*15 + motion_magnitude*2               │
│  Selection: Top 3-5 frames with temporal diversity          │
│  Constraint: Minimum 10-frame spacing                       │
│  Output: High-activity frames for VLM analysis              │
└─────┬───────────────────────────────────────────────────────┘
      │
      ├──────────────────┬─────────────────────────────────────┤
      │                  │                                     │
      ▼                  ▼                                     ▼
┌──────────────┐  ┌──────────────┐                    ┌─────────────┐
│ frame_X.jpg  │  │ frame_X_     │                    │  events.    │
│ (annotated)  │  │ cv.json      │                    │  json       │
└──────────────┘  └──────────────┘                    └─────────────┘
      │                  │                                     │
      └──────────────────┴─────────────────────────────────────┤
                         │                                     │
                         ▼                                     │
      ┌──────────────────────────────────────────────────────┐│
      │  VLM ANALYSIS (Qwen3-VL)                             ││
      │  Input: Frame image + CV detection context          ││
      │  Process: Scene understanding, entity extraction,    ││
      │           relationship mining, risk assessment       ││
      │  Output: frame_X_vlm.json                            ││
      │  {                                                   ││
      │    entities: [Person, Object, Action],               ││
      │    relationships: [CARRIES, USES, NEAR],             ││
      │    scene_intelligence: {risk, intent, anomalies},    ││
      │    knowledge_graph: {nodes, relationships}           ││
      │  }                                                   ││
      └──────────────────┬───────────────────────────────────┘│
                         │                                     │
                         └─────────────────────────────────────┤
                                                               │
                         ┌─────────────────────────────────────▼────┐
                         │  NEO4J KNOWLEDGE GRAPH                    │
                         │  - Video hub (anchor)                     │
                         │  - FrameSummary nodes (per frame)         │
                         │  - Person/Object/Action/Zone nodes        │
                         │  - Risk/Anomaly/Assessment nodes          │
                         │  - 15+ relationship types                 │
                         │  - Cross-frame tracking (CONTINUES_AS)    │
                         │  - Spatial proximity (NEAR)               │
                         │  - Ownership chains (CARRIES, USES, OWNS) │
                         └───────────────┬───────────────────────────┘
                                         │
                         ┌───────────────┴───────────────────────────┐
                         │                                           │
                         ▼                                           ▼
              ┌────────────────────┐                    ┌──────────────────┐
              │  JsonRAG           │                    │  GraphRAG        │
              │  - FAISS index     │                    │  - Cypher query  │
              │  - BGE-M3 embed    │                    │  - Multi-hop     │
              │  - k-NN retrieval  │                    │  - Chain-of-     │
              │  - Gemma 2b LLM    │                    │    thought       │
              └────────────────────┘                    └──────────────────┘
                         │                                           │
                         └───────────────┬───────────────────────────┘
                                         │
                                         ▼
                              ┌────────────────────┐
                              │  DASHBOARD UI      │
                              │  - Analytics       │
                              │  - Visualizations  │
                              │  - Query interface │
                              └────────────────────┘
```

---

## 5. Technology Stack

### Core Technologies

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| **Backend** | Python | 3.10+ | Core runtime |
| | Flask | 2.3.x | Web framework |
| | Flask-CORS | 4.0.x | Cross-origin requests |
| **Computer Vision** | Ultralytics YOLO | v8 | Object detection |
| | OpenCV | 4.8+ | Image processing |
| | NumPy | 1.24+ | Numerical operations |
| **VLM** | Ollama | Latest | Model serving |
| | Qwen3-VL | 2b-instruct-q4_K_M | Vision-language model |
| **Database** | Neo4j | 5.x | Knowledge graph |
| | neo4j-driver | 5.14+ | Python client |
| **RAG** | FAISS | 1.7+ | Vector similarity search |
| | sentence-transformers | Latest | Text embeddings |
| **Frontend** | HTML5/CSS3/JS | - | UI framework |
| | Chart.js | 4.4.0 | Data visualization |
| **DevOps** | Docker | 20.10+ | Neo4j containerization |
| | Bash | - | Deployment scripts |

### Model Details

**YOLOv8n:**
- Parameters: 3.2M
- Input: 640x640
- Inference: ~3ms (GPU), ~30ms (CPU)
- Classes: 80 COCO classes

**Qwen3-VL 2b-instruct-q4_K_M:**
- Parameters: 2B (4-bit quantized)
- Memory: ~1.5GB VRAM
- Inference: ~5-10s per frame
- Quantization: q4_K_M (mixed precision)

**BGE-M3 (JsonRAG embeddings):**
- Dimensions: 1024
- Context: 8192 tokens
- Language: Multilingual

**Gemma 2b (JsonRAG synthesis):**
- Parameters: 2.5B
- Context: 8K tokens
- Provider: Ollama

---

## 6. Deployment Architecture

### Local Development Setup

```
Host Machine (Ubuntu 18.04+)
├─ Python 3.10 (conda base environment)
├─ Ollama (localhost:11434)
│  ├─ qwen3-vl:2b-instruct-q4_K_M
│  ├─ bge-m3:latest
│  └─ gemma:2b
├─ Docker (Neo4j container)
│  └─ neo4j:5.x (bolt://localhost:7687, http://localhost:7474)
├─ Flask (localhost:5000)
└─ File System
   └─ /home/admin-/Desktop/sanjaya/sanjaya-video-analytics/
      ├─ backend/
      │  ├─ uploads/ (video files)
      │  ├─ static/frames/ (annotated images)
      │  ├─ json_outputs/ (events, CV/VLM data)
      │  ├─ cv_pipeline/
      │  ├─ modules/
      │  ├─ rag/
      │  ├─ templates/
      │  └─ app.py
      └─ frontend/ (optional separate deployment)
```

### Production Considerations

**Scalability:**
- Horizontal scaling: Multiple Flask workers (Gunicorn)
- Task queue: Celery for async video processing
- Object storage: S3/MinIO for frames/videos
- Neo4j clustering: Causal cluster for high availability

**Security:**
- Authentication: JWT tokens for API
- Authorization: Role-based access (admin, analyst, viewer)
- Neo4j: Strong password, encrypted connections
- Input validation: File size limits, format checks

**Monitoring:**
- Logging: Structured logging (JSON format)
- Metrics: Prometheus + Grafana
- Tracing: Distributed tracing (Jaeger)
- Alerts: Pipeline failures, high latency

---

## 7. Performance Characteristics

### Processing Metrics (Sample Video: 25fps, 3 minutes)

| Stage | Time | Throughput | Bottleneck |
|-------|------|-----------|-----------|
| Motion Gating | 0.5s | 50 fps | Frame I/O |
| YOLO Detection | 2-5s | 10-25 fps | GPU inference |
| DeepSORT Tracking | 1s | 100 fps | CPU (IoU calc) |
| Zone Analysis | 0.2s | 200 fps | CPU |
| Event Generation | 0.3s | 150 fps | CPU |
| VLM Analysis (3 frames) | 15-30s | 0.1-0.2 fps | VLM inference |
| Neo4j Insert | 2-5s | - | Network + Cypher |
| RAG Indexing | 1-3s | - | Embedding generation |
| **Total Pipeline** | **25-50s** | **~1.5 fps** | **VLM (dominant)** |

### Resource Usage

**Memory:**
- Flask + CV Pipeline: ~800MB
- YOLOv8n model: ~6MB
- Qwen3-VL (4-bit): ~1.5GB VRAM
- Neo4j: ~500MB-2GB (depends on graph size)
- FAISS index: ~100MB (for 500 documents)

**CPU:**
- CV pipeline: 2-4 cores (70-90% utilization)
- Flask: 1 core (20-30%)

**GPU:**
- YOLOv8n: 10-20% utilization (intermittent)
- Qwen3-VL: 80-100% during inference

**Disk:**
- Video: 50-200MB per minute
- Frames: 200-500KB per annotated frame
- JSON outputs: 50-200KB per frame
- Neo4j database: 10-50MB for 3-minute video

---

## 8. API Reference

### Video Upload

```http
POST /pipeline/upload
Content-Type: multipart/form-data

Parameters:
  file: video file (mp4, avi, mov)

Response:
{
  "status": "success",
  "salient_frames": [
    {
      "frame_id": 43,
      "timestamp": 1.72,
      "saliency": 0.85,
      "persons": 3,
      "objects": 2,
      "image_url": "/static/frames/salient_0_frame43.jpg",
      "image_path": "/absolute/path/to/frame.jpg"
    }
  ],
  "vlm_results": [
    {
      "frame_id": 43,
      "timestamp": 1.72,
      "scene": {
        "type": "indoor",
        "lighting": "bright",
        "time_of_day": "afternoon"
      },
      "behavioral_assessment": {
        "risk_level": "low",
        "inferred_intent": "normal_activity"
      },
      "knowledge_graph": {
        "nodes": [...],
        "relationships": [...]
      }
    }
  ],
  "cv_events": 496,
  "insights": {
    "risks": [],
    "anomalies": [],
    "overall_risk": "low",
    "detected_objects": ["laptop", "bag", "phone"]
  }
}
```

### RAG Queries

```http
GET /rag/search?q=Who+was+running?&k=5

Response:
{
  "answer": "Track-4 was running at 6-7 px/s across multiple frames...",
  "confidence": 0.85,
  "evidence": [
    {
      "file": "events.json",
      "snippet": "Track-4 was MOVING at 6.33 px/s in Zone1"
    }
  ],
  "sources": ["events.json", "frame_43_cv.json"],
  "type": "json_rag"
}
```

```http
GET /rag/graph?q=Show+person-object+interactions

Response:
{
  "answer": "P1 uses laptop, P2 carries bag...",
  "chain_of_thought": [
    {
      "step": 1,
      "reasoning": "Identified 3 persons in graph",
      "findings": ["P1 in Zone1", "P2 in Zone1"]
    }
  ],
  "reasoning_path": "P1→USES→Obj_Laptop_1→PLACED_IN→Zone1",
  "evidence": [
    {
      "type": "relationship",
      "source": "P1",
      "target": "Obj_Laptop_1",
      "type": "USES"
    }
  ],
  "confidence": 0.9,
  "type": "graph_rag"
}
```

---

## 9. Configuration

### Environment Variables

```bash
# Ollama Configuration
OLLAMA_URL=http://localhost:11434
VISION_MODEL=qwen3-vl:2b-instruct-q4_K_M

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=neo4j123

# Telegram Notifications (Optional)
TELEGRAM_BOT_TOKEN=YOUR_BOT_TOKEN
TELEGRAM_CHAT_ID=YOUR_CHAT_ID

# Flask Configuration
FLASK_DEBUG=true
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
```

### Model Configuration

**YOLOv8n** (`cv_pipeline/detector.py`):
```python
model = YOLO("yolov8n.pt")
conf_threshold = 0.5  # Detection confidence
```

**DeepSORT** (`cv_pipeline/tracking.py`):
```python
max_age = 30        # Track persistence (frames)
iou_threshold = 0.3  # Minimum IoU for matching
```

**Motion Gating** (`cv_pipeline/motion_gating.py`):
```python
threshold = 2.0  # Activity score threshold
```

**Zone Analyzer** (`cv_pipeline/zones.py`):
```python
grid_size = (3, 3)  # 9-zone grid
```

---

## 10. Troubleshooting Guide

### Common Issues

**1. Neo4j shows 0 relationships**
- **Cause**: VLM returned no relationships, fallback not triggered
- **Solution**: Code now auto-creates spatial fallback connections
- **Verify**: `MATCH (n)-[r]-(m) RETURN count(r)` should return >0

**2. Event insights not updating**
- **Cause**: Dashboard caching old events.json
- **Solution**: Hard refresh (Ctrl+F5) after upload
- **Code fix**: Now fetches fresh events.json with 800ms delay

**3. "0[object Object]" in dashboard**
- **Cause**: Object counting bug in updateStats()
- **Solution**: Fixed to collect actual class names (laptop, bag, etc.)

**4. VLM timeout errors**
- **Cause**: Qwen3-VL inference too slow
- **Solution**: Increase timeout to 180s, use 4-bit quantization

**5. YOLO detection misses persons**
- **Cause**: Low confidence threshold
- **Solution**: Lower conf_threshold to 0.3-0.4

---

## 11. Future Enhancements

### Short-term (Q1 2026)
- [ ] Real-time streaming support (RTSP/WebRTC)
- [ ] Multi-camera fusion
- [ ] Action recognition (fight detection, fall detection)
- [ ] Face recognition integration
- [ ] License plate recognition (LPR)

### Medium-term (Q2-Q3 2026)
- [ ] Cloud deployment (AWS/Azure/GCP)
- [ ] Mobile app (React Native)
- [ ] Advanced anomaly detection (autoencoders)
- [ ] Predictive analytics (future event forecasting)
- [ ] 3D scene reconstruction

### Long-term (Q4 2026+)
- [ ] Edge deployment (Jetson Nano, Raspberry Pi)
- [ ] Federated learning across multiple sites
- [ ] Explainable AI (SHAP, LIME for VLM decisions)
- [ ] Regulatory compliance (GDPR, privacy masking)
- [ ] Integration with physical security systems (alarms, access control)

---

## 12. Research Contributions

**Novel Aspects:**
1. **Multi-Modal Fusion**: Tight integration of CV (quantitative) + VLM (semantic) + KG (symbolic)
2. **Research-Grade Knowledge Graph**: 15+ relationship types with 13 connectivity mechanisms
3. **Hybrid RAG**: Vector similarity + graph reasoning for complementary retrieval
4. **Fallback Relationship Creation**: Guarantees connected graph even when VLM fails
5. **Efficiency Optimization**: Motion gating + edge-optimized models (YOLOv8n + 4-bit VLM)

**Potential Applications:**
- Retail analytics (customer behavior, theft detection)
- Smart city surveillance (traffic monitoring, crowd management)
- Industrial safety (PPE detection, hazard identification)
- Healthcare monitoring (patient fall detection, activity tracking)
- Smart home security (intrusion detection, activity recognition)

---

## 13. License & Credits

**License:** MIT (modify as needed)

**Core Technologies:**
- Ultralytics YOLOv8: AGPL-3.0
- Qwen3-VL: Apache-2.0
- Neo4j: GPL-3.0 (Community Edition)
- Flask: BSD-3-Clause
- Chart.js: MIT

**Developed by:** Sanjaya Team  
**Contact:** [Add contact information]  
**GitHub:** [Add repository URL]

---

## Appendix A: File Structure

```
sanjaya-video-analytics/
├── backend/
│   ├── app.py                          # Flask orchestration (492 lines)
│   ├── config.py                       # Configuration variables
│   ├── rebuild_rag.py                  # RAG index rebuilding utility
│   ├── start_sanjaya.sh                # Startup script
│   ├── test_vlm.py                     # VLM testing utility
│   ├── yolov8n.pt                      # YOLO model weights
│   │
│   ├── cv_pipeline/
│   │   ├── __init__.py
│   │   ├── pipeline.py                 # CV pipeline orchestrator (220 lines)
│   │   ├── detector.py                 # YOLO wrapper (100 lines)
│   │   ├── tracking.py                 # DeepSORT tracker (118 lines)
│   │   ├── zones.py                    # Zone analyzer (120 lines)
│   │   ├── event_detection.py          # Event generator (180 lines)
│   │   ├── motion_gating.py            # Motion filter (100 lines)
│   │   └── roi_selection.py            # ROI selection (deprecated)
│   │
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── vlm_analyzer.py             # VLM integration (256 lines)
│   │   ├── neo4j_kg.py                 # KG construction (1192 lines)
│   │   ├── neo4j_manager.py            # Neo4j lifecycle management
│   │   ├── neo4j_query.py              # Cypher query utilities
│   │   ├── telegram_notifier.py        # Telegram notifications
│   │   └── tracking.py                 # Additional tracking utilities
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── json_rag.py                 # Vector similarity RAG (594 lines)
│   │   ├── graph_rag.py                # Graph-based RAG (302 lines)
│   │   ├── qa_service.py               # QA orchestration
│   │   ├── queries.py                  # Query templates (150 lines)
│   │   ├── prompts.py                  # Prompt engineering
│   │   ├── retriever.py                # Retrieval utilities (200 lines)
│   │   └── indexer.py                  # Document indexing (200 lines)
│   │
│   ├── static/
│   │   ├── css/
│   │   │   └── dashboard.css           # Dashboard styling
│   │   ├── js/
│   │   │   ├── dashboard.js            # Dashboard logic (654 lines)
│   │   │   ├── charts.js               # Chart.js visualizations (355 lines)
│   │   │   ├── graph.js                # Neo4j graph visualization
│   │   │   └── rag.js                  # RAG query interface
│   │   ├── frames/                     # Annotated frame images
│   │   └── videos/                     # Uploaded videos
│   │
│   ├── templates/
│   │   ├── dashboard.html              # Main dashboard UI (386 lines)
│   │   ├── index.html                  # Landing page
│   │   └── test_events.html            # Event testing page
│   │
│   ├── uploads/                        # Video file storage
│   └── json_outputs/                   # Processing outputs
│       ├── events.json                 # 496 events
│       ├── cv_stats.json               # CV pipeline statistics
│       ├── frame_X_cv.json             # CV metadata per frame
│       └── frame_X_vlm.json            # VLM analysis per frame
│
├── frontend/                           # Optional separate frontend
│   └── (unused in current deployment)
│
└── SYSTEM_ARCHITECTURE.md              # This document
```

---

**Document Version:** 1.0  
**Last Updated:** December 25, 2025  
**Total Lines of Code:** ~8,500+ lines (Python + JavaScript + HTML/CSS)
