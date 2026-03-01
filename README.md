# Sanjaya — Semantic Graph-Based Framework for Intelligent Video Analytics and Advanced Surveillance

> *"Sanjaya had divine vision — the ability to see and narrate the entire Kurukshetra war to the blind king Dhritarashtra. This system aspires to grant that same all-seeing intelligence to modern surveillance."*

**Project Code: 5125** | AMD Slingshot Hackathon — AI for Smart Cities

---

## Overview

Sanjaya transforms raw, unstructured surveillance video into **structured, queryable intelligence** by fusing three AI paradigms:

| Paradigm | Technology | Role |
|---|---|---|
| Computer Vision | YOLOv8n + DeepSORT | Detect, track, quantify |
| Vision-Language Model | Qwen3-VL 4-bit (Ollama) | Understand, describe, assess risk |
| Knowledge Graph | Neo4j 5.x | Store, relate, reason |
| Hybrid RAG | FAISS + Cypher | Query in natural language |

---

## The Problem

Modern cities have thousands of cameras generating terabytes of footage daily — **95% goes unreviewed**.

Current surveillance systems:
- Offer only frame-level detection (bounding boxes + class labels)
- Cannot answer *who did what, with whom, where, and why*
- Require hours of manual review for post-incident analysis
- Provide no natural-language query capability
- Give no explainable reasoning for anomaly alerts

**The gap is not detection accuracy — it is the lack of structured, explainable understanding.**

---

## Architecture

```
┌─────────────┐      ┌─────────────────────────────────────────────┐
│   VIDEO     │      │           6-STAGE CV PIPELINE                │
│   UPLOAD    │─────▶│  1. Motion Gating    (~50% frame reduction)  │
│  (Flask)    │      │  2. YOLOv8n Detection (80 COCO classes)      │
└─────────────┘      │  3. DeepSORT Tracking (IoU + Kalman)         │
                     │  4. 9-Zone Spatial Analysis                   │
                     │  5. FSM Event Detection (loiter/move/stationary) │
                     │  6. Salient Frame Selection (activity score)  │
                     └─────────────────┬───────────────────────────┘
                                       │ Top-K frames
                     ┌─────────────────▼───────────────────────────┐
                     │   Qwen3-VL 4-bit (via Ollama)                │
                     │   - Entity extraction (Person/Object/Action)  │
                     │   - Relationship mining (15+ types)           │
                     │   - Risk/anomaly assessment (1–10 scale)      │
                     └─────────────────┬───────────────────────────┘
                                       │
              ┌────────────────────────┴──────────────────────────┐
              │                                                    │
   ┌──────────▼──────────┐                        ┌──────────────▼────────────┐
   │   Neo4j KG           │                        │   Hybrid RAG              │
   │   10 node types      │                        │   JsonRAG (FAISS/BGE-M3)  │
   │   15+ relationships  │                        │   GraphRAG (Cypher)       │
   │   13 connectivity    │                        │   Fusion (RRF blending)   │
   └──────────┬──────────┘                        └──────────────┬────────────┘
              │                                                    │
              └──────────────────────┬─────────────────────────────┘
                                     ▼
                     ┌───────────────────────────────┐
                     │   Dashboard + NL Query UI      │
                     │   "Who was loitering near Z3?" │
                     │   "Show all risk events"        │
                     └───────────────────────────────┘
```

---

## Key Features

- **Motion Gating** — Frame differencing skips low-activity frames, cutting compute cost ~50%
- **Semantic Event Detection** — FSM-based states: STATIONARY, WALKING, MOVING, LOITERING
- **Rich Knowledge Graph** — 10 node types, 15+ relationship types, cross-frame tracking via `CONTINUES_AS`
- **Hybrid RAG** — Vector similarity (FAISS) + symbolic graph reasoning (Cypher) in parallel
- **Edge-Ready** — 4-bit quantized VLM (~1.5GB VRAM), runs on Jetson Nano
- **Natural Language Queries** — Ask questions over surveillance data, get cited answers
- **Explainable Alerts** — Every risk flag has a reasoning chain traceable through the graph

---

## Tech Stack

```
Backend:     Python 3.10, Flask, OpenCV
CV:          YOLOv8n (Ultralytics), DeepSORT (IoU tracker)
VLM:         Qwen3-VL 2b-instruct-q4_K_M via Ollama
Embeddings:  BGE-M3 (1024-dim, multilingual)
Vector DB:   FAISS (L2 index)
Graph DB:    Neo4j 5.x (bolt://localhost:7687)
LLM (RAG):   Gemma 2b via Ollama
Frontend:    HTML5, Chart.js, JavaScript
Infra:       Docker (Neo4j), Conda environment
```

---

## Setup

### Prerequisites
- Python 3.10+ (conda recommended)
- Docker
- [Ollama](https://ollama.com) installed

### 1. Pull required models via Ollama
```bash
ollama pull qwen3-vl:2b-instruct-q4_K_M
ollama pull bge-m3:latest
ollama pull gemma:2b
```

### 2. Start Neo4j
```bash
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/neo4j123 \
  neo4j:5.15-community
```

### 3. Install dependencies
```bash
cd sanjaya-video-analytics/backend
pip install -r requirements.txt
```

### 4. Download YOLO model weights
```bash
# YOLOv8n (~6MB) — downloaded automatically by Ultralytics on first run
# OR manually place yolov8n.pt in backend/
```

### 5. Run
```bash
cd sanjaya-video-analytics/backend
python app.py
```

Open: [http://localhost:5000](http://localhost:5000)
Neo4j Browser: [http://localhost:7474](http://localhost:7474) (neo4j / neo4j123)

---

## Usage

1. Upload a surveillance video via the dashboard
2. The 6-stage pipeline processes it automatically (~25–50s for a 3-min video)
3. View annotated salient frames, event timeline, and knowledge graph
4. Query the system in natural language:
   - *"What suspicious activities were detected?"*
   - *"Who was loitering near Zone 3?"*
   - *"Show all person-object interactions"*
   - *"What happened between 0:30 and 1:00?"*

---

## Knowledge Graph Schema

```
NODE TYPES (10):
  Video · FrameSummary · Person · Object · Scene
  Action · Zone · Risk · Anomaly · RiskAssessment

RELATIONSHIP TYPES (15+):
  Person ↔ Object:   CARRIES, USES, TOUCHES, OWNS, NEAR
  Person ↔ Person:   NEAR, MEETS_WITH, FOLLOWS, CONTINUES_AS
  Person ↔ Action:   PERFORMS
  Person ↔ Zone:     LOCATED_IN
  Object ↔ Zone:     PLACED_IN
  Frame ↔ Frame:     NEXT_FRAME
  Risk/Anomaly:      CONTRIBUTES_TO, DETECTED_IN
  Video ↔ Frame:     HAS_FRAME, CONTAINS
```

---

## Project Structure

```
sanjaya/
├── sanjaya-video-analytics/
│   ├── backend/
│   │   ├── app.py                  # Flask orchestrator
│   │   ├── config.py               # Configuration
│   │   ├── start_sanjaya.sh        # Startup script
│   │   ├── cv_pipeline/            # YOLOv8 + DeepSORT + motion gating
│   │   │   ├── pipeline.py
│   │   │   ├── detector.py
│   │   │   ├── tracking.py
│   │   │   ├── zones.py
│   │   │   ├── event_detection.py
│   │   │   ├── motion_gating.py
│   │   │   └── fsm.py
│   │   ├── modules/                # VLM + Neo4j KG + notifications
│   │   │   ├── vlm_analyzer.py
│   │   │   ├── neo4j_kg.py
│   │   │   └── telegram_notifier.py
│   │   ├── rag/                    # Hybrid RAG (JsonRAG + GraphRAG)
│   │   │   ├── json_rag.py
│   │   │   ├── graph_rag.py
│   │   │   └── qa_service.py
│   │   ├── templates/              # HTML dashboard
│   │   └── static/                 # CSS, JS, Chart.js
│   └── frontend/                   # Optional standalone frontend
├── SYSTEM_ARCHITECTURE.md          # Detailed architecture docs
└── README.md
```

---

## Impact

| Dimension | Impact |
|---|---|
| Operational | Reduces post-incident review from hours to seconds |
| Technical | First CV+VLM+KG+HybridRAG fusion for surveillance |
| Privacy | Fully on-premise, no cloud video transmission |
| Cost | Runs on ~$100 edge hardware (Jetson Nano) |
| Scale | REST API integrates with City Command Centre infrastructure |

---

## Team

Developed for AMD Slingshot Hackathon 2026 — AI for Smart Cities track.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
