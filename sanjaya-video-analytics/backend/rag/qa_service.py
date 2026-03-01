import json, os, logging, requests
from config import OLLAMA_URL
from .json_rag import JsonRAG
from .queries import GraphClient, q_objects

log = logging.getLogger("rag.qa_service")
PREFERRED_MODELS = ["qwen3:1.7b"]

_json_rag = None
def get_json_rag():
    global _json_rag
    if _json_rag is None:
        _json_rag = JsonRAG(json_dirs=["json_outputs", "static/frames"])
        _json_rag.build_index()
    return _json_rag

def _call_llm(prompt: str):
    for model in PREFERRED_MODELS:
        try:
            r = requests.post(f"{OLLAMA_URL}/api/generate", json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 384}
            }, timeout=60)
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except Exception as e:
            log.warning(f"LLM {model} failed: {e}")
    return ""

def _extract_json(txt: str):
    try:
        return json.loads(txt)
    except Exception:
        pass
    if "```" in txt:
        for seg in txt.split("```"):
            s = seg.strip()
            if s.lower().startswith("json"):
                s = s[4:].strip()
            try:
                return json.loads(s)
            except Exception:
                continue
    import re
    for m in re.findall(r"\{[\s\S]*\}", txt)[::-1]:
        try:
            return json.loads(m)
        except Exception:
            continue
    return None

def ask_question(video_name: str = "", question: str = "", mode: str = "hybrid"):
    if not question:
        return {"answer": "No question provided", "insights": [], "evidence": [], "confidence": 0.0}

    rag = get_json_rag()
    json_result = rag.ask(question, video_name=video_name, k=5)

    graph_result = None
    if mode in {"hybrid", "graph"}:
        try:
            gc = GraphClient()
            rows = q_objects(gc, video_name) if video_name else []
            if rows:
                objs = [f"{r.get('class','unknown')}({r.get('count',0)})" for r in rows]
                graph_result = {
                    "answer": "Objects detected: " + ", ".join(objs),
                    "insights": [f"Total classes: {len(rows)}"],
                    "evidence": [{"type": "graph", "snippet": json.dumps(rows[:5])}],
                    "confidence": 0.8
                }
        except Exception as e:
            log.warning(f"Graph query failed: {e}")

    if mode == "json":
        return json_result
    if mode == "graph":
        return graph_result or {"answer": "No graph data", "insights": [], "evidence": [], "confidence": 0.0}

    prompt = f"""
You are a system that outputs strict JSON only.
Question: {question}
JSON evidence: {json.dumps(json_result, ensure_ascii=False)}
Graph evidence: {json.dumps(graph_result or {}, ensure_ascii=False)}
Return exactly:
{{"answer":"...","insights":["..."],"evidence":[{{"type":"json|graph","snippet":"..."}}],"confidence":0.0}}
""".strip()

    txt = _call_llm(prompt)
    obj = _extract_json(txt) or {}

    if not isinstance(obj, dict):
        obj = {}
    obj.setdefault("answer", json_result.get("answer") or (graph_result or {}).get("answer") or "No evidence found")
    obj.setdefault("insights", json_result.get("insights", []))
    obj.setdefault("evidence", json_result.get("evidence", []))
    if obj.get("confidence") is None:
        obj["confidence"] = 0.6

    log.info(f"LLM synthesized answer: {obj.get('answer','')[:120]}")
    return obj