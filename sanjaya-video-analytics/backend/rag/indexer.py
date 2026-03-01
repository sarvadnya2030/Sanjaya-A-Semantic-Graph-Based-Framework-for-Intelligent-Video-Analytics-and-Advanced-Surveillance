import os, json, numpy as np, requests
from config import OLLAMA_URL

EMBED_MODEL = "nomic-embed-text"  # run: ollama pull nomic-embed-text

class JsonIndexer:
    def __init__(self, json_dir="json_outputs"):
        self.json_dir = json_dir
        self.docs = []          # [{id, path, text, meta}]
        self.emb = None         # np.ndarray [N, D]

    def _collect_text(self, data):
        parts = []
        if isinstance(data, dict):
            parts.append(data.get("surveillance_narrative") or data.get("natural_description") or "")
            for n in data.get("nodes", []):
                props = n.get("properties", {})
                parts.append(f"{n.get('node_type')} {n.get('node_id')} {json.dumps(props, ensure_ascii=False)}")
            for e in data.get("edges", []):
                parts.append(f"{e.get('relation')} {e.get('source')}->{e.get('target')} {e.get('confidence')}")
        return "\n".join([p for p in parts if p])

    def _embed(self, texts):
        r = requests.post(f"{OLLAMA_URL}/api/embeddings", json={
            "model": EMBED_MODEL,
            "input": texts
        }, timeout=60)
        r.raise_for_status()
        vecs = r.json().get("embeddings", [])
        X = np.array(vecs, dtype=np.float32)
        # normalize for cosine similarity
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
        return X / norms

    def build(self):
        self.docs, texts = [], []
        if not os.path.isdir(self.json_dir):
            self.emb = None
            return
        for fn in os.listdir(self.json_dir):
            if not fn.endswith(".json"): continue
            path = os.path.join(self.json_dir, fn)
            try:
                data = json.load(open(path))
            except:
                continue
            text = self._collect_text(data)
            self.docs.append({
                "id": len(self.docs),
                "path": path,
                "text": text,
                "meta": {"video": data.get("video_name"), "frame": data.get("frame_id")}
            })
            texts.append(text)
        if texts:
            self.emb = self._embed(texts)

    def search(self, query, k=5):
        if self.emb is None or not len(self.docs):
            return []
        qv = self._embed([query])[0]  # [D]
        scores = (self.emb @ qv).tolist()  # cosine
        idx = np.argsort(scores)[::-1][:k]
        return [{"score": float(scores[i]), **self.docs[i]} for i in idx]