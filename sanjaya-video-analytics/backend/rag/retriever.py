import json
from .queries import GraphClient, q_persons_interacting_with_object, q_gestures_toward, q_activity_participants
from .indexer import JsonIndexer

class HybridRetriever:
    def __init__(self, video_name):
        self.gc = GraphClient()
        self.video = video_name
        self.indexer = JsonIndexer()
        self.indexer.build()

    def retrieve(self, question: str):
        # semantic hits from JSON
        json_hits = self.indexer.search(question, k=5)
        # structural graph snippets based on question heuristics
        graph_hits = []
        if "laptop" in question or "object" in question:
            graph_hits += q_persons_interacting_with_object(self.gc, self.video, object_class="laptop", min_conf=0.7)
        if "gesture" in question or "point" in question:
            graph_hits += q_gestures_toward(self.gc, self.video, min_conf=0.7)
        if "activity" in question or "meeting" in question:
            graph_hits += q_activity_participants(self.gc, self.video)
        # package
        evidence_json = [{"type":"json","path":h["path"],"snippet":h["text"][:300],"score":h["score"]} for h in json_hits]
        evidence_graph = [{"type":"graph","snippet":json.dumps(g)} for g in graph_hits]
        return {"json": evidence_json, "graph": evidence_graph}