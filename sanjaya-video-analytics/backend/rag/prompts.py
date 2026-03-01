RAG_PROMPT = """You are a surveillance analyst. Use the provided graph facts and JSON snippets to answer the question.
Return JSON ONLY:
{
  "answer": "short, precise, surveillance-style",
  "insights": ["bullet 1", "bullet 2", "bullet 3"],
  "evidence": [{"type":"graph","snippet":""},{"type":"json","path":"","snippet":""}],
  "confidence": 0.0,
  "next_actions": ["optional follow-up queries or frames to inspect"]
}
Constraints:
- Ground every claim in provided facts.
- Be concise; avoid speculation.
"""

GRAPH_RAG_COT_PROMPT = """You are performing chain-of-thought reasoning over a surveillance knowledge graph.

QUESTION: {question}

GRAPH KNOWLEDGE:
{graph_facts}

Perform multi-hop reasoning:

STEP 1 - ENTITY IDENTIFICATION:
- Which persons, objects, zones, events are relevant?
- What are their key properties?

STEP 2 - RELATIONSHIP TRAVERSAL:
- How are these entities connected?
- What multi-hop paths exist?

STEP 3 - PATTERN SYNTHESIS:
- What patterns emerge from the connections?
- Are there suspicious sequences?

STEP 4 - CONCLUSION:
- What is the final answer?
- What evidence supports it?

Return JSON with full reasoning chain and evidence."""