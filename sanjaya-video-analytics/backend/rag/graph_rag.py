from neo4j import GraphDatabase
import requests
import logging
import json

log = logging.getLogger("graph_rag")

class GraphRAG:
    """
    GraphRAG with Chain-of-Thought reasoning over Neo4j surveillance KG.
    Performs multi-hop traversal and semantic reasoning.
    """
    
    def __init__(self, uri="bolt://localhost:7687", auth=("neo4j", "neo4j123"), ollama_url="http://localhost:11434"):
        self.driver = GraphDatabase.driver(uri, auth=auth)
        self.ollama_url = ollama_url
        log.info("[GraphRAG] Initialized with Neo4j connection")
    
    def query_graph(self, cypher_query, params=None):
        """Execute Cypher query and return results."""
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query, params or {})
                return [record.data() for record in result]
        except Exception as e:
            log.error(f"[GraphRAG] Cypher query error: {e}")
            return []
    
    def get_relevant_subgraph(self, question):
        """
        Extract relevant subgraph based on question keywords.
        Uses semantic matching to find relevant nodes/relationships.
        """
        # Detect entities in question
        keywords = self._extract_keywords(question)
        
        log.info(f"[GraphRAG] Extracted keywords: {keywords}")
        
        # Build multi-hop Cypher query
        subgraph_queries = []
        
        # Query 1: Find persons with suspicious behavior
        if any(k in question.lower() for k in ["suspicious", "risk", "threat", "unusual", "anomaly"]):
            subgraph_queries.append({
                "name": "suspicious_behavior",
                "cypher": """
                MATCH (v:VLMAnalysis)
                WHERE v.risk_level IN ['medium', 'high']
                OPTIONAL MATCH (v)-[:ANALYZES]->(p:Person)
                OPTIONAL MATCH (p)-[r:PERFORMS|INTERACTS_WITH|NEAR]->(target)
                RETURN v, p, r, target
                LIMIT 10
                """
            })
        
        # Query 2: Find person interactions
        if any(k in question.lower() for k in ["person", "people", "who", "interaction", "meeting"]):
            subgraph_queries.append({
                "name": "person_interactions",
                "cypher": """
                MATCH (p1:Person)-[r:INTERACTS_WITH|NEAR|TALKING_TO]->(p2:Person)
                RETURN p1, r, p2
                LIMIT 10
                """
            })
        
        # Query 3: Find object interactions
        if any(k in question.lower() for k in ["object", "laptop", "bag", "item", "using", "holding"]):
            subgraph_queries.append({
                "name": "object_interactions",
                "cypher": """
                MATCH (p:Person)-[r:USES|INTERACTS_WITH|NEAR]->(o:Object)
                RETURN p, r, o
                LIMIT 10
                """
            })
        
        # Query 4: Zone activity
        if any(k in question.lower() for k in ["zone", "area", "location", "where"]):
            subgraph_queries.append({
                "name": "zone_activity",
                "cypher": """
                MATCH (p:Person)-[:LOCATED_IN]->(z:Zone)
                OPTIONAL MATCH (o:Object)-[:PLACED_IN]->(z)
                RETURN p, z, o
                LIMIT 15
                """
            })
        
        # Query 5: Temporal sequences (events over time)
        if any(k in question.lower() for k in ["sequence", "timeline", "happened", "before", "after", "then"]):
            subgraph_queries.append({
                "name": "temporal_sequence",
                "cypher": """
                MATCH (e:Event)
                OPTIONAL MATCH (p:Person)-[:TRIGGERS|PARTICIPATES_IN]->(e)
                RETURN e, p
                ORDER BY e.timestamp
                LIMIT 10
                """
            })
        
        # Query 6: Full VLM Analysis summary
        if any(k in question.lower() for k in ["summary", "overview", "describe", "what", "scene"]):
            subgraph_queries.append({
                "name": "vlm_summaries",
                "cypher": """
                MATCH (v:VLMAnalysis)
                OPTIONAL MATCH (v)-[:ANALYZES]->(p:Person)
                RETURN v, p
                LIMIT 5
                """
            })
        
        # Execute all queries
        graph_facts = []
        for query_def in subgraph_queries:
            results = self.query_graph(query_def["cypher"])
            graph_facts.append({
                "query_type": query_def["name"],
                "results": results,
                "count": len(results)
            })
            log.info(f"[GraphRAG] {query_def['name']}: {len(results)} results")
        
        return graph_facts
    
    def _extract_keywords(self, question):
        """Simple keyword extraction from question."""
        keywords = []
        question_lower = question.lower()
        
        # Entity types
        if "person" in question_lower or "people" in question_lower or "who" in question_lower:
            keywords.append("person")
        if "object" in question_lower or "laptop" in question_lower or "bag" in question_lower:
            keywords.append("object")
        if "zone" in question_lower or "area" in question_lower or "where" in question_lower:
            keywords.append("zone")
        if "event" in question_lower or "happened" in question_lower:
            keywords.append("event")
        
        # Behavioral indicators
        if any(word in question_lower for word in ["suspicious", "unusual", "anomaly", "threat"]):
            keywords.append("suspicious")
        if any(word in question_lower for word in ["risk", "danger", "alert"]):
            keywords.append("risk")
        if any(word in question_lower for word in ["interaction", "meeting", "talking"]):
            keywords.append("interaction")
        
        return keywords if keywords else ["general"]
    
    def chain_of_thought_reasoning(self, question, graph_facts):
        """
        LLM-powered chain-of-thought reasoning over graph facts.
        Uses Qwen to synthesize multi-hop insights.
        """
        # Format graph facts for LLM
        formatted_facts = self._format_graph_facts(graph_facts)
        
        # Chain-of-thought prompt
        cot_prompt = f"""You are a surveillance intelligence analyst performing chain-of-thought reasoning over a knowledge graph.

QUESTION: {question}

GRAPH KNOWLEDGE BASE:
{formatted_facts}

REASONING TASK:
Perform step-by-step chain-of-thought reasoning:

1. IDENTIFY: What entities/relationships are relevant to the question?
2. TRAVERSE: What multi-hop paths connect to the answer?
3. SYNTHESIZE: What insights emerge from combining graph facts?
4. CONCLUDE: What is the final answer with supporting evidence?

Return your reasoning as JSON:
{{
  "chain_of_thought": [
    {{"step": 1, "reasoning": "identified relevant entities...", "findings": ["P1 is in Z1", "P1 has high risk"]}},
    {{"step": 2, "reasoning": "traversed relationships...", "findings": ["P1 interacts with O_laptop", "O_laptop in Z1"]}},
    {{"step": 3, "reasoning": "synthesized insights...", "findings": ["P1's behavior indicates..."]}},
    {{"step": 4, "reasoning": "final conclusion...", "findings": ["Answer: ..."]}}
  ],
  "answer": "concise answer to the question",
  "evidence": [
    {{"type": "node", "id": "P1", "property": "risk_level", "value": "high"}},
    {{"type": "relationship", "source": "P1", "target": "O_laptop", "type": "INTERACTS_WITH"}}
  ],
  "confidence": 0.85,
  "reasoning_path": "P1 → LOCATED_IN → Z1 → CONTAINS → O_laptop → INDICATES → suspicious_activity"
}}

CRITICAL: Ground every step in the provided graph facts. Show your reasoning chain."""
        
        log.info(f"[GraphRAG] Calling LLM for chain-of-thought reasoning...")
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": "qwen2.5:7b",
                    "prompt": cot_prompt,
                    "stream": False,
                    "temperature": 0.2,
                    "options": {
                        "num_predict": 1000
                    }
                },
                timeout=60
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama error: {response.status_code}")
            
            llm_response = response.json().get("response", "")
            
            log.info(f"[GraphRAG] LLM response: {llm_response[:300]}")
            
            # Extract JSON
            import re
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                reasoning_result = json.loads(json_match.group())
            else:
                # Fallback
                reasoning_result = {
                    "chain_of_thought": [{"step": 1, "reasoning": "LLM response parsing failed", "findings": []}],
                    "answer": llm_response[:200],
                    "evidence": [],
                    "confidence": 0.5,
                    "reasoning_path": "unknown"
                }
            
            log.info(f"[GraphRAG] ✅ Chain-of-thought: {len(reasoning_result.get('chain_of_thought', []))} steps")
            
            return reasoning_result
            
        except Exception as e:
            log.error(f"[GraphRAG] LLM reasoning error: {e}")
            return {
                "chain_of_thought": [{"step": 1, "reasoning": f"Error: {e}", "findings": []}],
                "answer": "Unable to perform reasoning due to error",
                "evidence": [],
                "confidence": 0.0,
                "reasoning_path": "error"
            }
    
    def _format_graph_facts(self, graph_facts):
        """Format graph query results for LLM consumption."""
        formatted = ""
        
        for fact_group in graph_facts:
            query_type = fact_group["query_type"]
            results = fact_group["results"]
            
            formatted += f"\n## {query_type.upper()} ({len(results)} results)\n"
            
            for i, record in enumerate(results[:5], 1):  # Limit to top 5 per query
                formatted += f"{i}. "
                for key, value in record.items():
                    if isinstance(value, dict):
                        # Node or relationship
                        if 'properties' in value:
                            props = value['properties']
                            formatted += f"{value.get('labels', [value.get('type', 'Entity')])[0] if isinstance(value.get('labels'), list) else 'Entity'} "
                            formatted += f"{props.get('id', key)}: "
                            formatted += ", ".join([f"{k}={v}" for k, v in props.items() if k not in ['id', 'video', 'createdAt', 'lastUpdated']])
                            formatted += " | "
                formatted += "\n"
        
        return formatted
    
    def ask(self, question):
        """
        Main GraphRAG query interface with chain-of-thought reasoning.
        """
        log.info(f"[GraphRAG] Question: {question}")
        
        # Step 1: Get relevant subgraph
        graph_facts = self.get_relevant_subgraph(question)
        
        if not any(g['count'] > 0 for g in graph_facts):
            return {
                "answer": "No relevant information found in the knowledge graph.",
                "chain_of_thought": [],
                "evidence": [],
                "confidence": 0.0,
                "reasoning_path": "no_data"
            }
        
        # Step 2: Chain-of-thought reasoning
        reasoning_result = self.chain_of_thought_reasoning(question, graph_facts)
        
        # Step 3: Add graph facts to evidence
        reasoning_result["graph_facts"] = graph_facts
        
        return reasoning_result
    
    def close(self):
        """Close Neo4j connection."""
        self.driver.close()