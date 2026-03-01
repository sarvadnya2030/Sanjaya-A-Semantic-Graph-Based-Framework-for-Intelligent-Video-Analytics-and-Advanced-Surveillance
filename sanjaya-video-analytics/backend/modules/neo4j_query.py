from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
import logging

log = logging.getLogger("neo4j.query")

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "neo4j123"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

def push_frame_analysis_to_neo4j(analyses):
    """Push frame analyses to Neo4j graph"""
    if not analyses:
        return
    
    try:
        with driver.session() as session:
            session.run("""
                MERGE (v:Video {name: $video_name})
                WITH v
                UNWIND $frames AS frame
                MERGE (f:Frame {video_name: $video_name, frame_id: frame.frame_id})
                SET f.timestamp = frame.timestamp,
                    f.description = frame.description
                MERGE (v)-[:HAS_FRAME]->(f)
                WITH f, frame
                UNWIND frame.entities AS entity
                MERGE (e:Entity {name: entity.name, type: entity.type})
                MERGE (f)-[:CONTAINS]->(e)
            """, 
            video_name=analyses[0].get("video_name", "unknown"),
            frames=analyses)
        log.info(f"Pushed {len(analyses)} frames to Neo4j")
    except ServiceUnavailable as e:
        log.warning(f"Neo4j unavailable: {e}")
    except Exception as e:
        log.error(f"Neo4j error: {e}")

def query_neo4j(cypher_query, params=None):
    """Execute a Cypher query"""
    try:
        with driver.session() as session:
            result = session.run(cypher_query, params or {})
            return [record.data() for record in result]
    except Exception as e:
        log.error(f"Query error: {e}")
        return []
