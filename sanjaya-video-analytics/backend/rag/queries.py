from neo4j import GraphDatabase
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

class GraphClient:
    def __init__(self):
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    def run(self, cypher, **params):
        with self.driver.session() as s:
            return [r.data() for r in s.run(cypher, **params)]

def q_persons_interacting_with_object(gc, video_name, object_class=None, min_conf=0.7):
    return gc.run("""
        MATCH (h:Human {video_name:$v})-[e:GRAPH_EDGE {relation:'INTERACTS_WITH'}]->(o:Object {video_name:$v})
        WHERE e.confidence >= $min_conf AND ($cls IS NULL OR coalesce(o.object_class, o.class) = $cls)
        OPTIONAL MATCH (h)-[:LOCATED_IN]->(z:Zone {video_name:$v})
        RETURN h.node_id AS person,
               coalesce(o.object_class, o.class) AS class,
               z.properties.zone_label AS zone,
               e.confidence AS confidence
        ORDER BY confidence DESC
    """, v=video_name, cls=object_class, min_conf=min_conf)

def q_gestures_toward(gc, video_name, min_conf=0.7):
    return gc.run("""
        MATCH (h:Human {video_name:$v})-[e:GRAPH_EDGE {relation:'GESTURES_TOWARD'}]->(o:Object {video_name:$v})
        WHERE e.confidence >= $min_conf
        RETURN h.node_id AS person, o.display AS object, e.confidence AS confidence
        ORDER BY confidence DESC
    """, v=video_name, min_conf=min_conf)

def q_activity_participants(gc, video_name, activity_label=None):
    return gc.run("""
        MATCH (h:Human {video_name:$v})-[e:GRAPH_EDGE {relation:'PERFORMS'}]->(a:Activity {video_name:$v})
        WHERE ($al IS NULL OR a.activity_label = $al OR a.properties.activity_type = $al)
        RETURN a.display AS activity, collect(h.node_id) AS participants, avg(e.confidence) AS confidence
    """, v=video_name, al=activity_label)

def q_objects(gc, video_name):
    return gc.run("""
        MATCH (o:Object {video_name:$v})
        RETURN coalesce(o.object_class, o.class) AS class, count(o) AS count
        ORDER BY count DESC
    """, v=video_name)