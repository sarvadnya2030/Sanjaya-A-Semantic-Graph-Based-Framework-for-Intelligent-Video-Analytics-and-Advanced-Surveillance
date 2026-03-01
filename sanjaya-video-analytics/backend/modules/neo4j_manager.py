import subprocess
import time
import logging
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

log = logging.getLogger("neo4j.manager")

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "neo4j123"

def is_neo4j_running():
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))
        driver.verify_connectivity()
        driver.close()
        return True
    except ServiceUnavailable:
        return False

def start_neo4j():
    log.info("Starting Neo4j...")
    try:
        subprocess.run(["sudo", "systemctl", "start", "neo4j"], check=True, capture_output=True)
        log.info("Neo4j started via systemd")
    except:
        try:
            subprocess.Popen(["neo4j", "start"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            log.info("Neo4j started via neo4j command")
        except Exception as e:
            log.warning(f"Could not start Neo4j: {e}")
            return False
    
    for i in range(30):
        time.sleep(1)
        if is_neo4j_running():
            log.info("Neo4j is ready")
            return True
    log.warning("Neo4j did not become ready in 30s")
    return False

def ensure_neo4j():
    if is_neo4j_running():
        log.info("Neo4j already running")
        return True
    return start_neo4j()
