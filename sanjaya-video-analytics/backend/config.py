import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
VIDEOS_FOLDER = os.path.join(BASE_DIR, "static", "videos")
FRAMES_FOLDER = os.path.join(BASE_DIR, "static", "frames")
JSON_FOLDER   = os.path.join(BASE_DIR, "json_outputs")

OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen3-vl:2b-instruct-q4_K_M")

NEO4J_URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j123")
