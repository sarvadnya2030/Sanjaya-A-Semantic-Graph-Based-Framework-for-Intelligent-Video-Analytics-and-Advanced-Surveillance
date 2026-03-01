#!/usr/bin/env python3
"""Rebuild RAG index from existing JSON files."""

import sys
import logging
from rag.json_rag import JsonRAG

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)

log = logging.getLogger("rebuild_rag")

if __name__ == "__main__":
    log.info("=" * 60)
    log.info("REBUILDING RAG INDEX")
    log.info("=" * 60)
    
    # Initialize RAG
    rag = JsonRAG(json_dirs=["json_outputs"])
    
    # Build index
    log.info("\n[1] Building FAISS index...")
    rag.build_index()
    
    if not rag.documents:
        log.error("❌ NO DOCUMENTS INDEXED!")
        sys.exit(1)
    
    log.info(f"\n[2] ✅ Index built: {len(rag.documents)} documents")
    
    # Test search
    log.info("\n[3] Testing search...")
    queries = [
        "What is happening in the store?",
        "Who are the people?",
        "What objects are visible?"
    ]
    
    for query in queries:
        log.info(f"\n   Query: '{query}'")
        results = rag.search(query, k=3)
        log.info(f"   Found {len(results)} results")
        
        if results:
            for i, (text, meta, score) in enumerate(results, 1):
                log.info(f"   [{i}] {meta['filename']} (score: {score:.2f})")
                log.info(f"       {text[:150]}...")
    
    log.info("\n" + "=" * 60)
    log.info("✅ RAG REBUILD COMPLETE")
    log.info("=" * 60)
