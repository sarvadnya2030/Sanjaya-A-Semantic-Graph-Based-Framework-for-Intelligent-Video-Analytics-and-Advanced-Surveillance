#!/usr/bin/env python3
"""
Quick test script for Ollama VLM API
Tests qwen3-vl:2b-instruct-q4_K_M model
"""

import requests
import base64
import json
import sys

def test_ollama_vlm(image_path: str):
    """Test Ollama VLM with an image."""
    
    print(f"[TEST] Testing Ollama VLM with: {image_path}")
    
    # 1. CHECK OLLAMA HEALTH
    print("\n[1] Checking Ollama health...")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"✅ Ollama running with {len(models)} models")
            for model in models:
                print(f"   - {model.get('name')}")
        else:
            print(f"❌ Ollama error: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        return
    
    # 2. LOAD IMAGE
    print(f"\n[2] Loading image: {image_path}")
    try:
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        
        img_data = base64.b64encode(img_bytes).decode("utf-8")
        print(f"✅ Image loaded: {len(img_bytes)} bytes → {len(img_data)} chars (base64)")
    except Exception as e:
        print(f"❌ Failed to load image: {e}")
        return
    
    # 3. CALL VLM
    print("\n[3] Calling qwen3-vl:2b-instruct-q4_K_M...")
    
    prompt = "surveillance type json kg"
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen3-vl:2b-instruct-q4_K_M",
                "prompt": prompt,
                "images": [img_data],
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 500
                }
            },
            timeout=120
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            vlm_response = result.get("response", "")
            done = result.get("done", False)
            done_reason = result.get("done_reason", "")
            
            print(f"\n✅ VLM Response ({len(vlm_response)} chars):")
            print("=" * 80)
            print(vlm_response)
            print("=" * 80)
            print(f"\nDone: {done}, Reason: {done_reason}")
            
            # Show timing info
            if "total_duration" in result:
                total_ms = result["total_duration"] / 1_000_000
                print(f"\n⏱️  Total time: {total_ms:.0f}ms")
                
                if "prompt_eval_duration" in result:
                    prompt_ms = result["prompt_eval_duration"] / 1_000_000
                    print(f"   Prompt eval: {prompt_ms:.0f}ms")
                
                if "eval_duration" in result:
                    eval_ms = result["eval_duration"] / 1_000_000
                    print(f"   Generation: {eval_ms:.0f}ms")
            
            return vlm_response
        else:
            print(f"❌ HTTP {response.status_code}")
            print(response.text)
            return None
            
    except requests.Timeout:
        print("❌ Request timed out after 120s")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_vlm.py <image_path>")
        print("\nExample:")
        print("  python test_vlm.py /home/admin-/Desktop/testimg.png")
        print("  python test_vlm.py static/frames/frame_0.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    test_ollama_vlm(image_path)
