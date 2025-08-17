#!/usr/bin/env python3
"""
Check the backend's current FAISS index state
"""

import requests
import json

def check_backend_state():
    """Check what the backend thinks its current state is"""
    print("🔍 Checking backend vector search state...")
    
    # Try to get some info about the backend's current state
    try:
        # First check if backend is responding
        response = requests.get("http://127.0.0.1:8000/")
        if response.status_code == 200:
            data = response.json()
            print("✅ Backend is responding")
            features = data.get('features', {})
            print(f"Vector search enabled: {features.get('vector_search', False)}")
            print(f"LLM insights enabled: {features.get('llm_insights', False)}")
        else:
            print(f"❌ Backend health check failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Cannot connect to backend: {e}")

if __name__ == "__main__":
    check_backend_state()
