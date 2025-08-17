#!/usr/bin/env python3
"""
Test the actual backend API to see what it returns vs our debug script
"""

import requests
import json

def test_backend_search(query_text, max_results=15):
    """Test the actual backend API"""
    print(f"🌐 Testing backend API with query: '{query_text}'")
    
    url = "http://127.0.0.1:8000/search"
    payload = {
        "selected_text": query_text,
        "max_results": max_results
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            results = response.json()
            print(f"✅ Backend returned {len(results)} results")
            
            # Analyze results by document
            doc_counts = {}
            for result in results:
                doc_name = result.get('document_name', 'Unknown')
                doc_counts[doc_name] = doc_counts.get(doc_name, 0) + 1
            
            print("📊 Results by document:")
            for doc_name, count in doc_counts.items():
                print(f"  {doc_name}: {count} results")
            
            print("📋 Detailed results:")
            for i, result in enumerate(results):
                doc_name = result.get('document_name', 'Unknown')
                section_title = result.get('section_title', 'Untitled')
                score = result.get('similarity_score', 0)
                print(f"  {i+1}. {doc_name} - {section_title} (score: {score:.3f})")
                
            return results
        else:
            print(f"❌ Backend error: {response.status_code} - {response.text}")
            return []
            
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return []

if __name__ == "__main__":
    test_backend_search("french cuisine", 15)
