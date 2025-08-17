#!/usr/bin/env python3
"""
Debug script to test the exact same search logic as the backend
"""

import os
import sys
import sqlite3
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

def test_backend_search():
    """Test the exact same search logic as the backend API"""
    
    # Initialize paths
    DATA_DIR = Path("./data")
    DB_PATH = DATA_DIR / "documents.db"
    faiss_store = Path(os.getenv('FAISS_INDEX_PATH', './data/embeddings'))
    index_file = faiss_store / 'faiss.index'
    
    print(f"🔍 Testing backend search logic...")
    
    # Load FAISS and model (same as backend)
    try:
        import faiss
        print("✅ FAISS library loaded")
    except ImportError:
        print("❌ FAISS library not available")
        return False
    
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Sentence transformer model loaded")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # Load FAISS index
    if not index_file.exists():
        print("❌ FAISS index file not found")
        return False
    
    try:
        faiss_index = faiss.read_index(str(index_file))
        print(f"✅ Loaded FAISS index with {faiss_index.ntotal} vectors")
    except Exception as e:
        print(f"❌ Failed to load FAISS index: {e}")
        return False
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Test the exact same search as the API call
    selected_text = "restaurants and dining"
    max_results = 10
    
    print(f"🔍 Searching for: '{selected_text}'")
    
    # Encode query (same as backend)
    query_vec = model.encode([selected_text], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)
    
    k = min(max(max_results, 1), 50)
    distances, ids = faiss_index.search(query_vec.astype('float32'), k)
    
    print(f"📊 FAISS returned {len([i for i in ids[0] if i != -1])} results")
    
    results = []
    found_ids = [int(i) for i in ids[0] if i != -1]
    print(f"📊 Found section IDs: {found_ids}")
    
    if not found_ids:
        print("❌ No section IDs found")
        conn.close()
        return False
    
    # Fetch matching sections from DB by id (same as backend)
    placeholders = ','.join('?' for _ in found_ids)
    cursor.execute(f"SELECT id, section_title, section_text, page_number, document_id FROM sections WHERE id IN ({placeholders})", tuple(found_ids))
    rows = cursor.fetchall()
    
    print(f"📊 Database returned {len(rows)} rows for {len(found_ids)} section IDs")
    
    # Map id->row (same as backend)
    row_map = {r[0]: r for r in rows}
    print(f"📊 Row map keys: {list(row_map.keys())}")
    
    for idx, raw_id in enumerate(ids[0]):
        if raw_id == -1:
            continue
        sid = int(raw_id)
        row = row_map.get(sid)
        if not row:
            print(f"⚠️ No row found for section ID {sid}")
            continue
        
        # Compute similarity score from distances
        sim = float(distances[0][idx])
        
        # Get document filename (same as backend)
        cursor.execute("SELECT filename FROM documents WHERE id = ?", (row[4],))
        doc_row = cursor.fetchone()
        doc_name = doc_row[0] if doc_row else 'Unknown'
        
        full_text = row[2] or row[1] or ''
        snippet = full_text[:300] + '...' if len(full_text) > 300 else full_text
        
        result = {
            "document_name": doc_name,
            "section_title": row[1] or 'Untitled',
            "snippet": snippet,
            "page_number": row[3] or 1,
            "similarity_score": sim,
            "highlight_text": full_text.strip() if full_text else selected_text
        }
        
        results.append(result)
        print(f"📄 Result {len(results)}: {doc_name} - {row[1][:30]}... (score: {sim:.3f})")
    
    print(f"\n📊 Final results: {len(results)} from {len(set(r['document_name'] for r in results))} different documents")
    
    # Show document distribution
    doc_counts = {}
    for result in results:
        doc_name = result['document_name']
        doc_counts[doc_name] = doc_counts.get(doc_name, 0) + 1
    
    print("📊 Document distribution:")
    for doc_name, count in doc_counts.items():
        print(f"  {doc_name}: {count} results")
    
    conn.close()
    return True

if __name__ == "__main__":
    success = test_backend_search()
    if not success:
        print("\n❌ Test failed")
        sys.exit(1)
    else:
        print("\n✅ Test completed")
