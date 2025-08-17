#!/usr/bin/env python3
"""
Debug script to test the exact search functionality that the frontend is calling.
This will help identify if there are any fallback issues.
"""

import sqlite3
import faiss
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
from datetime import datetime

# Mimic the backend's search logic exactly
DB_PATH = './data/documents.db'
EMBEDDINGS_DIR = Path('./data/embeddings')
FAISS_INDEX_PATH = EMBEDDINGS_DIR / 'faiss.index'

def load_models():
    """Load the same models as the backend"""
    print("⏳ Loading MiniLM model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Model loaded")
    
    print("⏳ Loading FAISS index...")
    faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
    print(f"✅ FAISS index loaded: {faiss_index.ntotal} vectors")
    
    return model, faiss_index

def test_search(query_text, max_results=15):
    """Test search with the exact same logic as the backend"""
    print(f"\n🔍 Testing search for: '{query_text}'")
    print(f"Requesting {max_results} results")
    
    model, faiss_index = load_models()
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    start_time = datetime.now()
    
    try:
        # Encode query (same as backend)
        query_vec = model.encode([query_text], convert_to_numpy=True)
        faiss.normalize_L2(query_vec)

        k = min(max(max_results or 8, 1), 50)
        print(f"Searching for k={k} results...")
        
        distances, ids = faiss_index.search(query_vec.astype('float32'), k)

        results = []
        found_ids = [int(i) for i in ids[0] if i != -1]
        print(f"FAISS returned {len(found_ids)} valid IDs: {found_ids}")
        
        if not found_ids:
            print("❌ No results found!")
            return []

        # Fetch matching sections from DB by id (same as backend)
        placeholders = ','.join('?' for _ in found_ids)
        cursor.execute(f"SELECT id, section_title, section_text, page_number, document_id FROM sections WHERE id IN ({placeholders})", tuple(found_ids))
        rows = cursor.fetchall()
        
        print(f"Database returned {len(rows)} matching sections")

        # Map id->row
        row_map = {r[0]: r for r in rows}

        for idx, raw_id in enumerate(ids[0]):
            if raw_id == -1:
                continue
            sid = int(raw_id)
            row = row_map.get(sid)
            if not row:
                print(f"⚠️ Section ID {sid} not found in database")
                continue
                
            # Compute similarity score from distances (inner product on normalized vectors)
            sim = float(distances[0][idx])

            # Get document filename (same as backend)
            cursor.execute("SELECT filename FROM documents WHERE id = ?", (row[4],))
            doc_row = cursor.fetchone()
            doc_name = doc_row[0] if doc_row else 'Unknown'

            full_text = row[2] or row[1] or ''
            snippet = full_text[:300] + '...' if len(full_text) > 300 else full_text

            results.append({
                'id': sid,
                'document_name': doc_name,
                'section_title': row[1] or 'Untitled',
                'snippet': snippet,
                'page_number': row[3] or 1,
                'similarity_score': sim,
                'document_id': row[4]
            })

        # Sort and return top-k (same as backend)
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        final = results[: (max_results or 8)]

        search_time = (datetime.now() - start_time).total_seconds()
        print(f"🔍 Search completed in {search_time:.3f}s -> {len(final)} results")
        
        # Analyze results by document
        doc_counts = {}
        for result in final:
            doc_name = result['document_name']
            doc_counts[doc_name] = doc_counts.get(doc_name, 0) + 1
        
        print("\n📊 Results by document:")
        for doc_name, count in doc_counts.items():
            print(f"  {doc_name}: {count} results")
        
        print("\n📋 Detailed results:")
        for i, result in enumerate(final):
            print(f"  {i+1}. {result['document_name']} - {result['section_title']} (score: {result['similarity_score']:.3f})")
        
        return final

    except Exception as e:
        print(f"❌ Search error: {e}")
        return []
    finally:
        conn.close()

def main():
    """Test with different queries"""
    test_queries = [
        "restaurants and dining",
        "traditional recipes", 
        "vacation activities",
        "cultural traditions"
    ]
    
    for query in test_queries:
        test_search(query, max_results=15)
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Test with custom query
        query = " ".join(sys.argv[1:])
        test_search(query, max_results=15)
    else:
        main()
