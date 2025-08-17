#!/usr/bin/env python3
"""
Clean rebuild of the FAISS index to remove duplicates.
"""

import os
import sys
import sqlite3
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

def rebuild_faiss_index():
    """Completely rebuild the FAISS index from scratch to eliminate duplicates."""
    
    # Initialize paths
    DATA_DIR = Path("./data")
    DB_PATH = DATA_DIR / "documents.db"
    faiss_store = Path(os.getenv('FAISS_INDEX_PATH', './data/embeddings'))
    
    # Create embeddings directory if it doesn't exist
    faiss_store.mkdir(parents=True, exist_ok=True)
    index_file = faiss_store / 'faiss.index'
    backup_file = faiss_store / 'faiss.index.backup'
    
    print(f"🔧 Rebuilding FAISS index from scratch...")
    print(f"📂 Database: {DB_PATH}")
    print(f"📂 FAISS index: {index_file}")
    
    # Backup existing index
    if index_file.exists():
        import shutil
        shutil.copy2(index_file, backup_file)
        print(f"📋 Backed up existing index to {backup_file}")
    
    # Load FAISS and model
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
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get all indexable sections (using the same criteria as before)
    cursor.execute("""
        SELECT s.id, s.section_text, d.filename
        FROM sections s
        JOIN documents d ON s.document_id = d.id
        WHERE d.processing_status = 'completed' 
        AND s.section_text IS NOT NULL 
        AND LENGTH(TRIM(s.section_text)) > 20
        ORDER BY s.id
    """)
    
    sections = cursor.fetchall()
    print(f"📊 Found {len(sections)} sections to index")
    
    if not sections:
        print("⚠️ No sections found to index")
        conn.close()
        return False
    
    # Check for duplicates in the database query results
    section_ids = [s[0] for s in sections]
    unique_ids = set(section_ids)
    if len(section_ids) != len(unique_ids):
        print(f"⚠️ Found duplicate section IDs in database query! {len(section_ids)} total, {len(unique_ids)} unique")
        # Remove duplicates
        seen_ids = set()
        unique_sections = []
        for section in sections:
            if section[0] not in seen_ids:
                unique_sections.append(section)
                seen_ids.add(section[0])
        sections = unique_sections
        print(f"🧹 Cleaned to {len(sections)} unique sections")
    
    # Initialize new FAISS index
    embedding_dim = 384  # MiniLM-L6-v2 dimension
    quantizer = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
    faiss_index = faiss.IndexIDMap(quantizer)
    
    # Process sections in batches
    batch_size = 100
    total_indexed = 0
    
    for i in range(0, len(sections), batch_size):
        batch = sections[i:i + batch_size]
        batch_ids = [section[0] for section in batch]
        batch_texts = [section[1] for section in batch]
        batch_files = [section[2] for section in batch]
        
        print(f"🔄 Processing batch {i//batch_size + 1}/{(len(sections) + batch_size - 1)//batch_size}")
        print(f"   Files in batch: {set(batch_files)}")
        
        try:
            # Generate embeddings
            embeddings = model.encode(batch_texts, convert_to_numpy=True)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add to index
            ids_arr = np.array(batch_ids, dtype='int64')
            faiss_index.add_with_ids(embeddings.astype('float32'), ids_arr)
            
            total_indexed += len(batch)
            print(f"   ✅ Indexed {len(batch)} sections (total: {total_indexed})")
            
        except Exception as e:
            print(f"   ❌ Error processing batch: {e}")
            continue
    
    # Save the new index
    try:
        faiss.write_index(faiss_index, str(index_file))
        print(f"✅ Successfully saved new FAISS index with {faiss_index.ntotal} vectors")
        print(f"📂 Index saved to: {index_file}")
    except Exception as e:
        print(f"❌ Failed to save FAISS index: {e}")
        conn.close()
        return False
    
    # Test the new index
    print(f"🧪 Testing new index...")
    query_vec = model.encode(['restaurants and food'], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)
    distances, ids = faiss_index.search(query_vec.astype('float32'), 10)
    
    found_ids = [int(i) for i in ids[0] if i != -1]
    unique_found = set(found_ids)
    print(f"🧪 Test search found {len(found_ids)} results, {len(unique_found)} unique section IDs")
    
    if len(found_ids) != len(unique_found):
        print("❌ Still have duplicates in search results!")
        conn.close()
        return False
    else:
        print("✅ No duplicates found in test search")
    
    conn.close()
    
    print("🎉 FAISS index rebuilt successfully!")
    print(f"📊 Total sections indexed: {total_indexed}")
    print(f"📊 Unique vectors in index: {faiss_index.ntotal}")
    
    return True

if __name__ == "__main__":
    success = rebuild_faiss_index()
    if success:
        print("\n✨ Clean FAISS index ready! Restart your backend server.")
    else:
        print("\n❌ Failed to rebuild FAISS index.")
        sys.exit(1)
