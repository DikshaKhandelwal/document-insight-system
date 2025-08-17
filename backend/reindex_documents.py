#!/usr/bin/env python3
"""
Reindex all existing documents in the FAISS vector search index.
This script processes all completed documents and adds their sections to the FAISS index.
"""

import os
import sys
import sqlite3
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

def reindex_all_documents():
    """Reindex all completed documents in the FAISS vector search index."""
    
    # Initialize paths
    DATA_DIR = Path("./data")
    DB_PATH = DATA_DIR / "documents.db"
    faiss_store = Path(os.getenv('FAISS_INDEX_PATH', './data/embeddings'))
    
    # Create embeddings directory if it doesn't exist
    faiss_store.mkdir(parents=True, exist_ok=True)
    index_file = faiss_store / 'faiss.index'
    
    print(f"🚀 Starting reindexing process...")
    print(f"📂 Database: {DB_PATH}")
    print(f"📂 FAISS index: {index_file}")
    
    # Check if database exists
    if not DB_PATH.exists():
        print(f"❌ Database not found at {DB_PATH}")
        return False
    
    # Initialize FAISS
    try:
        import faiss
        print("✅ FAISS library loaded")
    except ImportError:
        print("❌ FAISS library not available")
        return False
    
    # Load sentence transformer model
    try:
        print("🔧 Loading sentence transformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Sentence transformer model loaded")
    except Exception as e:
        print(f"❌ Failed to load sentence transformer model: {e}")
        return False
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get all completed documents
    cursor.execute("""
        SELECT COUNT(*) FROM documents WHERE processing_status = 'completed'
    """)
    doc_count = cursor.fetchone()[0]
    print(f"📊 Found {doc_count} completed documents")
    
    # Get all sections from completed documents
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
    
    # Initialize FAISS index
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
    
    # Save the index
    try:
        faiss.write_index(faiss_index, str(index_file))
        print(f"✅ Successfully saved FAISS index with {faiss_index.ntotal} vectors")
        print(f"📂 Index saved to: {index_file}")
    except Exception as e:
        print(f"❌ Failed to save FAISS index: {e}")
        conn.close()
        return False
    
    conn.close()
    
    print("🎉 Reindexing completed successfully!")
    print(f"📊 Total documents indexed: {doc_count}")
    print(f"📊 Total sections indexed: {total_indexed}")
    
    return True

if __name__ == "__main__":
    success = reindex_all_documents()
    if success:
        print("\n✨ All documents are now available for semantic search!")
    else:
        print("\n❌ Reindexing failed. Please check the errors above.")
        sys.exit(1)
