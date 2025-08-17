#!/usr/bin/env python3
"""
Complete FAISS reindexing script to include ALL sections in the vector search index.
This will rebuild the FAISS index with all 476 sections from the database.
"""

import sqlite3
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import os

# Configuration
DB_PATH = './data/documents.db'
EMBEDDINGS_DIR = Path('./data/embeddings')
FAISS_INDEX_PATH = EMBEDDINGS_DIR / 'faiss.index'

def main():
    print("🚀 Starting complete FAISS reindexing...")
    
    # Ensure embeddings directory exists
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load the embedding model
    print("⏳ Loading MiniLM model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Model loaded successfully")
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get all sections with text content
    print("📊 Fetching all sections from database...")
    cursor.execute("""
        SELECT id, section_title, section_text 
        FROM sections 
        WHERE section_text IS NOT NULL AND section_text != ''
        ORDER BY id
    """)
    sections = cursor.fetchall()
    
    print(f"Found {len(sections)} sections to index")
    
    if not sections:
        print("❌ No sections found to index!")
        conn.close()
        return
    
    # Prepare texts for embedding
    print("🔤 Preparing texts for embedding...")
    texts = []
    section_ids = []
    
    for section_id, title, text in sections:
        # Combine title and text for better semantic representation
        combined_text = f"{title or ''}\n{text or ''}".strip()
        if combined_text:  # Only add non-empty texts
            texts.append(combined_text)
            section_ids.append(section_id)
    
    print(f"Prepared {len(texts)} texts for embedding")
    
    # Generate embeddings in batches
    print("🧠 Generating embeddings...")
    batch_size = 32
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = model.encode(
            batch_texts,
            convert_to_tensor=False,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        all_embeddings.extend(batch_embeddings)
        print(f"  Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")
    
    embeddings_array = np.array(all_embeddings, dtype='float32')
    print(f"✅ Generated {embeddings_array.shape[0]} embeddings of dimension {embeddings_array.shape[1]}")
    
    # Create new FAISS index
    print("🔍 Building FAISS index...")
    dimension = embeddings_array.shape[1]
    
    # Use IndexIDMap to preserve original section IDs
    base_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    index = faiss.IndexIDMap(base_index)
    
    # Add vectors with their IDs
    section_ids_array = np.array(section_ids, dtype='int64')
    index.add_with_ids(embeddings_array, section_ids_array)
    
    print(f"✅ FAISS index built with {index.ntotal} vectors")
    
    # Save the index
    print("💾 Saving FAISS index...")
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    print(f"✅ FAISS index saved to {FAISS_INDEX_PATH}")
    
    # Verify the saved index
    print("🔍 Verifying saved index...")
    loaded_index = faiss.read_index(str(FAISS_INDEX_PATH))
    print(f"✅ Verified: Index contains {loaded_index.ntotal} vectors")
    
    # Show some statistics
    cursor.execute("""
        SELECT d.filename, COUNT(s.id) as section_count
        FROM documents d 
        JOIN sections s ON d.id = s.document_id 
        WHERE s.id IN ({})
        GROUP BY d.id, d.filename
        ORDER BY section_count DESC
    """.format(','.join(map(str, section_ids))))
    
    doc_stats = cursor.fetchall()
    print("\n📈 Indexed sections by document:")
    for filename, count in doc_stats:
        print(f"  {filename}: {count} sections")
    
    print(f"\n🎉 Complete reindexing finished!")
    print(f"   Total sections indexed: {len(section_ids)}")
    print(f"   FAISS index size: {loaded_index.ntotal} vectors")
    print(f"   Index file: {FAISS_INDEX_PATH}")
    
    conn.close()

if __name__ == "__main__":
    main()
