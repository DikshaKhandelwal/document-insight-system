#!/usr/bin/env python3
"""
Add missing recent documents to the FAISS index.
"""

import os
import sys
import sqlite3
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

def add_missing_sections():
    """Add sections that are in the database but missing from FAISS index."""
    
    # Initialize paths
    DATA_DIR = Path("./data")
    DB_PATH = DATA_DIR / "documents.db"
    faiss_store = Path(os.getenv('FAISS_INDEX_PATH', './data/embeddings'))
    index_file = faiss_store / 'faiss.index'
    
    print(f"🔍 Checking for missing sections...")
    print(f"📂 Database: {DB_PATH}")
    print(f"📂 FAISS index: {index_file}")
    
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
    
    # Load existing FAISS index
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
    
    # Get all indexable sections
    cursor.execute("""
        SELECT s.id, s.section_text, d.filename
        FROM sections s
        JOIN documents d ON s.document_id = d.id
        WHERE d.processing_status = 'completed' 
        AND s.section_text IS NOT NULL 
        AND LENGTH(TRIM(s.section_text)) > 20
        ORDER BY s.id
    """)
    
    all_sections = cursor.fetchall()
    print(f"📊 Found {len(all_sections)} indexable sections in database")
    
    # Check which section IDs are already in the FAISS index
    # We'll do this by trying to reconstruct which sections should be indexed
    existing_ids = set()
    
    # Get sections that should already be indexed (from our reindex script)
    cursor.execute("""
        SELECT s.id FROM sections s
        JOIN documents d ON s.document_id = d.id
        WHERE d.processing_status = 'completed' 
        AND s.section_text IS NOT NULL 
        AND LENGTH(TRIM(s.section_text)) > 20
        AND s.id <= (SELECT MAX(id) FROM sections WHERE document_id <= 17)
    """)
    
    old_section_ids = [row[0] for row in cursor.fetchall()]
    print(f"📊 Expected {len(old_section_ids)} sections from previous reindex")
    
    # Find missing sections (newer sections that aren't indexed)
    missing_sections = []
    for section_id, section_text, filename in all_sections:
        if section_id not in old_section_ids:
            missing_sections.append((section_id, section_text, filename))
    
    print(f"📊 Found {len(missing_sections)} missing sections to index")
    
    if not missing_sections:
        print("✅ No missing sections found")
        conn.close()
        return True
    
    # Show which files have missing sections
    missing_files = set(section[2] for section in missing_sections)
    print(f"📁 Files with missing sections: {missing_files}")
    
    # Add missing sections to FAISS index
    try:
        missing_ids = [section[0] for section in missing_sections]
        missing_texts = [section[1] for section in missing_sections]
        
        print(f"🔄 Computing embeddings for {len(missing_sections)} missing sections...")
        embeddings = model.encode(missing_texts, convert_to_numpy=True)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        ids_arr = np.array(missing_ids, dtype='int64')
        faiss_index.add_with_ids(embeddings.astype('float32'), ids_arr)
        
        print(f"✅ Added {len(missing_sections)} sections to FAISS index")
        print(f"📊 FAISS index now has {faiss_index.ntotal} vectors")
        
        # Save updated index
        faiss.write_index(faiss_index, str(index_file))
        print(f"✅ Saved updated FAISS index to {index_file}")
        
    except Exception as e:
        print(f"❌ Error adding missing sections: {e}")
        conn.close()
        return False
    
    conn.close()
    print("🎉 Successfully added missing sections!")
    return True

if __name__ == "__main__":
    success = add_missing_sections()
    if success:
        print("\n✨ All recent documents are now indexed!")
    else:
        print("\n❌ Failed to add missing sections.")
        sys.exit(1)
