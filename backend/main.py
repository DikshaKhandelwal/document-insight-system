#!/usr/bin/env python3
"""
Document Insight System - Fastry:
    from google.cloud import texttospeech
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("⚠️ Google Cloud Text-to-Speech not available. Audio generation disabled.")ackend
Implements all features:
1. PDF Upload & Parsing (Bulk + Fresh)
2. Text Selection & Semantic Search
3. Insights Generation (Related/Overlapping/Contradicting/Examples)
4. Audio Overview / Podcast Mode
5. FAISS Vector Storage
6. SQLite Metadata Storage
"""

import os
import sys
import json
import sqlite3
import tempfile
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import uvicorn
try:
    import fitz
except Exception:
    fitz = None
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import aiofiles

# Add lib directory to path for our PDF extractors
sys.path.append(str(Path(__file__).parent.parent / "lib" / "pdf-extractors"))

try:
    from pdf_outline_extractor import SimplePDFExtractor
except ImportError as e:
    print(f"Error importing PDF extractors: {e}")
    sys.exit(1)

# Try to import optional dependencies - Use MiniLM without FAISS
MINILM_AVAILABLE = False
SENTENCE_TRANSFORMER_AVAILABLE = False
sentence_model = None

# Import cosine similarity regardless
try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ sklearn not available for cosine similarity")

try:
    from sentence_transformers import SentenceTransformer
    print("📦 sentence_transformers imported, will load model on first use")
    # Lazy loading - don't load model immediately to start server faster
    SENTENCE_TRANSFORMER_AVAILABLE = True
except Exception as e:
    MINILM_AVAILABLE = False
    SENTENCE_TRANSFORMER_AVAILABLE = False
    print(f"⚠️ sentence_transformers not available: {e}")
    print("📝 Using enhanced text-based search")

# Use enhanced text similarity 
from difflib import SequenceMatcher
import re

# Disable FAISS for now - using MiniLM with cosine similarity instead
VECTOR_AVAILABLE = False
model = None
faiss_index = None
print("📝 Using MiniLM-based semantic search (FAISS disabled for compatibility)")

# Function to load MiniLM model lazily
def load_minilm_model():
    """Load MiniLM model on first use to speed up server startup"""
    global sentence_model, MINILM_AVAILABLE
    if sentence_model is None and SENTENCE_TRANSFORMER_AVAILABLE:
        try:
            print("⏳ Loading MiniLM model (first time may take a few moments)...")
            sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            MINILM_AVAILABLE = True
            print("✅ MiniLM model loaded successfully for semantic search")
        except Exception as e:
            print(f"❌ Failed to load MiniLM model: {e}")
            MINILM_AVAILABLE = False
    return sentence_model


def extract_page_text_from_pdf(filename: str, page_number: Optional[int]):
    """Try to extract text from a specific page in an uploaded PDF file.
    Returns None on failure or when fitz is not available."""
    try:
        if fitz is None:
            return None
        if not filename:
            return None
        file_path = UPLOADS_DIR / filename
        if not file_path.exists():
            return None

        # page_number is expected to be 1-based in DB; convert to 0-based index
        try:
            idx = int(page_number) - 1 if page_number is not None else 0
        except Exception:
            idx = 0

        doc = fitz.open(str(file_path))
        try:
            if 0 <= idx < len(doc):
                text = doc[idx].get_text()
                return text.strip() if text else None
        finally:
            doc.close()
    except Exception as e:
        print('⚠️ Failed to extract page text for snippet:', e)
    return None


def build_snippet(preferred_text: Optional[str], fallback_title: Optional[str], filename: Optional[str] = None, page_number: Optional[int] = None, min_lines: int = 2, max_chars: int = 300) -> str:
    """Construct a 2-3 line snippet for search results.
    - preferred_text: section_text from DB (preferred)
    - fallback_title: section title (heading)
    - filename/page_number: used to extract page text if preferred_text is short
    ``Returns`` a short snippet (<= max_chars) preferably containing 2 lines.
    """
    text = (preferred_text or '').strip()

    # If section text is too short, try to pull page-level text from PDF
    page_text = None
    if (not text or len(text) < 60) and filename:
        page_text = extract_page_text_from_pdf(filename, page_number)
        # If we have page text and a fallback title, try to find the heading and extract the lines after it
        if page_text:
            try:
                ft = (fallback_title or '').strip()
                if ft:
                    # Search lines to locate the heading line and capture the following non-empty lines
                    all_lines = [ln for ln in page_text.splitlines()]
                    found_snippet_lines = []
                    for idx, ln in enumerate(all_lines):
                        if ft.lower() in ln.lower():
                            # Collect next few non-empty lines after the heading
                            for next_ln in all_lines[idx+1: idx+1+ (min_lines * 3)]:
                                if next_ln and next_ln.strip():
                                    found_snippet_lines.append(next_ln.strip())
                                if len(found_snippet_lines) >= min_lines:
                                    break
                            break

                    if found_snippet_lines:
                        # Use the lines after the heading as the snippet
                        text = ' '.join(found_snippet_lines[:min_lines])
                    else:
                        # fallback to the whole page text if no heading-specific snippet found
                        if len(page_text.strip()) > len(text):
                            text = page_text.strip()
            except Exception as e:
                print('⚠️ Error while extracting heading-following snippet from page text:', e)
                if page_text and len(page_text.strip()) > len(text):
                    text = page_text.strip()

    if not text:
        text = (fallback_title or '').strip()

    # Prefer returning 2-3 non-empty lines for context
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        if len(lines) >= min_lines + 1:
            snippet = ' '.join(lines[: min_lines + 1 ])
        else:
            snippet = ' '.join(lines[: min_lines]) if len(lines) >= min_lines else ' '.join(lines)
    else:
        snippet = text

    # Fallback to a character-limited snippet
    if not snippet:
        snippet = text[:max_chars]

    if len(snippet) > max_chars:
        snippet = snippet[:max_chars].rstrip() + '...'

    return snippet

try:
    import google.generativeai as genai
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("⚠️ Google Gemini not available. Insights generation disabled.")

try:
    import azure.cognitiveservices.speech as speechsdk
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("⚠️ Azure Speech Services not available. Audio generation disabled.")

# FastAPI app
app = FastAPI(
    title="Document Insight System",
    description="Advanced PDF analysis with semantic search and AI insights",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001", "http://127.0.0.1:3001", "http://localhost:3002", "http://127.0.0.1:3002"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global configurations
DATA_DIR = Path("data")
UPLOADS_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
AUDIO_DIR = DATA_DIR / "audio"
DB_PATH = DATA_DIR / "documents.db"

# Ensure directories exist
for directory in [DATA_DIR, UPLOADS_DIR, PROCESSED_DIR, AUDIO_DIR]:
    directory.mkdir(exist_ok=True)

# Global instances
pdf_extractor = SimplePDFExtractor()
model = None
faiss_index = None

# Database initialization
def init_database():
    """Initialize SQLite database for document metadata."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Documents table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE NOT NULL,
            title TEXT,
            upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            processing_status TEXT DEFAULT 'pending',
            total_sections INTEGER DEFAULT 0,
            file_size INTEGER,
            file_hash TEXT
        )
    """)
    
    # Sections table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER,
            section_title TEXT NOT NULL,
            section_text TEXT,
            page_number INTEGER,
            level TEXT,
            embedding_index INTEGER,
            FOREIGN KEY (document_id) REFERENCES documents (id)
        )
    """)
    
    # Search history table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS search_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_text TEXT NOT NULL,
            selected_text TEXT,
            results_count INTEGER,
            search_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()

def init_vector_search():
    """Initialize FAISS vector search and sentence transformer model."""
    global model, faiss_index, VECTOR_AVAILABLE

    # Try to import faiss and the sentence-transformers model
    try:
        import faiss
    except Exception as e:
        print(f"⚠️ faiss not available: {e}")
        VECTOR_AVAILABLE = False
        return False

    # Load or create embedding model (reuse existing lazy loader)
    emb_model = load_minilm_model()
    if not emb_model:
        print("⚠️ Embedding model not available; cannot enable FAISS vector search")
        VECTOR_AVAILABLE = False
        return False

    # Set global model to the sentence-transformers instance used for embeddings
    model = emb_model

    # Prepare index storage path
    faiss_store = Path(os.getenv('FAISS_INDEX_PATH', './data/embeddings'))
    faiss_store.mkdir(parents=True, exist_ok=True)
    index_file = faiss_store / 'faiss.index'

    # Known MiniLM dimension
    EMBEDDING_DIM = 384

    try:
        # Try to load existing index
        if index_file.exists():
            try:
                faiss_index = faiss.read_index(str(index_file))
                print(f"✅ Loaded FAISS index from {index_file}")
            except Exception as e:
                print(f"⚠️ Failed to read existing FAISS index: {e}. Creating new index.")
                # create fresh
                quant = faiss.IndexFlatIP(EMBEDDING_DIM)
                faiss_index = faiss.IndexIDMap(quant)
        else:
            quant = faiss.IndexFlatIP(EMBEDDING_DIM)
            faiss_index = faiss.IndexIDMap(quant)
            print(f"✅ Created new FAISS index (dim={EMBEDDING_DIM})")

        VECTOR_AVAILABLE = True
        return True
    except Exception as e:
        print(f"❌ Failed to initialize FAISS index: {e}")
        VECTOR_AVAILABLE = False
        faiss_index = None
        return False

def init_llm():
    """Initialize Gemini LLM for insights generation."""
    try:
        # Configure Gemini (you'll need to set GEMINI_API_KEY environment variable)
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            print("✅ Gemini LLM initialized successfully")
            return True
        else:
            print("⚠️ GEMINI_API_KEY not set. Insights generation will be limited.")
            return False
    except Exception as e:
        print(f"❌ Failed to initialize Gemini: {e}")
        return False

# Pydantic models
class DocumentMetadata(BaseModel):
    id: int
    filename: str
    title: str
    upload_time: str
    processing_status: str
    total_sections: int
    file_size: int

class SectionData(BaseModel):
    id: int
    document_id: int
    section_title: str
    section_text: str
    page_number: int
    level: str

class SearchRequest(BaseModel):
    selected_text: str
    context: Optional[str] = None
    max_results: Optional[int] = 8
    similarity_threshold: Optional[float] = 0.25  # Lower threshold for more results
    include_metadata: Optional[bool] = True

class SearchResult(BaseModel):
    document_name: str
    section_title: str
    snippet: str
    page_number: int
    similarity_score: float
    highlight_text: str

class InsightsRequest(BaseModel):
    selected_text: str
    related_sections: List[Dict[str, Any]]
    insight_types: List[str] = ["related", "overlapping", "contradicting", "examples", "extensions", "problems", "applications", "methodology"]
    analysis_depth: Optional[str] = "standard"
    focus_areas: Optional[List[str]] = None

class AudioRequest(BaseModel):
    selected_text: str
    related_sections: List[Dict[str, Any]]
    insights: Optional[Dict[str, Any]] = None
    script_style: str = "engaging_podcast"
    audio_format: Optional[Dict[str, Any]] = None
    content_focus: Optional[List[str]] = None

# Initialize everything
print("🚀 Initializing Document Insight System Backend...")
init_database()
vector_enabled = init_vector_search()
llm_enabled = init_llm()

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "Document Insight System API",
        "version": "1.0.0",
        "features": {
            "vector_search": vector_enabled,
            "llm_insights": llm_enabled,
            "audio_generation": TTS_AVAILABLE
        }
    }

@app.post("/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload and process PDF documents (bulk upload supported).
    """
    uploaded_files = []
    
    for file in files:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail=f"Only PDF files allowed. Got: {file.filename}")
        
        # Save uploaded file
        file_path = UPLOADS_DIR / file.filename
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Store in database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO documents (filename, file_size, processing_status)
                VALUES (?, ?, 'pending')
            """, (file.filename, len(content)))
            
            document_id = cursor.lastrowid
            conn.commit()
            
            uploaded_files.append({
                "id": document_id,
                "filename": file.filename,
                "size": len(content),
                "status": "uploaded"
            })
            
            # Schedule background processing
            if background_tasks:
                background_tasks.add_task(process_document, file_path, document_id)
            
        except sqlite3.IntegrityError:
            # File already exists, get existing ID
            cursor.execute("SELECT id FROM documents WHERE filename = ?", (file.filename,))
            document_id = cursor.fetchone()[0]
            uploaded_files.append({
                "id": document_id,
                "filename": file.filename,
                "size": len(content),
                "status": "already_exists"
            })
        
        finally:
            conn.close()
    
    return {
        "message": f"Successfully uploaded {len(uploaded_files)} files",
        "files": uploaded_files
    }

async def process_document(file_path: Path, document_id: int):
    """
    Background task to process uploaded PDF and extract sections.
    """
    try:
        # Update status to processing
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE documents SET processing_status = 'processing' WHERE id = ?",
            (document_id,)
        )
        conn.commit()
        conn.close()
        
        # Extract outline using our PDF extractor
        print(f"📖 Processing PDF: {file_path}")
        result = pdf_extractor.extract_outline(str(file_path))
        
        title = result.get('title', file_path.stem)
        outline = result.get('outline', [])
        
        # Store sections in database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Update document with title and section count
        cursor.execute("""
            UPDATE documents 
            SET title = ?, total_sections = ?, processing_status = 'completed'
            WHERE id = ?
        """, (title, len(outline), document_id))
        
        # Store sections
        embeddings_to_add = []
        section_id_list = []
        for section in outline:
            # The extractor returns 0-based page indices (page numbers starting at 0).
            # Convert to 1-based page numbers for UI/Viewer compatibility.
            raw_page = section.get('page', 0) or 0
            try:
                page_number = int(raw_page) + 1 if int(raw_page) >= 0 else 0
            except Exception:
                page_number = 0
            # Choose section text: prefer explicit text, but if too short, extract full page text
            sec_text = section.get('text', '') or ''
            if len(sec_text.strip()) < 40:
                # Try to extract page text using PyMuPDF if available
                try:
                    if fitz is not None:
                        doc = fitz.open(str(file_path))
                        # raw_page is 0-based; ensure within range
                        if 0 <= int(raw_page) < len(doc):
                            page_text = doc[int(raw_page)].get_text()
                            # Use the page text if it's longer and meaningful
                            if page_text and len(page_text.strip()) > len(sec_text):
                                sec_text = page_text.strip()
                        doc.close()
                except Exception as e:
                    print('⚠️ Failed to extract page text for fuller section content:', e)

            cursor.execute("""
                INSERT INTO sections (document_id, section_title, section_text, page_number, level)
                VALUES (?, ?, ?, ?, ?)
            """, (
                document_id,
                section.get('text', ''),
                sec_text,
                page_number,
                section.get('level', 'H3')
            ))
            section_id = cursor.lastrowid
            section_id_list.append(section_id)

            # Prepare for vector embedding
            if model and sec_text and len(sec_text.strip()) > 20:
                embeddings_to_add.append((section_id, sec_text))
        
        conn.commit()
        conn.close()
        
        # Add to FAISS index if vector search is enabled
        if VECTOR_AVAILABLE and model and faiss_index and embeddings_to_add:
            try:
                import faiss
                print(f"🔍 Computing embeddings for {len(embeddings_to_add)} sections...")
                ids = [sid for sid, txt in embeddings_to_add]
                texts = [txt for sid, txt in embeddings_to_add]
                # Encode to numpy
                embeddings = model.encode(texts, convert_to_numpy=True)
                # Normalize and add
                faiss.normalize_L2(embeddings)
                ids_arr = np.array(ids, dtype='int64')
                faiss_index.add_with_ids(embeddings.astype('float32'), ids_arr)

                # Persist index to disk
                try:
                    faiss_store = Path(os.getenv('FAISS_INDEX_PATH', './data/embeddings'))
                    index_file = faiss_store / 'faiss.index'
                    faiss.write_index(faiss_index, str(index_file))
                    print(f"✅ Persisted FAISS index to {index_file}")
                except Exception as e:
                    print('⚠️ Could not persist FAISS index:', e)

                print(f"✅ Added {len(embeddings_to_add)} embeddings to FAISS index")
            except Exception as e:
                print('❌ Error while indexing embeddings to FAISS:', e)
        else:
            print("📝 Vector indexing skipped - using text search")
        
        print(f"✅ Successfully processed {file_path}")
        
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        
        # Update status to failed
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE documents SET processing_status = 'failed' WHERE id = ?",
            (document_id,)
        )
        conn.commit()
        conn.close()

@app.get("/documents", response_model=List[DocumentMetadata])
async def list_documents():
    """Get list of all uploaded documents."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, filename, title, upload_time, processing_status, total_sections, file_size
        FROM documents
        ORDER BY upload_time DESC
    """)
    
    documents = []
    for row in cursor.fetchall():
        documents.append(DocumentMetadata(
            id=row[0],
            filename=row[1],
            title=row[2] or row[1],
            upload_time=row[3],
            processing_status=row[4],
            total_sections=row[5],
            file_size=row[6] or 0
        ))
    
    conn.close()
    return documents

@app.get("/documents/{document_id}/sections")
async def get_document_sections(document_id: int):
    """Get all sections for a specific document."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, section_title, section_text, page_number, level
        FROM sections
        WHERE document_id = ?
        ORDER BY page_number, id
    """, (document_id,))
    
    sections = []
    for row in cursor.fetchall():
        sections.append({
            "id": row[0],
            "title": row[1],
            "text": row[2],
            "page": row[3],
            "level": row[4]
        })
    
    conn.close()
    return {"document_id": document_id, "sections": sections}

@app.post("/search", response_model=List[SearchResult])
async def search_related(request: SearchRequest):
    """
    High-speed semantic search for related sections based on selected text.
    Core feature for "connecting the dots" - optimized for speed and relevance.
    """
    # Store search in history with enhanced metadata
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO search_history (query_text, selected_text)
        VALUES (?, ?)
    """, (request.selected_text, request.selected_text))
    search_id = cursor.lastrowid
    conn.commit()
    
    try:
        start_time = datetime.now()
        
        # If FAISS index exists and contains vectors, prefer vector search.
        # If FAISS index is present but empty, fall back to text search to avoid empty results.
        try:
            faiss_ntotal = faiss_index.ntotal if faiss_index is not None else 0
        except Exception:
            faiss_ntotal = 0

        if model and faiss_index and VECTOR_AVAILABLE and faiss_ntotal > 0:
            # Use vector search if available and populated
            results = await vector_search(request, cursor, search_id, start_time)
        else:
            # Use fallback text search
            results = await fallback_text_search(request, cursor, search_id, start_time)
        
        return results
        
    except Exception as e:
        print(f"❌ Search error: {e}")
        # Graceful fallback - return empty results rather than failing
        return []
    
    finally:
        conn.close()

async def vector_search(request: SearchRequest, cursor, search_id: int, start_time):
    """Vector-based semantic search using FAISS"""
    try:
        import faiss
    except Exception:
        print("⚠️ FAISS not available at runtime, falling back to text search")
        return await fallback_text_search(request, cursor, search_id, start_time)

    if not (model and faiss_index and VECTOR_AVAILABLE):
        print("⚠️ Vector search dependencies missing, falling back to text search")
        return await fallback_text_search(request, cursor, search_id, start_time)

    try:
        # Encode query
        query_vec = model.encode([request.selected_text], convert_to_numpy=True)
        # Normalize for cosine similarity via inner product
        faiss.normalize_L2(query_vec)

        k = min(max(request.max_results or 8, 1), 50)
        distances, ids = faiss_index.search(query_vec.astype('float32'), k)

        results = []
        found_ids = [int(i) for i in ids[0] if i != -1]
        if not found_ids:
            return []

        # Fetch matching sections from DB by id
        placeholders = ','.join('?' for _ in found_ids)
        cursor.execute(f"SELECT id, section_title, section_text, page_number, document_id FROM sections WHERE id IN ({placeholders})", tuple(found_ids))
        rows = cursor.fetchall()

        # Map id->row
        row_map = {r[0]: r for r in rows}

        for idx, raw_id in enumerate(ids[0]):
            if raw_id == -1:
                continue
            sid = int(raw_id)
            row = row_map.get(sid)
            if not row:
                continue
            # Compute similarity score from distances (inner product on normalized vectors)
            sim = float(distances[0][idx])

            # Get document filename
            cursor.execute("SELECT filename FROM documents WHERE id = ?", (row[4],))
            doc_row = cursor.fetchone()
            doc_name = doc_row[0] if doc_row else 'Unknown'

            # Build a multi-line snippet (prefer section text, fall back to page text)
            full_text = row[2] or ''
            snippet = build_snippet(full_text, row[1], filename=row[4], page_number=row[3])

            # Use the actual section content for highlighting instead of selected text
            highlight_candidate = (full_text.strip() if full_text else row[1] or request.selected_text)
            
            results.append(SearchResult(
                document_name=doc_name,
                section_title=row[1] or 'Untitled',
                snippet=snippet,
                page_number=row[3] or 1,
                similarity_score=sim,
                highlight_text=highlight_candidate
            ))

        # Sort and return top-k
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        final = results[: (request.max_results or 8) ]

        search_time = (datetime.now() - start_time).total_seconds()
        cursor.execute("UPDATE search_history SET results_count = ?, search_time = CURRENT_TIMESTAMP WHERE id = ?", (len(final), search_id))
        print(f"🔍 FAISS vector search completed in {search_time:.3f}s -> {len(final)} results")
        return final

    except Exception as e:
        print(f"❌ Vector search failed: {e}")
        return await fallback_text_search(request, cursor, search_id, start_time)

async def fallback_text_search(request: SearchRequest, cursor, search_id: int, start_time):
    """Enhanced text search using MiniLM semantic embeddings if available, otherwise keyword matching"""
    print(f"🔍 Using {'MiniLM semantic' if MINILM_AVAILABLE else 'keyword'} search for: '{request.selected_text[:50]}...'")
    
    # Get all sections from database
    cursor.execute("""
        SELECT s.id, s.section_title, s.section_text, s.page_number, d.filename, d.title
        FROM sections s
        JOIN documents d ON s.document_id = d.id
        WHERE d.processing_status = 'completed'
        ORDER BY s.id
    """)
    
    all_sections = cursor.fetchall()
    
    if SENTENCE_TRANSFORMER_AVAILABLE:
        # Try to use MiniLM for semantic similarity (will load on demand)
        results = await minilm_semantic_search(request, all_sections)
    else:
        # Fallback to basic keyword matching
        results = await basic_keyword_search(request, all_sections)
    
    # Sort by similarity and limit results
    results.sort(key=lambda x: x.similarity_score, reverse=True)
    final_results = results[:request.max_results]
    
    # Calculate search performance metrics
    search_time = (datetime.now() - start_time).total_seconds()
    
    # Update search history
    cursor.execute("""
        UPDATE search_history 
        SET results_count = ?, search_time = CURRENT_TIMESTAMP 
        WHERE id = ?
    """, (len(final_results), search_id))
    
    print(f"🔍 Search completed in {search_time:.3f}s: '{request.selected_text[:50]}...' -> {len(final_results)} results")
    
    return final_results

async def minilm_semantic_search(request: SearchRequest, all_sections):
    """Semantic search using MiniLM embeddings and cosine similarity"""
    results = []
    
    # Load model lazily on first use
    model = load_minilm_model()
    if not model:
        print("⚠️ MiniLM model not available, falling back to keyword search")
        return await basic_keyword_search(request, all_sections)
    
    try:
        # Generate embedding for the query text
        query_embedding = model.encode([request.selected_text])
        
        # Process sections in batches for efficiency
        section_texts = []
        valid_sections = []
        
        for section in all_sections:
            section_text = section[2] or section[1] or ""
            if len(section_text.strip()) > 10:  # Only process sections with meaningful content
                section_texts.append(section_text)
                valid_sections.append(section)
        
        if not section_texts:
            return results
        
        # Generate embeddings for all section texts
        section_embeddings = model.encode(section_texts)
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, section_embeddings)[0]
        
        # Create results with similarity scores
        for i, (section, similarity) in enumerate(zip(valid_sections, similarities)):
            # Set a reasonable threshold for semantic similarity
            if similarity > 0.15:  # MiniLM threshold
                full_text = section[2] or ''
                snippet = build_snippet(full_text, section[1], filename=section[4], page_number=section[3])

                # Use actual section content for better highlighting
                highlight_candidate = (full_text.strip() if full_text else section[1] or request.selected_text)
                
                results.append(SearchResult(
                    document_name=section[4],
                    section_title=section[1] or "Untitled Section", 
                    snippet=snippet,
                    page_number=section[3] or 1,
                    similarity_score=float(similarity),
                    highlight_text=highlight_candidate
                ))
        
        print(f"✅ MiniLM semantic search found {len(results)} relevant sections")
        
    except Exception as e:
        print(f"❌ MiniLM search failed: {e}")
        # Fallback to basic keyword search
        results = await basic_keyword_search(request, all_sections)
    
    return results

async def basic_keyword_search(request: SearchRequest, all_sections):
    """Basic keyword-based search as fallback"""
    results = []
    query_words = set(request.selected_text.lower().split())
    
    for section in all_sections:
        section_text = (section[2] or section[1] or "").lower()
        section_words = set(section_text.split())
        
        # Calculate simple similarity based on word overlap
        if len(query_words) == 0:
            continue
            
        common_words = query_words.intersection(section_words)
        similarity = len(common_words) / len(query_words)
        
        # Additional scoring for exact phrase matches
        if request.selected_text.lower() in section_text:
            similarity += 0.5
        
        # Check for partial phrase matches
        query_phrases = request.selected_text.lower().split('. ')
        for phrase in query_phrases:
            if len(phrase.strip()) > 10 and phrase.strip() in section_text:
                similarity += 0.2
        
        # Only include results with reasonable similarity
        if similarity > 0.1:  # Lower threshold for basic text search
            full_text = section[2] or ''
            snippet = build_snippet(full_text, section[1], filename=section[4], page_number=section[3])

            # Use actual section content for better highlighting
            highlight_candidate = (full_text.strip() if full_text else section[1] or request.selected_text)
            
            results.append(SearchResult(
                document_name=section[4],
                section_title=section[1] or "Untitled Section", 
                snippet=snippet,
                page_number=section[3] or 1,
                similarity_score=min(similarity, 1.0),  # Cap at 1.0
                highlight_text=highlight_candidate
            ))
    
    print(f"📝 Basic keyword search found {len(results)} relevant sections")
    return results
    
    return final_results

@app.post("/insights")
async def generate_insights(request: InsightsRequest):
    """
    Generate comprehensive AI insights: Related, Overlapping, Contradicting, Examples, Extensions, Problems, Applications, Methodology.
    Uses Gemini 2.5 Flash for fast, accurate analysis grounded in user's document library.
    """
    if not LLM_AVAILABLE:
        return {
            "insights": {
                "related": f"Found {len(request.related_sections)} related sections in your document library.",
                "status": "LLM not available. Using rule-based insights."
            },
            "message": "LLM not available. Using rule-based insights."
        }
    
    try:
        # Prepare context from related sections
        context_sections = []
        for section in request.related_sections:
            context_sections.append(f"Document: {section.get('document_name', 'Unknown')}")
            context_sections.append(f"Section: {section.get('section_title', '')}")
            context_sections.append(f"Content: {section.get('snippet', '')}")
            context_sections.append(f"Similarity: {section.get('similarity_score', 0):.2f}")
            context_sections.append("---")
        
        context_text = "\n".join(context_sections)
        
        insights = {}
        
        # Enhanced prompt system for comprehensive analysis
        base_context = f"""
        You are analyzing text from a user's personal document library. Provide insights that are:
        - Grounded ONLY in the provided documents (not general knowledge)
        - Specific and actionable
        - Research-oriented and academic in tone
        - Focused on helping users connect insights across their documents
        
        Selected Text: "{request.selected_text}"
        
        Related Sections from User's Documents:
        {context_text}
        
        Analysis Depth: {request.analysis_depth}
        Focus Areas: {', '.join(request.focus_areas or [])}
        """
        
        # Generate different types of insights based on user specifications
        for insight_type in request.insight_types:
            if insight_type == "related":
                prompt = f"""{base_context}
                
                TASK: Identify RELATED content and similar methods/concepts.
                Provide 2-3 bullet points about:
                • Similar methods, techniques, or concepts found in the user's documents
                • Supporting evidence or corroborating findings
                • Connections between the selected text and related research
                
                Focus on direct connections and methodological similarities."""
                
            elif insight_type == "overlapping":
                prompt = f"""{base_context}
                
                TASK: Identify OVERLAPPING information and shared concepts.
                Provide 2-3 bullet points about:
                • Common themes or shared terminology across documents
                • Overlapping research areas or methodologies
                • Consistent findings or repeated concepts
                
                Focus on what's consistent across the user's document collection."""
                
            elif insight_type == "contradicting":
                prompt = f"""{base_context}
                
                TASK: Identify CONTRADICTORY findings and opposing viewpoints.
                Provide 2-3 bullet points about:
                • Conflicting results or conclusions in the user's documents
                • Different interpretations of similar data
                • Opposing methodological approaches
                
                If no contradictions exist, state "No contradictions found in your document library."
                Focus only on genuine conflicts, not just different approaches."""
                
            elif insight_type == "examples":
                prompt = f"""{base_context}
                
                TASK: Identify specific EXAMPLES and case studies.
                Provide 2-3 bullet points about:
                • Concrete examples or case studies mentioned in the documents
                • Practical applications described in the research
                • Real-world implementations or experiments
                
                Focus on tangible, specific examples rather than theoretical concepts."""
                
            elif insight_type == "extensions":
                prompt = f"""{base_context}
                
                TASK: Identify how other research has EXTENDED or built upon similar concepts.
                Provide 2-3 bullet points about:
                • How other papers in the collection have advanced the technique/concept
                • Improvements or modifications described in related documents
                • Follow-up research or subsequent developments
                
                Focus on progression and advancement of ideas."""
                
            elif insight_type == "problems":
                prompt = f"""{base_context}
                
                TASK: Identify PROBLEMS and limitations discussed in the documents.
                Provide 2-3 bullet points about:
                • Limitations or shortcomings mentioned in the research
                • Technical problems or implementation challenges
                • Criticisms or concerns raised about the approach
                
                Focus on documented issues and challenges."""
                
            elif insight_type == "applications":
                prompt = f"""{base_context}
                
                TASK: Identify REAL-WORLD APPLICATIONS and practical uses.
                Provide 2-3 bullet points about:
                • Industry applications or commercial uses mentioned
                • Practical implementations in real-world scenarios
                • Use cases or deployment examples
                
                Focus on practical, applied aspects rather than theoretical potential."""
                
            elif insight_type == "methodology":
                prompt = f"""{base_context}
                
                TASK: Identify different METHODOLOGICAL APPROACHES.
                Provide 2-3 bullet points about:
                • Alternative research methods or experimental designs
                • Different analytical approaches to similar problems
                • Variations in data collection or analysis techniques
                
                Focus on methodological diversity and approaches."""
            
            try:
                # Use Gemini for analysis with enhanced prompting
                model_gemini = genai.GenerativeModel('gemini-1.5-flash')
                response = model_gemini.generate_content(prompt)
                insights[insight_type] = response.text.strip()
                
            except Exception as e:
                insights[insight_type] = f"Error generating {insight_type} insights: {str(e)}"
        
        # Add metadata about the analysis
        analysis_meta = {
            "selected_text_length": len(request.selected_text),
            "analyzed_sections": len(request.related_sections),
            "analysis_timestamp": datetime.now().isoformat(),
            "insight_types_generated": len([k for k, v in insights.items() if v and not v.startswith("Error")])
        }
        
        return {
            "insights": insights,
            "selected_text": request.selected_text,
            "analyzed_sections": len(request.related_sections),
            "metadata": analysis_meta,
            "status": "comprehensive_analysis_complete"
        }
        
    except Exception as e:
        return {
            "insights": {
                "related": f"Found {len(request.related_sections)} related sections in your document library for: \"{request.selected_text[:100]}...\"",
                "status": f"Analysis error: {str(e)}. Showing basic results."
            },
            "error": f"Insights generation failed: {str(e)}",
            "fallback": True
        }

@app.post("/audio")
async def generate_audio(request: AudioRequest):
    """
    Generate engaging podcast-style audio overview.
    Two speakers discuss the selected content and insights in a natural, dynamic way.
    Content is grounded in the user's document library for trust and accuracy.
    """
    if not TTS_AVAILABLE:
        return {
            "script": "Audio generation not available - TTS services not configured.",
            "audio_files": [],
            "message": "Text-to-speech not available"
        }
    
    try:
        # Enhanced script generation using comprehensive insights
        if LLM_AVAILABLE:
            # Prepare comprehensive context
            insights_context = ""
            if request.insights:
                for insight_type, content in request.insights.items():
                    if content and not content.startswith("Error") and insight_type != "status":
                        insights_context += f"\n{insight_type.title()} Insights: {content}\n"
            
            related_content = []
            for section in request.related_sections:
                related_content.append({
                    "document": section.get('document_name', 'Unknown'),
                    "section": section.get('section_title', ''),
                    "content": section.get('snippet', ''),
                    "relevance": section.get('similarity_score', 0)
                })
            
            # Enhanced script prompt for engaging podcast content
            script_prompt = f"""
            Create an engaging, natural podcast conversation between two AI hosts discussing academic/research content.
            Make it sound like a dynamic conversation between knowledgeable researchers, not robotic.
            
            CONTENT TO DISCUSS:
            Selected Text: "{request.selected_text}"
            
            AI-Generated Insights:
            {insights_context}
            
            Related Documents from User's Library:
            {json.dumps(related_content, indent=2)}
            
            REQUIREMENTS:
            - 3-5 minutes of natural conversation when spoken
            - Two distinct voices: Host A (curious, asks questions) and Expert B (knowledgeable, provides insights)
            - Include specific references to the user's documents
            - Highlight key findings, contrasts, and connections
            - Make it engaging and easy to follow
            - Include natural conversational elements (transitions, emphasis, pauses)
            - Ground everything in the provided documents - no external knowledge
            
            STRUCTURE:
            1. Host A introduces the topic based on selected text
            2. Expert B explains the key concept
            3. Discussion of related findings from user's documents
            4. Exploration of different perspectives or contradictions
            5. Practical implications and examples
            6. Wrap-up with key takeaways
            
            FORMAT: "Speaker A: [text]" and "Speaker B: [text]"
            
            Make it sound like two real people having an intelligent, enthusiastic discussion about the research!
            """
            
            model_gemini = genai.GenerativeModel('gemini-1.5-flash')
            response = model_gemini.generate_content(script_prompt)
            script = response.text.strip()
        else:
            # Enhanced fallback script
            script = f"""
            Speaker A: Welcome to your personalized research insight podcast! Today we're diving deep into a fascinating topic from your document library.

            Speaker B: That's right! We're exploring "{request.selected_text[:100]}..." and I have to say, the connections we found across your documents are really intriguing.

            Speaker A: Tell us more about what you discovered.

            Speaker B: Well, we analyzed {len(request.related_sections)} related sections from your personal research collection, and there are some compelling patterns emerging.

            Speaker A: What kind of patterns are we talking about?

            Speaker B: {f"Looking at your insights, we see {', '.join([k for k in request.insights.keys() if k != 'status'])} themes emerging." if request.insights else "The semantic analysis reveals interesting thematic connections across your documents."}

            Speaker A: That's fascinating. How does this connect to the broader research landscape you've been building?

            Speaker B: What's particularly interesting is that this isn't just theoretical - your document collection shows practical applications and real-world implementations that bridge different research areas.

            Speaker A: For our listeners who want to explore this further, where should they look next in their document collection?

            Speaker B: I'd recommend diving deeper into the related sections we identified - they offer complementary perspectives that could spark new research directions or validate current approaches.

            Speaker A: Excellent insights! Thanks for this deep dive into your personalized research connections.
            """
        
        # Parse script and generate audio with different voices
        audio_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Combine the script into a single SSML payload alternating two distinct voices
        # to produce a 2-3 minute podcast. This avoids generating many small files and
        # keeps Azure usage low for free accounts.
        speech_key = os.getenv("AZURE_SPEECH_KEY")
        speech_region = os.getenv("AZURE_SPEECH_REGION")

        if not (speech_key and speech_region):
            print("Azure Speech credentials not found. Audio files not generated.")
        else:
            try:
                # Build alternating SSML using two voices
                # Normalize script into speaker segments
                segments = []
                for raw in script.split('\n'):
                    if not raw.strip():
                        continue
                    if raw.startswith('Speaker A:') or raw.startswith('Host A:'):
                        segments.append(('A', raw.split(':', 1)[1].strip()))
                    elif raw.startswith('Speaker B:') or raw.startswith('Expert B:'):
                        segments.append(('B', raw.split(':', 1)[1].strip()))
                    else:
                        # If not labeled, append to last speaker if present, else A
                        if segments:
                            segments[-1] = (segments[-1][0], segments[-1][1] + ' ' + raw.strip())
                        else:
                            segments.append(('A', raw.strip()))

                # Merge successive same-speaker segments to reduce switches
                merged = []
                for s, t in segments:
                    if merged and merged[-1][0] == s:
                        merged[-1] = (s, merged[-1][1] + ' ' + t)
                    else:
                        merged.append((s, t))

                # Estimate total duration and truncate if needed (target 150s default)
                full_text = ' '.join([t for _, t in merged])
                est_secs = max(1, len(full_text) / 12)
                target_secs = 150  # ~2.5 minutes
                if est_secs > target_secs:
                    # truncate merged segments to fit target length
                    allowed_chars = int(target_secs * 12)
                    truncated = full_text[:allowed_chars]
                    # Rebuild segments from truncated text as a single block
                    merged = [('A', truncated)]

                # Build SSML with alternating voices
                ssml_parts = ['<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">']
                for speaker, text in merged:
                    voice = 'en-US-JennyNeural' if speaker == 'A' else 'en-US-RyanNeural'
                    # Escape any problematic characters
                    safe_text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    ssml_parts.append(f'<voice name="{voice}"><prosody rate="0.95" pitch="+1%">{safe_text}</prosody></voice>')
                ssml_parts.append('</speak>')
                ssml_text = ''.join(ssml_parts)

                # Prepare single output file
                filename = f"podcast_{timestamp}.wav"
                audio_path = AUDIO_DIR / filename

                speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
                speech_config.speech_synthesis_voice_name = 'en-US-JennyNeural'
                # Use WAV output for browser-friendly playback
                audio_config = speechsdk.audio.AudioOutputConfig(filename=str(audio_path))
                synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

                result = synthesizer.speak_ssml_async(ssml_text).get()
                if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                    duration_est = max(1, int(len(' '.join([t for _, t in merged])) / 12))
                    audio_files.append({
                        'speaker': 'A+B',
                        'filename': filename,
                        'text': full_text[:1000],
                        'duration_estimate': duration_est,
                        'segment_order': 1
                    })
                else:
                    print('Speech synthesis failed:', result.reason)

            except Exception as e:
                print('TTS error while generating combined podcast:', e)
        
        # Calculate total estimated duration
        total_duration = sum(file.get("duration_estimate", 0) for file in audio_files)
        
        return {
            "script": script,
            "audio_files": audio_files,
            "total_segments": len(audio_files),
            "estimated_duration_seconds": total_duration,
            "content_summary": {
                "selected_text_length": len(request.selected_text),
                "related_documents": len(set(s.get('document_name', '') for s in request.related_sections)),
                "insight_types": list(request.insights.keys()) if request.insights else [],
                "generation_timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        return {
            "script": f"Error generating podcast script: {str(e)}",
            "audio_files": [],
            "error": str(e),
            "fallback_message": "Audio generation encountered an error. Please try again."
        }

@app.get("/audio/{filename}")
async def get_audio_file(filename: str):
    """Serve generated audio files."""
    file_path = AUDIO_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        file_path, 
        media_type="audio/wav",
        filename=filename
    )

@app.get("/files/{filename}")
async def get_uploaded_file(filename: str):
    """Serve uploaded PDF files for viewing."""
    file_path = UPLOADS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    # Security check - only allow PDF files
    if not filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    return FileResponse(
        file_path, 
        media_type="application/pdf",
        filename=filename,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Access-Control-Allow-Origin": "*",
        }
    )

@app.get("/stats")
async def get_system_stats():
    """Get system statistics and status."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Document stats
    cursor.execute("SELECT COUNT(*) FROM documents")
    total_docs = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM documents WHERE processing_status = 'completed'")
    processed_docs = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM sections")
    total_sections = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM search_history")
    total_searches = cursor.fetchone()[0]
    
    # FAISS index stats
    faiss_stats = {
        "total_vectors": faiss_index.ntotal if faiss_index else 0,
        "vector_dimension": 384 if model else 0
    }
    
    conn.close()
    
    return {
        "documents": {
            "total": total_docs,
            "processed": processed_docs,
            "pending": total_docs - processed_docs
        },
        "sections": {"total": total_sections},
        "searches": {"total": total_searches},
        "vector_index": faiss_stats,
        "features": {
            "vector_search": vector_enabled,
            "llm_insights": llm_enabled,
            "audio_generation": TTS_AVAILABLE
        }
    }

# Serve the React frontend (if built)
frontend_path = Path(__file__).parent.parent / "build"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=frontend_path / "static"), name="static")
    
    @app.get("/{path:path}")
    async def serve_frontend(path: str):
        """Serve React frontend for any non-API route."""
        if path.startswith("api/"):
            raise HTTPException(status_code=404)
        
        file_path = frontend_path / path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        
        # Fallback to index.html for React routing
        return FileResponse(frontend_path / "index.html")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )