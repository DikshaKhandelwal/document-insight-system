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

# Base directory (directory containing this file). Use this to build absolute paths
# so the server behaves the same regardless of current working directory.
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from backend/.env first (falls back to process env)
load_dotenv(BASE_DIR / '.env')

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

# Enable FAISS by default if available; will fall back to MiniLM/text search if not.
VECTOR_AVAILABLE = False
model = None
faiss_index = None
print("� Attempting to initialize FAISS vector search. Will fall back to MiniLM/text search if unavailable.")

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

# Global configurations (resolved relative to backend folder)
DATA_DIR = BASE_DIR / "data"
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


    # Prepare index storage path. Honor FAISS_INDEX_PATH env var; if it's relative
    # treat it as relative to the backend directory so behavior is deterministic.
    env_path = os.getenv('FAISS_INDEX_PATH')
    if env_path:
        candidate = Path(env_path)
        if not candidate.is_absolute():
            faiss_store = BASE_DIR / candidate
        else:
            faiss_store = candidate
    else:
        faiss_store = BASE_DIR / 'data' / 'embeddings'
    if not faiss_store.exists():
        try:
            faiss_store.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created FAISS embeddings directory: {faiss_store.resolve()}")
        except Exception as e:
            print(f"❌ Could not create FAISS embeddings directory {faiss_store}: {e}")
            VECTOR_AVAILABLE = False
            return False
    try:
        faiss_store.mkdir(parents=True, exist_ok=True)
        print(f"✅ FAISS index directory: {faiss_store.resolve()}")
    except Exception as e:
        print(f"❌ Could not create FAISS index directory {faiss_store}: {e}")
        VECTOR_AVAILABLE = False
        return False
    index_file = faiss_store / 'faiss.index'
    print(f"ℹ️ FAISS index file: {index_file.resolve()}")

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
                
                # Reload the latest FAISS index from disk to ensure we have the most current version
                faiss_store = Path(os.getenv('FAISS_INDEX_PATH', './data/embeddings'))
                index_file = faiss_store / 'faiss.index'
                if index_file.exists():
                    faiss_index = faiss.read_index(str(index_file))
                    print(f"🔄 Reloaded FAISS index with {faiss_index.ntotal} vectors")
                
                print(f"🔍 Computing embeddings for {len(embeddings_to_add)} sections...")
                ids = [sid for sid, txt in embeddings_to_add]
                texts = [txt for sid, txt in embeddings_to_add]
                # Encode to numpy
                embeddings = model.encode(texts, convert_to_numpy=True)
                # Normalize and add
                faiss.normalize_L2(embeddings)
                ids_arr = np.array(ids, dtype='int64')
                faiss_index.add_with_ids(embeddings.astype('float32'), ids_arr)

                # Persist index to disk and update global variable
                try:
                    faiss_store = Path(os.getenv('FAISS_INDEX_PATH', './data/embeddings'))
                    index_file = faiss_store / 'faiss.index'
                    faiss.write_index(faiss_index, str(index_file))
                    
                    # Update the global faiss_index variable to ensure future operations use the updated index
                    globals()['faiss_index'] = faiss_index
                    
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
            # Vector search not available or index empty - return no results
            print("❌ Vector search not available or FAISS index empty - returning no results")
            results = []
        
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
        print("❌ FAISS import failed at runtime - cannot perform vector search")
        return []

    if not (model and faiss_index and VECTOR_AVAILABLE):
        print("❌ Vector search dependencies missing - cannot perform vector search")
        return []

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

            full_text = row[2] or row[1] or ''
            snippet = full_text[:300] + '...' if len(full_text) > 300 else full_text

            # Use the actual section content for highlighting instead of selected text
            # This gives a much better chance of finding the text in the PDF
            highlight_candidate = full_text.strip() if full_text else request.selected_text
            
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
        return []

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
            CRITICAL: Ensure FREQUENT alternation between speakers - aim for 8-12 exchanges minimum.
            
            CONTENT TO DISCUSS:
            Selected Text: "{request.selected_text}"
            
            AI-Generated Insights:
            {insights_context}
            
            Related Documents from User's Library:
            {json.dumps(related_content, indent=2)}
            
            REQUIREMENTS:
            - 2-3 minutes of natural conversation when spoken
            - FREQUENT speaker alternation (short exchanges, not long monologues)
            - Speaker A (Male): Curious host, asks probing questions, guides conversation
            - Speaker B (Female): Expert analyst, provides insights, explains concepts
            - Include specific references to the user's documents
            - Natural conversational flow with "Hmm", "Interesting", "Right", etc.
            - Ground everything in the provided documents
            
            STRUCTURE (with frequent alternation):
            1. Speaker A: Brief intro to topic
            2. Speaker B: Quick explanation of key concept  
            3. Speaker A: Question about user's documents
            4. Speaker B: Analysis of document findings
            5. Speaker A: Follow-up question or contradiction
            6. Speaker B: Addresses the point
            7. Speaker A: Asks about implications
            8. Speaker B: Practical applications
            9. Speaker A: Final question
            10. Speaker B: Key takeaway
            11. Speaker A: Closing remarks
            12. Speaker B: Thank you/goodbye
            
            FORMAT: Always use "Speaker A: [text]" and "Speaker B: [text]"
            Keep individual segments 2-3 sentences max for natural flow.
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
                    # truncate merged segments to fit target length while preserving alternation
                    allowed_chars = int(target_secs * 12)
                    truncated_merged = []
                    current_chars = 0
                    for speaker, text in merged:
                        if current_chars + len(text) <= allowed_chars:
                            truncated_merged.append((speaker, text))
                            current_chars += len(text)
                        else:
                            # Partial text to fit limit
                            remaining = allowed_chars - current_chars
                            if remaining > 50:  # Only add if meaningful length
                                truncated_merged.append((speaker, text[:remaining]))
                            break
                    merged = truncated_merged

                # Ensure we have at least 2 segments for voice alternation
                if len(merged) < 2:
                    # Split single segment into two parts for alternation
                    if merged:
                        original_text = merged[0][1]
                        mid_point = len(original_text) // 2
                        # Find a good break point near the middle
                        break_point = mid_point
                        for i in range(mid_point - 50, mid_point + 50):
                            if i < len(original_text) and original_text[i] in '.!?':
                                break_point = i + 1
                                break
                        
                        part1 = original_text[:break_point].strip()
                        part2 = original_text[break_point:].strip()
                        merged = [('A', part1), ('B', part2)]
                    else:
                        merged = [('A', 'Welcome to this podcast discussion.'), ('B', 'Thank you for joining us today.')]

                # Build SSML with alternating configured voices (male/female)
                # Default to Indian voices for centralindia region if not provided
                male_voice = os.getenv('AZURE_TTS_VOICE_MALE', os.getenv('AZURE_TTS_VOICE_1', 'en-IN-PrabhatNeural'))
                female_voice = os.getenv('AZURE_TTS_VOICE_FEMALE', os.getenv('AZURE_TTS_VOICE_2', 'en-IN-NeerjaNeural'))
                
                print(f"DEBUG: Using voices - Male: {male_voice}, Female: {female_voice}")
                print(f"DEBUG: Merged segments: {[(s, t[:50]+'...' if len(t) > 50 else t) for s, t in merged]}")
                
                ssml_parts = ['<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-IN">']
                for idx, (speaker, text) in enumerate(merged):
                    # Map Speaker A to male, Speaker B to female (regardless of order)
                    voice = male_voice if speaker == 'A' else female_voice
                    print(f"DEBUG: Segment {idx}: Speaker {speaker} -> Voice {voice}")
                    
                    # Escape any problematic characters
                    safe_text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    
                    # Add more prosody variation for podcast style
                    if speaker == 'A':
                        # Male host - slightly lower pitch, authoritative
                        prosody = '<prosody rate="0.92" pitch="-2%">'
                    else:
                        # Female expert - slightly higher pitch, engaging
                        prosody = '<prosody rate="0.95" pitch="+3%">'
                    
                    ssml_parts.append(f'<voice name="{voice}">{prosody}{safe_text}</prosody></voice>')
                    # Add small pause between speakers for natural flow
                    ssml_parts.append('<break time="0.5s"/>')
                
                ssml_parts.append('</speak>')
                ssml_text = ''.join(ssml_parts)
                
                print(f"DEBUG: Generated SSML length: {len(ssml_text)} chars")

                # Prepare single output file (write to a temp file first to avoid zero-byte final files)
                filename = f"podcast_{timestamp}.wav"
                audio_path = AUDIO_DIR / filename
                audio_tmp_path = AUDIO_DIR / f"{filename}.part"

                speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
                # Default synthesis voice (SSML voice tags will override per-segment)
                speech_config.speech_synthesis_voice_name = male_voice
                # Use WAV output into a temporary file
                audio_config = speechsdk.audio.AudioOutputConfig(filename=str(audio_tmp_path))
                synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

                result = synthesizer.speak_ssml_async(ssml_text).get()
                # Debug: report synthesis result reason and temp file status
                try:
                    print('TTS synth result reason:', getattr(result, 'reason', 'unknown'))
                except Exception:
                    pass
                try:
                    if audio_tmp_path.exists():
                        print('Temp audio path exists, size=', audio_tmp_path.stat().st_size)
                    else:
                        print('Temp audio path does not exist:', audio_tmp_path)
                except Exception as e:
                    print('Error checking temp audio path:', e)

                # Try to release SDK resources (helps Windows release file handles)
                try:
                    # Azure SDK doesn't have close() method, but we can delete the object
                    del synthesizer
                except Exception:
                    pass

                if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                    # Ensure the temp file was actually written and is non-empty before renaming
                    try:
                        if audio_tmp_path.exists() and audio_tmp_path.stat().st_size > 0:
                            # Retry moving the file a few times on Windows to avoid file-lock errors
                            moved = False
                            for attempt in range(8):
                                try:
                                    if audio_path.exists():
                                        audio_path.unlink()
                                    # Prefer atomic replace when available
                                    try:
                                        audio_tmp_path.replace(audio_path)
                                    except Exception:
                                        audio_tmp_path.rename(audio_path)
                                    moved = True
                                    break
                                except PermissionError as pe:
                                    # File might still be held by the synthesizer - wait and retry
                                    await asyncio.sleep(0.25)
                                except Exception as e:
                                    # Unexpected error - log and break
                                    print('Error while moving temp audio file (attempt', attempt + 1, '):', e)
                                    await asyncio.sleep(0.1)

                            if not moved:
                                print('Error: could not atomically move temp audio file to final path after retries:', audio_tmp_path)
                                # leave temp file for diagnostics
                            else:
                                duration_est = max(1, int(len(' '.join([t for _, t in merged])) / 12))
                                audio_files.append({
                                    'speaker': 'male+female',
                                    'filename': filename,
                                    'text': full_text[:1000],
                                    'duration_estimate': duration_est,
                                    'segment_order': 1
                                })
                        else:
                            print('Warning: synthesis reported completed but temp output file is empty or missing:', audio_tmp_path)
                            # Attempt to remove zero-byte temp file if present
                            try:
                                if audio_tmp_path.exists() and audio_tmp_path.stat().st_size == 0:
                                    audio_tmp_path.unlink()
                            except Exception as e:
                                print('Could not remove empty temp audio file:', e)
                    except Exception as e:
                        print('Error while verifying/renaming output audio file:', e)
                        try:
                            if audio_tmp_path.exists():
                                audio_tmp_path.unlink()
                        except Exception:
                            pass
                else:
                    print('Speech synthesis failed:', result.reason)
                    # Cleanup any partial temp file
                    try:
                        if audio_tmp_path.exists():
                            audio_tmp_path.unlink()
                    except Exception:
                        pass

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


@app.post('/reload_index')
async def reload_faiss_index():
    """Reload the FAISS index from disk into the running process.
    This helps when the on-disk index was rebuilt outside the running server
    and we want the process to pick up the new vectors without a restart.
    """
    global faiss_index
    try:
        import faiss
        faiss_store = Path(os.getenv('FAISS_INDEX_PATH', './data/embeddings'))
        index_file = faiss_store / 'faiss.index'
        if not index_file.exists():
            return {"ok": False, "message": f"FAISS index file not found: {index_file}"}

        # Read index and replace in-memory object
        new_index = faiss.read_index(str(index_file))
        faiss_index = new_index
        print(f"🔁 Reloaded FAISS index from {index_file} with {faiss_index.ntotal} vectors")
        return {"ok": True, "total_vectors": faiss_index.ntotal}
    except Exception as e:
        print('❌ Failed to reload FAISS index:', e)
        return {"ok": False, "error": str(e)}

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