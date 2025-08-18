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
from fastapi import Request

# Base directory (directory containing this file). Use this to build absolute paths
# so the server behaves the same regardless of current working directory.
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from backend/.env (falls back to process env)
load_dotenv(BASE_DIR.parent / '.env')

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

        # Normalize filename: caller may pass a document id (int), a Path, or a filename string.
        resolved_filename = None
        try:
            # If a Path was passed, use its name
            if isinstance(filename, Path):
                resolved_filename = filename.name
            # If an int or numeric string was passed, look up filename from DB
            elif isinstance(filename, int) or (isinstance(filename, str) and filename.isdigit()):
                doc_id = int(filename)
                try:
                    conn = sqlite3.connect(DB_PATH)
                    cur = conn.cursor()
                    cur.execute("SELECT filename FROM documents WHERE id = ?", (doc_id,))
                    row = cur.fetchone()
                    conn.close()
                    if row:
                        resolved_filename = row[0]
                except Exception:
                    resolved_filename = None
            else:
                # Expect a filename string
                resolved_filename = str(filename)
        except Exception:
            resolved_filename = str(filename)

        if not resolved_filename:
            return None

        file_path = UPLOADS_DIR / resolved_filename
        if not file_path.exists():
            return None

        # page_number is expected to be 1-based in DB; convert to 0-based index
        try:
            if isinstance(page_number, Path):
                page_number = int(page_number.name)
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

    # Ensure filename is a string and page_number is an int
    try:
        if filename is not None and not isinstance(filename, str):
            filename = str(filename)
        if page_number is not None and not isinstance(page_number, int):
            page_number = int(page_number)
    except Exception:
        # If conversion fails, skip page text extraction
        filename = None
        page_number = None

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


def strip_intro_outro(text: str, remove_sentences: int = 1) -> str:
    """Remove likely introductory or concluding sentences from text to focus semantic embeddings on core content.
    This uses a simple heuristic: drop very short leading sentences or sentences containing common intro/executive words,
    and drop trailing sentences that look like conclusions or references.
    """
    if not text:
        return text
    try:
        # Split into sentences conservatively
        sents = re.split(r'(?<=[\.\!\?])\s+', text.strip())
        if len(sents) <= 2:
            return text

        # stronger heuristics
        def looks_intro(s: str) -> bool:
            sl = s.strip().lower()
            intro_kw = ['introduction', 'abstract', 'overview', 'background', 'preface', 'table of contents', 'contents', 'about this', 'about the']
            short = len(sl) < 60
            phrase = any(k in sl for k in intro_kw)
            starts_like = sl.startswith('in this') or sl.startswith('this chapter') or sl.startswith('this paper') or sl.startswith('we ') or sl.startswith('the aim')
            # More aggressive: if sentence contains "introduction" as heading/title, remove it
            heading_like = sl.startswith('introduction') or sl == 'introduction' or 'introduction:' in sl
            return short or phrase or starts_like or heading_like

        def looks_outro(s: str) -> bool:
            sl = s.strip().lower()
            outro_kw = ['conclusion', 'conclusions', 'in conclusion', 'summary', 'references', 'acknowledg', 'further work', 'future work', 'thanks', 'thank you']
            short = len(sl) < 60
            phrase = any(k in sl for k in outro_kw)
            # More aggressive: if sentence contains "conclusion" as heading/title, remove it
            heading_like = sl.startswith('conclusion') or sl == 'conclusion' or 'conclusion:' in sl or sl.startswith('conclusions')
            return short or phrase or heading_like

        start = 0
        end = len(sents)
        # aggressively remove up to N leading intro-like sentences
        max_leading = 3
        leading = 0
        while leading < max_leading and start < end and looks_intro(sents[start]):
            start += 1
            leading += 1

        # aggressively remove up to N trailing outro-like sentences
        max_trailing = 3
        trailing = 0
        while trailing < max_trailing and end - 1 >= start and looks_outro(sents[end - 1]):
            end -= 1
            trailing += 1

        # If we removed almost everything, fall back to original conservative behavior
        if end - start <= 1:
            # try to keep the longest middle sentence instead of empty
            mid = max(sents, key=lambda s: len(s))
            return mid.strip()

        trimmed = ' '.join(sents[start:end]).strip()

        # Remove obvious boilerplate lines inside the text
        boilerplate_patterns = ['table of contents', 'references', 'acknowledgements', 'copyright', 'all rights reserved']
        lowered = trimmed.lower()
        for bp in boilerplate_patterns:
            if bp in lowered:
                # drop everything from the pattern onwards
                idx = lowered.find(bp)
                trimmed = trimmed[:idx].strip()
                break

        return trimmed if trimmed else text
    except Exception:
        return text

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

    # Prepare index storage path (resolve relative paths against backend folder)
    faiss_store_env = os.getenv('FAISS_INDEX_PATH', './data/embeddings')
    faiss_store = Path(faiss_store_env)
    if not faiss_store.is_absolute():
        faiss_store = (BASE_DIR / faiss_store).resolve()
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
                print(f"⚠️ Failed to read existing FAISS index: {e}. Creating new IndexIDMap-backed index.")
                # create fresh IndexIDMap-backed index
                base = faiss.IndexFlatIP(EMBEDDING_DIM)
                faiss_index = faiss.IndexIDMap(base)
        else:
            base = faiss.IndexFlatIP(EMBEDDING_DIM)
            faiss_index = faiss.IndexIDMap(base)
            print(f"✅ Created new FAISS IndexIDMap index (dim={EMBEDDING_DIM})")

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


@app.post("/qa")
async def qa_endpoint(request: Request):
    """
    Answer user questions about a document using Gemini LLM.
    """
    if not LLM_AVAILABLE:
        return {"answer": "Gemini LLM is not available. Please check your API key and setup."}

    body = await request.json()
    print("Received /qa request:", body)
    question = body.get("question", "")
    document_id = body.get("document_id")

    # Fetch document context (sections) if document_id is provided
    context_text = ""
    if document_id:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM documents WHERE filename = ?", (document_id,))
        doc_row = cursor.fetchone()
        if doc_row:
            doc_id = doc_row[0]
            cursor.execute("SELECT section_title, section_text FROM sections WHERE document_id = ? ORDER BY page_number, id LIMIT 10", (doc_id,))
            sections = cursor.fetchall()
            for title, text in sections:
                context_text += f"Section: {title}\n{text}\n---\n"
        conn.close()

    # Build prompt for Gemini
    if context_text:
        prompt = f"""
        You are an academic research assistant. Answer the user's question using the provided document context below. If the answer is not present, you may use your own knowledge to help the user.

        Document Context:
        {context_text}

        User Question: {question}
        """
    else:
        prompt = f"""
        You are an academic research assistant. Answer the user's question as helpfully as possible. If document context is provided, use it. Otherwise, answer from your own knowledge.

        User Question: {question}
        """

    try:
        model_gemini = genai.GenerativeModel('gemini-1.5-flash')
        response = model_gemini.generate_content(prompt)
        print("Gemini response:", response.text)
        answer = response.text.strip()
    except Exception as e:
        import traceback
        print("Gemini API error in /qa endpoint:", traceback.format_exc())
        answer = f"Error generating answer: {str(e)}"

    return {"answer": answer}

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
                    # Resolve faiss_store the same way as init_vector_search
                    faiss_store_env = os.getenv('FAISS_INDEX_PATH', './data/embeddings')
                    faiss_store = Path(faiss_store_env)
                    if not faiss_store.is_absolute():
                        faiss_store = (BASE_DIR / faiss_store).resolve()
                    faiss_store.mkdir(parents=True, exist_ok=True)
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
        try:
            faiss.normalize_L2(query_vec)
        except Exception:
            # fallback normalization
            qnorm = np.linalg.norm(query_vec, axis=1, keepdims=True)
            qnorm[qnorm == 0] = 1.0
            query_vec = query_vec / qnorm

        k = min(max(request.max_results or 8, 1), 50)
        distances, ids = faiss_index.search(query_vec.astype('float32'), k)

        results = []

        # Build list of candidate ids in the same order as ids[0] (skip -1)
        candidate_pos = [pos for pos, raw in enumerate(ids[0]) if raw != -1]
        candidate_ids = [int(ids[0][pos]) for pos in candidate_pos]
        if not candidate_ids:
            return []

        # Fetch matching sections from DB by id
        placeholders = ','.join('?' for _ in candidate_ids)
        cursor.execute(f"SELECT id, section_title, section_text, page_number, document_id FROM sections WHERE id IN ({placeholders})", tuple(candidate_ids))
        rows = cursor.fetchall()
        row_map = {r[0]: r for r in rows}

        # Prepare texts in the same order for MiniLM scoring if available.
        # Augment each section with page-level text and neighboring sections to give MiniLM more context.
        candidate_texts = []
        candidate_rows = []
        # Collect document ids for candidates so we can load neighboring sections in bulk
        doc_ids = set()
        for cid in candidate_ids:
            row = row_map.get(cid)
            if row:
                doc_ids.add(row[4])

        # Load all sections for these documents to find neighbors
        neighbors_map = {}
        try:
            if doc_ids:
                placeholders = ','.join('?' for _ in doc_ids)
                cursor.execute(f"SELECT id, document_id, section_title, section_text, page_number FROM sections WHERE document_id IN ({placeholders}) ORDER BY document_id, page_number, id", tuple(doc_ids))
                all_doc_sections = cursor.fetchall()
                # build mapping: document_id -> ordered list of sections
                cur_map = {}
                for r in all_doc_sections:
                    did = r[1]
                    cur_map.setdefault(did, []).append(r)
                neighbors_map = cur_map
        except Exception:
            neighbors_map = {}

        for cid in candidate_ids:
            row = row_map.get(cid)
            if not row:
                candidate_rows.append(None)
                candidate_texts.append("")
                continue
            candidate_rows.append(row)

            # Base text: section text or title
            base_text = row[2] or row[1] or ""

            # Page-level context (if available)
            page_ctx = ""
            try:
                # row[4] is document_id; need filename for extract_page_text_from_pdf
                cursor.execute("SELECT filename FROM documents WHERE id = ?", (row[4],))
                doc_row = cursor.fetchone()
                doc_name = doc_row[0] if doc_row else None
                if doc_name and row[3]:
                    page_ctx = extract_page_text_from_pdf(doc_name, row[3]) or ""
            except Exception:
                page_ctx = ""

            # Neighboring sections (previous + next) from the same document
            neighbor_ctx = ""
            try:
                doc_sections = neighbors_map.get(row[4], [])
                # find index
                idx_in_doc = None
                for i, s in enumerate(doc_sections):
                    if s[0] == row[0]:
                        idx_in_doc = i
                        break
                if idx_in_doc is not None:
                    # previous
                    if idx_in_doc - 1 >= 0:
                        prev = doc_sections[idx_in_doc - 1]
                        neighbor_ctx += (prev[3] or prev[2] or '') + '\n'
                    # next
                    if idx_in_doc + 1 < len(doc_sections):
                        nxt = doc_sections[idx_in_doc + 1]
                        neighbor_ctx += (nxt[3] or nxt[2] or '')
            except Exception:
                neighbor_ctx = ""

            combined = base_text
            if page_ctx:
                combined = combined + '\n\n' + page_ctx
            if neighbor_ctx:
                combined = combined + '\n\n' + neighbor_ctx

            candidate_texts.append(combined)

        # Compute MiniLM similarities for candidates in batch if available
        minilm_sims = None
        try:
            if SENTENCE_TRANSFORMER_AVAILABLE:
                emb_model = load_minilm_model()
                if emb_model is not None:
                    q_emb = emb_model.encode([request.selected_text], convert_to_numpy=True)
                    # strip intro/outro to focus embeddings on core content
                    cleaned_texts = [strip_intro_outro(t) for t in candidate_texts]
                    sec_embs = emb_model.encode(cleaned_texts, convert_to_numpy=True)
                    # normalize
                    sec_norms = np.linalg.norm(sec_embs, axis=1, keepdims=True)
                    sec_norms[sec_norms == 0] = 1.0
                    sec_embs = sec_embs / sec_norms
                    q_norm = np.linalg.norm(q_emb)
                    if q_norm == 0:
                        q_norm = 1.0
                    q_emb = q_emb / q_norm
                    # cosine similarities
                    try:
                        minilm_sims = cosine_similarity(q_emb, sec_embs)[0].tolist()
                    except Exception:
                        minilm_sims = (sec_embs @ q_emb.reshape(-1,)).tolist()
        except Exception as e:
            print(f"⚠️ Error computing MiniLM sims for FAISS candidates: {e}")

        # Iterate in original ids order and compute combined score using MiniLM (heavy) + FAISS + title boost
        ml_index = 0
        for idx, raw_id in enumerate(ids[0]):
            if raw_id == -1:
                continue
            sid = int(raw_id)
            row = row_map.get(sid)
            if not row:
                continue

            # FAISS similarity at this position
            faiss_sim = float(distances[0][idx])

            # Document filename
            cursor.execute("SELECT filename FROM documents WHERE id = ?", (row[4],))
            doc_row = cursor.fetchone()
            doc_name = doc_row[0] if doc_row else 'Unknown'

            full_text = row[2] or ''
            snippet = build_snippet(full_text, row[1], filename=doc_name, page_number=row[3])
            highlight_candidate = (full_text.strip() if full_text else row[1] or request.selected_text)

            # MiniLM similarity for this candidate (if computed)
            minilm_sim = 0.0
            try:
                if minilm_sims is not None and ml_index < len(minilm_sims):
                    minilm_sim = float(minilm_sims[ml_index])
            except Exception:
                minilm_sim = 0.0

            # Title boost/penalty: boost relevant terms, penalize intro/conclusion headings
            title = (row[1] or '').lower()
            title_adjustment = 0.0
            try:
                q_words = set([w.strip() for w in re.split(r"\W+", request.selected_text.lower()) if w.strip()])
                title_words = set([w.strip() for w in re.split(r"\W+", title) if w.strip()])

                # Check for intro/conclusion headings and penalize them
                intro_conclusion_terms = ['introduction', 'conclusion', 'conclusions', 'abstract', 'summary']
                if any(term in title for term in intro_conclusion_terms):
                    title_adjustment = -0.15  # Penalize intro/conclusion sections
                elif q_words and len(q_words & title_words) > 0:
                    title_adjustment = 0.10  # Boost relevant terms
            except Exception:
                title_adjustment = 0.0

            # Normalize faiss sim
            try:
                faiss_sim_norm = max(0.0, min(1.0, float(faiss_sim)))
            except Exception:
                faiss_sim_norm = 0.0

            # Combine signals with higher weight for MiniLM
            combined_score = (0.55 * minilm_sim) + (0.35 * faiss_sim_norm) + title_adjustment

            # Apply threshold
            threshold = float(request.similarity_threshold or 0.0)
            if combined_score < threshold:
                ml_index += 1
                continue

            results.append(SearchResult(
                document_name=doc_name,
                section_title=row[1] or 'Untitled',
                snippet=snippet,
                page_number=row[3] or 1,
                similarity_score=combined_score,
                highlight_text=highlight_candidate
            ))
            ml_index += 1

        # Sort and return top-k
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        final = results[: (request.max_results or 8) ]

        # If MiniLM is available, and we have fewer than requested results from FAISS,
        # supplement with MiniLM semantic search to improve recall (cross-document).
        try:
            if SENTENCE_TRANSFORMER_AVAILABLE and len(final) < (request.max_results or 8):
                remaining = (request.max_results or 8) - len(final)
                # Fetch all sections for MiniLM semantic search
                cursor.execute("""
                    SELECT s.id, s.section_title, s.section_text, s.page_number, d.filename, d.title
                    FROM sections s
                    JOIN documents d ON s.document_id = d.id
                    WHERE d.processing_status = 'completed'
                    ORDER BY s.id
                """)
                all_sections = cursor.fetchall()
                ml_results = await minilm_semantic_search(request, all_sections)

                # Dedupe by section (use document_name+section_title or highlight_text) -- prefer section id mapping
                existing_ids = set()
                for r in final:
                    # We can try to map by snippet and title if id not present; best-effort
                    existing_ids.add((r.document_name, r.section_title))

                for mr in ml_results:
                    if remaining <= 0:
                        break
                    key = (mr.document_name, mr.section_title)
                    if key in existing_ids:
                        continue
                    final.append(mr)
                    existing_ids.add(key)
                    remaining -= 1
        except Exception as e:
            print(f"⚠️ Error while supplementing FAISS results with MiniLM: {e}")

        search_time = (datetime.now() - start_time).total_seconds()
        cursor.execute("UPDATE search_history SET results_count = ?, search_time = CURRENT_TIMESTAMP WHERE id = ?", (len(final), search_id))
        print(f"🔍 FAISS vector search completed in {search_time:.3f}s -> {len(final)} results (after supplement)")
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
            section_text = strip_intro_outro(section_text)
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


        audio_files = []
        if LLM_AVAILABLE:
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
            import google.generativeai as genai
            model_gemini = genai.GenerativeModel('gemini-1.5-flash')
            response = model_gemini.generate_content(script_prompt)
            script = response.text.strip()
        else:
            insight_text = (
                f"Looking at your insights, we see {', '.join([k for k in request.insights.keys() if k != 'status'])} themes emerging."
                if request.insights else
                "The semantic analysis reveals interesting thematic connections across your documents."
            )
            script = (
                f"Speaker A: Welcome to your personalized research insight podcast! Today we're diving deep into a fascinating topic from your document library.\n"
                f"Speaker B: That's right! We're exploring \"{request.selected_text[:100]}...\" and I have to say, the connections we found across your documents are really intriguing.\n"
                "Speaker A: Tell us more about what you discovered.\n"
                f"Speaker B: Well, we analyzed {len(request.related_sections)} related sections from your personal research collection, and there are some compelling patterns emerging.\n"
                "Speaker A: What kind of patterns are we talking about?\n"
                f"Speaker B: {insight_text}\n"
                "Speaker A: That's fascinating. How does this connect to the broader research landscape you've been building?\n"
                "Speaker B: What's particularly interesting is that this isn't just theoretical - your document collection shows practical applications and real-world implementations that bridge different research areas.\n"
                "Speaker A: For our listeners who want to explore this further, where should they look next in their document collection?\n"
                "Speaker B: I'd recommend diving deeper into the related sections we identified - they offer complementary perspectives that could spark new research directions or validate current approaches.\n"
                "Speaker A: Excellent insights! Thanks for this deep dive into your personalized research connections.\n"
            )
        audio_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        speech_key = os.getenv("AZURE_SPEECH_KEY")
        speech_region = os.getenv("AZURE_SPEECH_REGION")
        total_duration = 0
        if not (speech_key and speech_region):
            print("Azure Speech credentials not found. Audio files not generated.")
        else:
            try:
                # Build alternating SSML using two voices
                segments = []
                for raw in script.split('\n'):
                    if not raw.strip():
                        continue
                    if raw.startswith('Speaker A:') or raw.startswith('Host A:'):
                        segments.append(('A', raw.split(':', 1)[1].strip()))
                    elif raw.startswith('Speaker B:') or raw.startswith('Expert B:'):
                        segments.append(('B', raw.split(':', 1)[1].strip()))
                    else:
                        if segments:
                            segments[-1] = (segments[-1][0], segments[-1][1] + ' ' + raw.strip())
                        else:
                            segments.append(('A', raw.strip()))
                # Build SSML with alternating voices for each segment
                full_text = ' '.join([t for _, t in segments])
                ssml_parts = [
                    '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="en-US">'
                ]
                for speaker, text in segments:
                    if speaker == 'A':
                        voice = 'en-US-AvaMultilingualNeural'
                        style = 'cheerful'
                        role = 'YoungAdultFemale'
                    else:
                        voice = 'en-US-AndrewMultilingualNeural'
                        style = 'calm'
                        role = 'YoungAdultMale'
                    safe_text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    ssml_parts.append(
                        f'<voice name="{voice}">'
                        f'<mstts:express-as style="{style}" role="{role}">'
                        f'<prosody rate="0.95" pitch="+1%">{safe_text}</prosody>'
                        f'</mstts:express-as></voice>'
                    )
                ssml_parts.append('</speak>')
                ssml_text = ''.join(ssml_parts)
                print("\n--- SSML to be synthesized ---\n")
                print(ssml_text)
                print("\n--- End SSML ---\n")
                filename = f"podcast_{timestamp}.wav"
                audio_path = AUDIO_DIR / filename
                import azure.cognitiveservices.speech as speechsdk
                speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
                audio_config = speechsdk.audio.AudioOutputConfig(filename=str(audio_path))
                synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
                result = synthesizer.speak_ssml_async(ssml_text).get()
                print("Audio file saved to:", audio_path)
                print("Azure TTS result reason:", result.reason)
                if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                    duration_est = max(1, int(len(full_text) / 12))
                    audio_files.append({
                        'speaker': 'A+B',
                        'filename': filename,
                        'text': full_text[:1000],
                        'duration_estimate': duration_est,
                        'segment_order': 1
                    })
                    total_duration = duration_est
                else:
                    print('Speech synthesis failed:', result.reason)
            except Exception as e:
                print('TTS error while generating combined podcast:', e)
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

    # except Exception as e:
    #     return {
    #         "script": f"Error generating podcast script: {str(e)}",
    #         "audio_files": [],
    #         "error": str(e),
    #         "fallback_message": "Audio generation encountered an error. Please try again."
    #     }

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

@app.delete("/documents/{document_id}")
async def delete_document(document_id: int):
    """Delete a document and all its associated data."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # First, get the document filename to delete the physical file
        cursor.execute("SELECT filename FROM documents WHERE id = ?", (document_id,))
        result = cursor.fetchone()
        
        if not result:
            conn.close()
            raise HTTPException(status_code=404, detail="Document not found")
        
        filename = result[0]
        
        # Delete physical file
        file_path = UPLOADS_DIR / filename
        if file_path.exists():
            file_path.unlink()
        
        # Delete from database (foreign key constraints should cascade)
        # Delete sections first
        cursor.execute("DELETE FROM sections WHERE document_id = ?", (document_id,))
        
        # Delete search history related to this document
        cursor.execute("""
            DELETE FROM search_history 
            WHERE query_text IN (
                SELECT section_text FROM sections WHERE document_id = ?
            )
        """, (document_id,))
        
        # Delete the document record
        cursor.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        
        # Update FAISS index by removing embeddings for this document
        # Note: For simplicity, we'll rebuild the index periodically
        # In production, you might want to remove specific embeddings
        
        conn.commit()
        conn.close()
        
        return {"message": "Document deleted successfully", "document_id": document_id}
        
    except Exception as e:
        conn.rollback()
        conn.close()
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


@app.post("/reload_index")
async def reload_faiss_index():
    """Reload FAISS index from disk. Returns index stats or error."""
    global faiss_index, VECTOR_AVAILABLE
    try:
        import faiss
        faiss_store_env = os.getenv('FAISS_INDEX_PATH', './data/embeddings')
        faiss_store = Path(faiss_store_env)
        if not faiss_store.is_absolute():
            faiss_store = (BASE_DIR / faiss_store).resolve()
        index_file = faiss_store / 'faiss.index'
        if not index_file.exists():
            return {"ok": False, "error": f"Index file not found at {index_file}"}

        faiss_index = faiss.read_index(str(index_file))
        VECTOR_AVAILABLE = True
        return {"ok": True, "total_vectors": faiss_index.ntotal}
    except Exception as e:
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