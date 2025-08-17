#!/bin/bash
# Complete backend cleanup and restart script

echo "🧹 Cleaning backend completely..."

# 1. Kill all Python processes (including the backend server)
echo "🔄 Stopping all Python processes..."
taskkill //F //IM python.exe 2>/dev/null || true
pkill -f "python main.py" 2>/dev/null || true
pkill -f "python" 2>/dev/null || true

# Wait a moment for processes to terminate
sleep 2

# 2. Clear Python cache
echo "🗑️ Clearing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

# 3. Verify FAISS index is the new one
echo "🔍 Verifying FAISS index..."
cd /d/document-insight-system/backend
python -c "
import faiss
from pathlib import Path
index_file = Path('./data/embeddings/faiss.index')
if index_file.exists():
    faiss_index = faiss.read_index(str(index_file))
    print(f'✅ FAISS index verified: {faiss_index.ntotal} vectors')
else:
    print('❌ FAISS index not found')
"

# 4. Restart backend with fresh environment
echo "🚀 Starting clean backend server..."
cd /d/document-insight-system/backend
python main.py

echo "✅ Backend cleanup and restart complete!"
