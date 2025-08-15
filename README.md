# Document Insight System

A comprehensive document analysis platform with AI-powered insights, semantic search, and audio summaries.

## Features

- 📄 **PDF Upload & Processing**: Upload and process PDF documents with automatic text extraction
- 🔍 **Semantic Search**: Find relevant content using natural language queries
- 🤖 **AI Insights**: Generate intelligent insights and summaries using Google Gemini
- 🎵 **Audio Summaries**: Convert insights to audio using Azure Text-to-Speech
- 📱 **Interactive UI**: Modern React interface with PDF viewer and text selection
- 🏗️ **Modular Architecture**: FastAPI backend with Next.js frontend

## Quick Start

### Prerequisites
- Python 3.8+
- Node.js 18+
- Azure account with Speech Services enabled

### Setup

1. **Install dependencies**:
   ```bash
   # Frontend
   npm install --legacy-peer-deps
   
   # Backend
   cd backend
   pip install -r requirements.txt
   cd ..
   ```

2. **Configure environment variables**:
   - Copy `backend/.env.example` to `backend/.env`
   - Add your Google Gemini API key
   - Set up Azure Speech Services credentials (see below)

3. **Start the servers**:
   ```bash
   # Terminal 1: Backend
   cd backend
   uvicorn main:app --reload --port 8000
   
   # Terminal 2: Frontend  
   npm run dev
   ```

4. **Access the application**:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

## Azure Speech Services Setup

1. **Create an Azure Account** (if you don't have one)
2. **Create a Speech Services resource**:
   - Go to Azure Portal
   - Create a new resource > AI + Machine Learning > Speech Services
   - Choose your subscription, resource group, and region
   - Select a pricing tier (F0 free tier available)

3. **Get your credentials**:
   - Go to your Speech Services resource
   - Navigate to "Keys and Endpoint"
   - Copy Key 1 and Region

4. **Set the credentials in your .env file**:
   ```
   AZURE_SPEECH_KEY=your_copied_key_here
   AZURE_SPEECH_REGION=your_region_here
   ```

## Environment Variables

### Backend (.env)
```
GEMINI_API_KEY=your_gemini_api_key
AZURE_SPEECH_KEY=your_azure_speech_key
AZURE_SPEECH_REGION=your_azure_region
DATABASE_URL=sqlite:///./documents.db
```

### Frontend (.env.local)
```
BACKEND_URL=http://localhost:8000
NEXT_PUBLIC_APP_NAME="Document Insight System"
```

## API Endpoints

- `POST /upload` - Upload PDF documents
- `POST /search` - Semantic search in documents
- `POST /insights` - Generate AI insights
- `POST /audio` - Generate audio summaries
- `GET /documents` - List all documents
- `DELETE /documents/{id}` - Delete a document

## Technology Stack

### Backend
- **FastAPI**: Modern Python web framework
- **SQLite**: Document metadata storage
- **FAISS**: Vector similarity search
- **Sentence Transformers**: Text embeddings
- **Google Gemini**: AI insights generation
- **Azure Speech Services**: Text-to-speech conversion

### Frontend
- **Next.js 14**: React framework with App Router
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first styling
- **shadcn/ui**: Modern UI components
- **PDF.js**: Client-side PDF rendering
- **React Dropzone**: File upload handling

## Development Workflow

1. **Upload PDF**: Use the drag-and-drop interface to upload documents
2. **Text Selection**: Select text in the PDF viewer to trigger semantic analysis
3. **Search**: Use natural language queries to find relevant content
4. **Generate Insights**: AI-powered analysis of selected content or search results
5. **Audio Playback**: Listen to generated insights as audio summaries

## License

MIT License
