# Multimodal RAG API

A FastAPI-based backend for a multimodal RAG pipeline.

## Features
- **PDF Ingestion**: Upload and parse PDFs (text and images).
- **Embeddings**: Generate vector embeddings for text chunks.
- **Vector Store**: Store and retrieve chunks using ChromaDB.
- **LLM Agent**: Query the knowledge base using Google Gemini.

## Setup

### Prerequisites
- Python 3.10+
- Docker (optional)
- Google API Key (for Gemini)

### Environment Variables
Copy `.env.example` to `.env` and fill in your API key.
```bash
cp .env.example .env
```

### Local Development
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the server:
   ```bash
   uvicorn app.main:app --reload
   ```
3. Access API docs at `http://localhost:8000/docs`.

### Docker
1. Build the image:
   ```bash
   docker build -t multimodal-rag .
   ```
2. Run the container:
   ```bash
   docker run -p 8000:8000 --env-file .env multimodal-rag
   ```

## Usage

### Upload PDF
POST `/api/v1/upload`
Form-data: `file` (@yourfile.pdf)

### Query
POST `/api/v1/query`
JSON:
```json
{
  "query": "What does the document say about X?"
}
```
