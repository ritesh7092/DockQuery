from fastapi import APIRouter, UploadFile, File, HTTPException
from app.config import settings
from app.services.pdf_parser import parse_pdf
from app.services.embeddings import embedding_service
from app.services.vector_store import vector_store
from app.models.schemas import UploadResponse
import os
import shutil
import uuid

router = APIRouter()

@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF")

    # Generate unique filename
    file_id = str(uuid.uuid4())
    filename = f"{file_id}_{file.filename}"
    file_path = os.path.join(settings.UPLOAD_DIR, filename)

    # Save file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save file: {e}")

    # Process PDF
    try:
        content = parse_pdf(file_path, file_id)
        text = content["text"]
        
        # Simple chunking (can be improved)
        chunk_size = 1000
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        embeddings = [embedding_service.get_text_embedding(chunk) for chunk in chunks]
        ids = [f"{file_id}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]
        
        if chunks:
             vector_store.add_documents(documents=chunks, metadatas=metadatas, ids=ids, embeddings=embeddings)
        
        return UploadResponse(
            filename=filename,
            message="File uploaded and processed successfully",
            pages_processed=len(content.get("images", [])) # Approximate
        )

    except Exception as e:
        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")
