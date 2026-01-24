from pydantic import BaseModel
from typing import List, Optional, Any, Dict
from dataclasses import dataclass
import numpy as np

class UploadResponse(BaseModel):
    """Response for successful file upload."""
    pdf_id: str
    filename: str
    status: str  # "processing", "queued"
    message: str

class UploadStatusResponse(BaseModel):
    """Response for upload status query."""
    pdf_id: str
    status: str  # "processing", "completed", "failed"
    progress: float  # 0.0 to 1.0
    result: Optional['ProcessingResult'] = None
    error: Optional[str] = None

class QueryRequest(BaseModel):
    """Enhanced query request with filtering and streaming options."""
    query: str
    pdf_id: Optional[str] = None  # Filter by specific PDF
    top_k: int = 5  # Number of results to retrieve
    include_images: bool = True  # Include visual elements
    stream: bool = False  # Stream response with SSE

class VisualElement(BaseModel):
    """Visual element reference in query response."""
    type: str  # "image", "table", "chart"
    page: int
    image_id: str  # Unique identifier for image retrieval
    pdf_id: str  # PDF this image belongs to
    url: str  # Endpoint to retrieve the image
    description: str  # Brief description of the visual

# RAG Pipeline Models (moved here to be available for QueryResponse)
class SourceReference(BaseModel):
    """Source reference with attribution details."""
    type: str  # "text", "image", "table", "chart"
    page: int
    content_preview: str
    image_path: Optional[str] = None
    confidence: float

class QueryResponse(BaseModel):
    """Enhanced query response with visual elements and timing."""
    query: str
    summary: str
    sources: List[SourceReference]
    visual_elements: List[VisualElement] = []
    processing_time_ms: float
    cached: bool = False

# Vector Store Models
@dataclass
class DocumentMetadata:
    """Metadata for a document stored in the vector store."""
    type: str  # "text", "image", "table", "chart"
    page: int
    source: str  # Source PDF filename
    image_path: Optional[str] = None
    bbox: Optional[Dict[str, float]] = None  # Bounding box coordinates

@dataclass
class Document:
    """Document representation for vector store operations."""
    id: str
    content: str  # Text content or image description
    metadata: DocumentMetadata
    embedding: Optional[np.ndarray] = None

class ProcessingResult(BaseModel):
    """Result from PDF processing pipeline."""
    pdf_id: str
    total_pages: int
    text_chunks: int
    images_extracted: int
    tables_found: int
    charts_found: int
    processing_time: float
    status: str  # "success", "partial", "failed"
    errors: Optional[List[str]] = None

class QueryResult(BaseModel):
    """Result from RAG query pipeline."""
    query: str
    summary: str
    sources: List[SourceReference]
    confidence: float
    processing_time: float
    cached: bool = False
