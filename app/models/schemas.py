from pydantic import BaseModel
from typing import List, Optional, Any, Dict
from dataclasses import dataclass
import numpy as np

class UploadResponse(BaseModel):
    filename: str
    message: str
    pages_processed: int

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[Any]

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
