from pydantic import BaseModel
from typing import List, Optional, Any

class UploadResponse(BaseModel):
    filename: str
    message: str
    pages_processed: int

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[Any]
