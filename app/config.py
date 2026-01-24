import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GOOGLE_API_KEY: str
    CHROMA_DB_PATH: str = "./data/chroma_db"
    UPLOAD_DIR: str = "./data/uploads"
    EXTRACTED_DIR: str = "./data/extracted"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CLIP_MODEL: str = "clip-ViT-B-32"
    DEVICE: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    EMBEDDING_BATCH_SIZE: int = 32
    
    # Gemini Agent Configuration
    GEMINI_MODEL: str = "gemini-1.5-flash"
    MAX_RETRIES: int = 3
    INITIAL_RETRY_DELAY: float = 1.0
    REQUEST_TIMEOUT: int = 30
    
    # RAG Pipeline Configuration
    CACHE_TTL_SECONDS: int = 3600
    CACHE_MAX_SIZE: int = 100
    RERANK_TOP_K: int = 20
    MIN_CONFIDENCE_THRESHOLD: float = 0.3
    
    # Upload Configuration
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024  # 50MB in bytes
    ALLOWED_MIME_TYPES: list = ["application/pdf"]
    PDF_MAGIC_BYTES: bytes = b"%PDF-"
    
    # Rate Limiting
    RATE_LIMIT_UPLOADS: int = 10  # uploads per window
    RATE_LIMIT_WINDOW: int = 3600  # 1 hour in seconds
    RATE_LIMIT_QUERIES: int = 100  # queries per window
    
    # Query Configuration
    MAX_QUERY_LENGTH: int = 1000  # Maximum query string length
    QUERY_CACHE_TTL: int = 1800  # 30 minutes

    class Config:
        env_file = ".env"

settings = Settings()

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.EXTRACTED_DIR, exist_ok=True)
