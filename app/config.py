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

    class Config:
        env_file = ".env"

settings = Settings()

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.EXTRACTED_DIR, exist_ok=True)
