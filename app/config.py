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

    class Config:
        env_file = ".env"

settings = Settings()

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.EXTRACTED_DIR, exist_ok=True)
