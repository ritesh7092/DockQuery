from fastapi import FastAPI
from app.api.routes import upload, query

app = FastAPI(title="Multimodal RAG API", version="0.1.0")

app.include_router(upload.router, prefix="/api/v1", tags=["upload"])
app.include_router(query.router, prefix="/api/v1", tags=["query"])

@app.get("/")
def root():
    return {"message": "Multimodal RAG API is running"}
