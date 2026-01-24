"""
Production-ready FastAPI Application.

This module configures the FastAPI application with:
- CORS middleware
- Request logging and timing
- Global exception handlers
- Health check endpoints
- Startup and shutdown events
"""

import logging
import time
import uuid
import os
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import settings
from app.api.routes import upload, query

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handle startup and shutdown events.
    
    Startup:
    - Initialize vector DB connection
    - Load embedding models  
    - Verify API keys
    - Create necessary directories
    
    Shutdown:
    - Cleanup temporary files
    - Close connections
    """
    # Startup
    logger.info("🚀 Starting Multimodal RAG API...")
    
    try:
        # Create necessary directories
        for directory in [settings.UPLOAD_DIR, settings.EXTRACTED_DIR]:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"✓ Directory ready: {directory}")
        
        # Verify Gemini API key
        if not settings.GOOGLE_API_KEY:
            logger.warning("⚠️  GOOGLE_API_KEY not set! Agent features will be disabled.")
        else:
            logger.info("✓ Gemini API key configured")
        
        # Initialize vector store (lazy loading on first use)
        logger.info("✓ Vector store: ChromaDB (lazy init)")
        
        # Initialize embedding service (lazy loading)
        logger.info("✓ Embedding service: Sentence Transformers (lazy init)")
        
        logger.info("✅ Application startup complete!")
        
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Multimodal RAG API...")
    
    try:
        # Cleanup temporary files
        cleanup_temp_files()
        logger.info("✓ Temporary files cleaned")
        
        # Close connections (if any)
        logger.info("✓ Connections closed")
        
        logger.info("👋 Shutdown complete")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


def cleanup_temp_files():
    """Clean up temporary uploaded files."""
    try:
        upload_dir = settings.UPLOAD_DIR
        if os.path.exists(upload_dir):
            for filename in os.listdir(upload_dir):
                file_path = os.path.join(upload_dir, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    logger.warning(f"Could not delete {file_path}: {e}")
    except Exception as e:
        logger.error(f"Error cleaning temp files: {e}")


# Create FastAPI application
app = FastAPI(
    title="Multimodal RAG API",
    description="Production-ready API for multimodal RAG with PDF processing, vector search, and LLM integration",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)


# CORS Configuration
origins = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
]

# Add environment-specific origins if configured
if os.getenv("ALLOWED_ORIGINS"):
    origins.extend(os.getenv("ALLOWED_ORIGINS").split(","))

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request ID and Timing Middleware
@app.middleware("http")
async def add_request_id_and_timing(request: Request, call_next):
    """
    Add request ID to all requests and track processing time.
    
    Headers added:
    - X-Request-ID: Unique request identifier
    - X-Process-Time: Processing time in seconds
    """
    # Generate request ID
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    # Track request start time
    start_time = time.time()
    
    # Log request
    logger.info(
        f"Request {request_id}: {request.method} {request.url.path} "
        f"from {request.client.host if request.client else 'unknown'}"
    )
    
    # Process request
    try:
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Add headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{process_time:.4f}"
        
        # Log response
        logger.info(
            f"Response {request_id}: {response.status_code} "
            f"({process_time:.4f}s)"
        )
        
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            f"Error {request_id}: {str(e)} ({process_time:.4f}s)",
            exc_info=True
        )
        raise


# Global Exception Handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions."""
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    logger.warning(
        f"HTTP {exc.status_code} on {request.method} {request.url.path}: {exc.detail}"
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "request_id": request_id
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    logger.warning(
        f"Validation error on {request.method} {request.url.path}: {exc.errors()}"
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": exc.errors(),
            "request_id": request_id
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions."""
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    logger.error(
        f"Unhandled exception on {request.method} {request.url.path}: {exc}",
        exc_info=True
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "request_id": request_id
        }
    )


# Include API routers
app.include_router(
    upload.router,
    prefix="/api/v1",
    tags=["upload"]
)

app.include_router(
    query.router,
    prefix="/api/v1",
    tags=["query"]
)


# Health Check Endpoints
@app.get("/health", tags=["health"])
async def health_check() -> Dict[str, Any]:
    """
    Basic health check endpoint.
    
    Returns:
        Health status and version info
    """
    return {
        "status": "healthy",
        "service": "Multimodal RAG API",
        "version": "1.0.0"
    }


@app.get("/api/v1/metrics", tags=["health"])
async def system_metrics() -> Dict[str, Any]:
    """
    System metrics endpoint.
    
    Returns:
        System metrics including:
        - Service status
        - Directory status
        - Configuration info
    """
    from app.services.rag_pipeline import get_pipeline
    
    try:
        # Get pipeline metrics
        pipeline = get_pipeline()
        pipeline_metrics = pipeline.get_metrics()
    except Exception as e:
        logger.warning(f"Could not get pipeline metrics: {e}")
        pipeline_metrics = {"error": str(e)}
    
    return {
        "status": "operational",
        "service": {
            "name": "Multimodal RAG API",
            "version": "1.0.0",
            "environment": os.getenv("ENVIRONMENT", "development")
        },
        "directories": {
            "upload_dir": settings.UPLOAD_DIR,
            "extracted_dir": settings.EXTRACTED_DIR,
            "chroma_db_path": settings.CHROMA_DB_PATH
        },
        "configuration": {
            "embedding_model": settings.EMBEDDING_MODEL,
            "clip_model": settings.CLIP_MODEL,
            "gemini_model": settings.GEMINI_MODEL,
            "device": settings.DEVICE
        },
        "rate_limits": {
            "uploads_per_hour": settings.RATE_LIMIT_UPLOADS,
            "queries_per_hour": settings.RATE_LIMIT_QUERIES
        },
        "pipeline_metrics": pipeline_metrics
    }


@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Multimodal RAG API",
        "version": "1.0.0",
        "docs": "/api/docs",
        "health": "/health",
        "metrics": "/api/v1/metrics"
    }
