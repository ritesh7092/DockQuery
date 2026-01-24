"""
Query Routes - Enhanced RAG query endpoints with streaming and image serving.

This module provides:
- POST /query: Query the RAG system with optional streaming
- GET /images/{pdf_id}/{image_id}: Serve extracted images securely
"""

import os
import re
import time
import logging
import mimetypes
from pathlib import Path
from typing import Optional, AsyncIterator

from fastapi import (
    APIRouter,
    HTTPException,
    Request,
    Depends
)
from fastapi.responses import StreamingResponse, FileResponse
from sse_starlette.sse import EventSourceResponse

from app.config import settings
from app.models.schemas import QueryRequest, QueryResponse, VisualElement
from app.services.upload_manager import get_upload_manager, UploadManager
from app.services.rag_pipeline import get_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()


def get_client_ip(request: Request) -> str:
    """
    Extract client IP from request.
    
    Args:
        request: FastAPI request
        
    Returns:
        Client IP address
    """
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def sanitize_query(query: str) -> str:
    """
    Sanitize query input to prevent injection attacks.
    
    Args:
        query: Raw query string
        
    Returns:
        Sanitized query string
    """
    # Remove HTML tags
    query = re.sub(r'<[^>]+>', '', query)
    
    # Remove potentially malicious characters
    query = query.replace('\x00', '')  # Null byte
    query = query.replace('\r', ' ')   # Carriage return
    query = query.replace('\n', ' ')   # Newline (allow in sanitized form)
    
    # Trim and limit length
    query = query.strip()[:settings.MAX_QUERY_LENGTH]
    
    return query


def check_query_rate_limit(
    request: Request,
    upload_manager: UploadManager = Depends(get_upload_manager)
) -> UploadManager:
    """
    Dependency to check query rate limits.
    
    Args:
        request: FastAPI request
        upload_manager: Upload manager instance
        
    Returns:
        Upload manager instance
        
    Raises:
        HTTPException: If rate limit exceeded
    """
    client_ip = get_client_ip(request)
    is_allowed, remaining = upload_manager.check_query_rate_limit(
        client_ip,
        max_queries=settings.RATE_LIMIT_QUERIES
    )
    
    if not is_allowed:
        logger.warning(f"Query rate limit exceeded for IP: {client_ip}")
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "message": f"Maximum {settings.RATE_LIMIT_QUERIES} queries per hour allowed",
                "retry_after": settings.RATE_LIMIT_WINDOW
            }
        )
    
    logger.debug(f"Rate limit check passed for {client_ip}: {remaining} queries remaining")
    return upload_manager


async def stream_response(
    query: str,
    pdf_id: Optional[str],
    top_k: int,
    include_images: bool
) -> AsyncIterator[str]:
    """
    Stream query response using SSE.
    
    Args:
        query: User query
        pdf_id: Optional PDF ID filter
        top_k: Number of results
        include_images: Include visual elements
        
    Yields:
        SSE-formatted response chunks
    """
    try:
        pipeline = get_pipeline()
        
        # Get query result (non-streaming from pipeline)
        result = pipeline.query(
            query=query,
            pdf_id=pdf_id,
            top_k=top_k,
            include_visuals=include_images
        )
        
        # Stream the summary word by word for demo
        # In production, you'd integrate with LLM streaming
        words = result.summary.split()
        
        for i, word in enumerate(words):
            yield f"data: {word} \n\n"
            
        # Send final metadata
        yield f"data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield f"data: Error: {str(e)}\n\n"


@router.post("/query", response_model=QueryResponse)
async def query_rag(
    request_data: QueryRequest,
    request: Request,
    upload_manager: UploadManager = Depends(check_query_rate_limit)
):
    """
    Query the RAG system with optional PDF filtering and streaming.
    
    Features:
    - Filter by PDF ID
    - Configurable top_k results
    - Include/exclude visual elements
    - Streaming support with SSE
    - Rate limiting (100 queries/hour)
    - Response caching
    - Query logging
    
    Args:
        request_data: Query request parameters
        request: Request object for metadata
        upload_manager: Dependency-injected upload manager
        
    Returns:
        QueryResponse with summary, sources, and visual elements
        OR StreamingResponse if stream=True
        
    Raises:
        HTTPException: On validation errors or processing failures
    """
    start_time = time.time()
    client_ip = get_client_ip(request)
    
    # Sanitize query
    sanitized_query = sanitize_query(request_data.query)
    
    if not sanitized_query:
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty after sanitization"
        )
    
    logger.info(f"Query from {client_ip}: {sanitized_query[:50]}... (pdf_id={request_data.pdf_id})")
    
    # Validate PDF ID if provided
    if request_data.pdf_id:
        status = upload_manager.get_status(request_data.pdf_id)
        if not status:
            raise HTTPException(
                status_code=404,
                detail=f"PDF ID '{request_data.pdf_id}' not found"
            )
        if status.status != "completed":
            raise HTTPException(
                status_code=400,
                detail=f"PDF is still {status.status}. Please wait for processing to complete."
            )
    
    # Handle streaming response
    if request_data.stream:
        logger.info("Streaming response enabled")
        return EventSourceResponse(
            stream_response(
                sanitized_query,
                request_data.pdf_id,
                request_data.top_k,
                request_data.include_images
            )
        )
    
    # Process query through RAG pipeline
    try:
        pipeline = get_pipeline()
        
        result = pipeline.query(
            query=sanitized_query,
            pdf_id=request_data.pdf_id,
            top_k=request_data.top_k,
            include_visuals=request_data.include_images
        )
        
        # Build visual elements list
        visual_elements = []
        if request_data.include_images:
            for source in result.sources:
                if source.type in ["image", "table", "chart"] and source.image_path:
                    # Extract image_id from path
                    image_path = Path(source.image_path)
                    image_id = image_path.stem  # filename without extension
                    
                    visual_elem = VisualElement(
                        type=source.type,
                        page=source.page,
                        image_id=image_id,
                        pdf_id=request_data.pdf_id or "all",
                        url=f"/api/v1/images/{request_data.pdf_id or 'all'}/{image_id}",
                        description=source.content_preview
                    )
                    visual_elements.append(visual_elem)
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        # Record query for rate limiting and logging
        upload_manager.record_query(client_ip)
        upload_manager.log_query(
            client_ip,
            sanitized_query,
            request_data.pdf_id,
            processing_time_ms / 1000
        )
        
        return QueryResponse(
            query=sanitized_query,
            summary=result.summary,
            sources=result.sources,
            visual_elements=visual_elements,
            processing_time_ms=processing_time_ms,
            cached=result.cached
        )
        
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@router.get("/images/{pdf_id}/{image_id}")
async def get_image(
    pdf_id: str,
    image_id: str,
    upload_manager: UploadManager = Depends(get_upload_manager)
):
    """
    Serve extracted images securely.
    
    Security features:
    - Validates PDF ID exists
    - Validates image belongs to PDF
    - Prevents path traversal
    - Checks file existence
    
    Args:
        pdf_id: PDF identifier
        image_id: Image identifier
        upload_manager: Dependency-injected upload manager
        
    Returns:
        FileResponse with image
        
    Raises:
        HTTPException: If PDF not found, image not found, or access denied
    """
    # Validate PDF ID
    if pdf_id != "all":  # "all" is used when querying across all PDFs
        status = upload_manager.get_status(pdf_id)
        if not status:
            raise HTTPException(
                status_code=404,
                detail=f"PDF ID '{pdf_id}' not found"
            )
    
    # Sanitize image_id to prevent path traversal
    image_id = os.path.basename(image_id)
    if '..' in image_id or '/' in image_id or '\\' in image_id:
        raise HTTPException(
            status_code=400,
            detail="Invalid image ID"
        )
    
    # Search for image in extracted directory
    # Images are stored as: {pdf_id}_image_{index}.{ext} or similar patterns
    extracted_dir = Path(settings.EXTRACTED_DIR)
    
    # Try common image extensions
    for ext in ['.png', '.jpg', '.jpeg', '.webp']:
        # Try different naming patterns
        possible_paths = [
            extracted_dir / f"{image_id}{ext}",
            extracted_dir / f"{pdf_id}_{image_id}{ext}",
            extracted_dir / f"{image_id}",  # If it already has extension
        ]
        
        for image_path in possible_paths:
            if image_path.exists() and image_path.is_file():
                # Verify the image belongs to the requested PDF or is accessible
                if pdf_id == "all" or pdf_id in str(image_path):
                    # Get MIME type
                    mime_type, _ = mimetypes.guess_type(str(image_path))
                    mime_type = mime_type or "image/png"
                    
                    logger.info(f"Serving image: {image_path}")
                    return FileResponse(
                        path=str(image_path),
                        media_type=mime_type,
                        filename=image_path.name
                    )
    
    # Image not found
    logger.warning(f"Image not found: {pdf_id}/{image_id}")
    raise HTTPException(
        status_code=404,
        detail=f"Image '{image_id}' not found for PDF '{pdf_id}'"
    )
