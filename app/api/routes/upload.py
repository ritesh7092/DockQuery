"""
Upload Routes - Secure PDF upload endpoints with validation and async processing.

This module provides:
- POST /upload: Upload and process PDF files
- GET /upload/{pdf_id}/status: Check processing status
"""

import os
import re
import shutil
import uuid
import logging
from pathlib import Path
from typing import Optional

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
    BackgroundTasks,
    Request,
    Depends
)

from app.config import settings
from app.models.schemas import UploadResponse, UploadStatusResponse
from app.services.upload_manager import get_upload_manager, UploadManager
from app.services.rag_pipeline import get_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal attacks.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename (alphanumeric, hyphens, underscores, dots only)
    """
    # Remove any path components
    filename = os.path.basename(filename)
    
    # Replace spaces with underscores
    filename = filename.replace(" ", "_")
    
    # Remove any characters that aren't alphanumeric, hyphen, underscore, or dot
    filename = re.sub(r'[^a-zA-Z0-9._-]', '', filename)
    
    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:250] + ext
    
    return filename or "upload.pdf"


def validate_pdf_magic_bytes(file_path: str) -> bool:
    """
    Validate that file is a real PDF by checking magic bytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        True if valid PDF, False otherwise
    """
    try:
        with open(file_path, 'rb') as f:
            header = f.read(5)
            return header == settings.PDF_MAGIC_BYTES
    except Exception as e:
        logger.error(f"Error reading file magic bytes: {e}")
        return False


def get_client_ip(request: Request) -> str:
    """
    Extract client IP from request.
    
    Args:
        request: FastAPI request
        
    Returns:
        Client IP address
    """
    # Check for forwarded IP (if behind proxy)
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    
    # Fallback to direct client
    return request.client.host if request.client else "unknown"


def check_rate_limit(
    request: Request,
    upload_manager: UploadManager = Depends(get_upload_manager)
) -> UploadManager:
    """
    Dependency to check rate limits.
    
    Args:
        request: FastAPI request
        upload_manager: Upload manager instance
        
    Returns:
        Upload manager instance
        
    Raises:
        HTTPException: If rate limit exceeded
    """
    client_ip = get_client_ip(request)
    is_allowed, remaining = upload_manager.check_rate_limit(client_ip)
    
    if not is_allowed:
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "message": f"Maximum {settings.RATE_LIMIT_UPLOADS} uploads per hour allowed",
                "retry_after": settings.RATE_LIMIT_WINDOW
            }
        )
    
    logger.info(f"Rate limit check passed for {client_ip}: {remaining} uploads remaining")
    return upload_manager


async def process_pdf_background(
    pdf_id: str,
    file_path: str,
    filename: str,
    upload_manager: UploadManager
):
    """
    Background task to process PDF through RAG pipeline.
    
    Args:
        pdf_id: Unique PDF identifier
        file_path: Path to uploaded file
        filename: Original filename
        upload_manager: Upload manager for status updates
    """
    try:
        logger.info(f"Starting background processing for {pdf_id}")
        
        # Update progress: Starting
        upload_manager.update_progress(pdf_id, 0.1)
        
        # Get pipeline instance
        pipeline = get_pipeline()
        
        # Update progress: Parsing
        upload_manager.update_progress(pdf_id, 0.3)
        
        # Process PDF
        result = pipeline.process_pdf(
            file_path=file_path,
            pdf_id=pdf_id,
            collection_name="multimodal_rag"
        )
        
        # Update progress: Completed
        upload_manager.update_progress(pdf_id, 0.9)
        
        # Mark as complete
        success = result.status in ["success", "partial"]
        upload_manager.complete_upload(pdf_id, result, success=success)
        
        logger.info(f"Background processing completed for {pdf_id}: {result.status}")
        
    except Exception as e:
        logger.error(f"Background processing failed for {pdf_id}: {e}")
        upload_manager.fail_upload(pdf_id, str(e))
        
    finally:
        # Cleanup: Remove uploaded file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to cleanup file {file_path}: {e}")


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    request: Request,
    file: UploadFile = File(...),
    upload_manager: UploadManager = Depends(check_rate_limit)
):
    """
    Upload and process a PDF file.
    
    Security features:
    - File size validation (max 50MB)
    - MIME type validation
    - PDF magic bytes verification
    - Filename sanitization
    - Rate limiting (10 uploads/hour per IP)
    
    Processing:
    - Asynchronous background processing
    - Progress tracking
    - Automatic file cleanup
    
    Args:
        background_tasks: FastAPI background tasks
        request: Request object for rate limiting
        file: Uploaded PDF file
        upload_manager: Dependency-injected upload manager
        
    Returns:
        UploadResponse with pdf_id and status
        
    Raises:
        HTTPException: On validation errors or processing failures
    """
    client_ip = get_client_ip(request)
    
    # Validate MIME type
    if file.content_type not in settings.ALLOWED_MIME_TYPES:
        logger.warning(f"Invalid MIME type from {client_ip}: {file.content_type}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Only PDF files are allowed. Received: {file.content_type}"
        )
    
    # Generate unique PDF ID
    pdf_id = str(uuid.uuid4())
    
    # Sanitize filename
    safe_filename = sanitize_filename(file.filename or "upload.pdf")
    stored_filename = f"{pdf_id}_{safe_filename}"
    file_path = os.path.join(settings.UPLOAD_DIR, stored_filename)
    
    logger.info(f"Processing upload from {client_ip}: {safe_filename} -> {pdf_id}")
    
    # Save file temporarily
    try:
        # Check file size while saving
        total_size = 0
        with open(file_path, "wb") as buffer:
            while chunk := await file.read(8192):  # Read in 8KB chunks
                total_size += len(chunk)
                
                # Check size limit
                if total_size > settings.MAX_UPLOAD_SIZE:
                    buffer.close()
                    os.remove(file_path)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size is {settings.MAX_UPLOAD_SIZE / (1024*1024):.0f}MB"
                    )
                
                buffer.write(chunk)
        
        logger.info(f"Saved file: {stored_filename} ({total_size} bytes)")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save file: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=500,
            detail=f"Could not save file: {str(e)}"
        )
    
    # Validate PDF magic bytes
    if not validate_pdf_magic_bytes(file_path):
        os.remove(file_path)
        logger.warning(f"Invalid PDF magic bytes from {client_ip}")
        raise HTTPException(
            status_code=400,
            detail="Invalid PDF file. File does not contain valid PDF header."
        )
    
    # Record upload for rate limiting
    upload_manager.record_upload(client_ip)
    
    # Create upload status entry
    upload_manager.create_upload(pdf_id, safe_filename)
    
    # Queue background processing
    background_tasks.add_task(
        process_pdf_background,
        pdf_id=pdf_id,
        file_path=file_path,
        filename=safe_filename,
        upload_manager=upload_manager
    )
    
    logger.info(f"Upload queued for processing: {pdf_id}")
    
    return UploadResponse(
        pdf_id=pdf_id,
        filename=safe_filename,
        status="processing",
        message="File uploaded successfully and processing has started"
    )


@router.get("/upload/{pdf_id}/status", response_model=UploadStatusResponse)
async def get_upload_status(
    pdf_id: str,
    upload_manager: UploadManager = Depends(get_upload_manager)
):
    """
    Get the processing status of an uploaded PDF.
    
    Args:
        pdf_id: Unique PDF identifier
        upload_manager: Dependency-injected upload manager
        
    Returns:
        UploadStatusResponse with current status and progress
        
    Raises:
        HTTPException: If PDF ID not found
    """
    status = upload_manager.get_status(pdf_id)
    
    if not status:
        raise HTTPException(
            status_code=404,
            detail=f"Upload with ID '{pdf_id}' not found"
        )
    
    return UploadStatusResponse(
        pdf_id=status.pdf_id,
        status=status.status,
        progress=status.progress,
        result=status.result,
        error=status.error
    )
