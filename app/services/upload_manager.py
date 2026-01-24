"""
Upload Manager - Tracks upload status and manages rate limiting.

This module provides in-memory tracking for upload processing status
and implements IP-based rate limiting for file uploads.
"""

import time
import threading
import logging
from typing import Dict, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from app.models.schemas import ProcessingResult

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class UploadStatus:
    """Status information for an upload."""
    pdf_id: str
    filename: str
    status: str  # "processing", "completed", "failed"
    progress: float = 0.0
    result: Optional[ProcessingResult] = None
    error: Optional[str] = None
    started_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class UploadManager:
    """
    Manages upload status tracking and rate limiting.
    
    Features:
    - In-memory status tracking for uploads
    - IP-based rate limiting
    - Thread-safe operations
    - Automatic cleanup of old entries
    """
    
    def __init__(self, max_uploads_per_hour: int = 10, window_seconds: int = 3600):
        """
        Initialize the upload manager.
        
        Args:
            max_uploads_per_hour: Maximum uploads allowed per IP per hour
            window_seconds: Rate limiting window in seconds
        """
        self._statuses: Dict[str, UploadStatus] = {}
        self._rate_limits: Dict[str, list] = defaultdict(list)  # ip -> [timestamps]
        self._lock = threading.Lock()
        
        self.max_uploads = max_uploads_per_hour
        self.window_seconds = window_seconds
    
    def create_upload(self, pdf_id: str, filename: str) -> UploadStatus:
        """
        Create a new upload status entry.
        
        Args:
            pdf_id: Unique PDF identifier
            filename: Original filename
            
        Returns:
            Created UploadStatus
        """
        with self._lock:
            status = UploadStatus(
                pdf_id=pdf_id,
                filename=filename,
                status="processing",
                progress=0.0
            )
            self._statuses[pdf_id] = status
            return status
    
    def get_status(self, pdf_id: str) -> Optional[UploadStatus]:
        """
        Get upload status by PDF ID.
        
        Args:
            pdf_id: PDF identifier
            
        Returns:
            UploadStatus if found, None otherwise
        """
        with self._lock:
            return self._statuses.get(pdf_id)
    
    def update_progress(self, pdf_id: str, progress: float):
        """
        Update processing progress.
        
        Args:
            pdf_id: PDF identifier
            progress: Progress value (0.0 to 1.0)
        """
        with self._lock:
            if pdf_id in self._statuses:
                self._statuses[pdf_id].progress = min(max(progress, 0.0), 1.0)
                self._statuses[pdf_id].updated_at = datetime.now()
    
    def complete_upload(
        self,
        pdf_id: str,
        result: ProcessingResult,
        success: bool = True
    ):
        """
        Mark upload as completed.
        
        Args:
            pdf_id: PDF identifier
            result: Processing result
            success: Whether processing succeeded
        """
        with self._lock:
            if pdf_id in self._statuses:
                self._statuses[pdf_id].status = "completed" if success else "failed"
                self._statuses[pdf_id].progress = 1.0
                self._statuses[pdf_id].result = result
                self._statuses[pdf_id].updated_at = datetime.now()
    
    def fail_upload(self, pdf_id: str, error: str):
        """
        Mark upload as failed.
        
        Args:
            pdf_id: PDF identifier
            error: Error message
        """
        with self._lock:
            if pdf_id in self._statuses:
                self._statuses[pdf_id].status = "failed"
                self._statuses[pdf_id].error = error
                self._statuses[pdf_id].updated_at = datetime.now()
    
    def check_rate_limit(self, client_ip: str) -> Tuple[bool, int]:
        """
        Check if client has exceeded rate limit.
        
        Args:
            client_ip: Client IP address
            
        Returns:
            Tuple of (is_allowed, remaining_uploads)
        """
        current_time = time.time()
        
        with self._lock:
            # Clean up old timestamps
            cutoff_time = current_time - self.window_seconds
            self._rate_limits[client_ip] = [
                ts for ts in self._rate_limits[client_ip]
                if ts > cutoff_time
            ]
            
            # Check limit
            upload_count = len(self._rate_limits[client_ip])
            remaining = max(0, self.max_uploads - upload_count)
            is_allowed = upload_count < self.max_uploads
            
            return is_allowed, remaining
    
    def record_upload(self, client_ip: str):
        """
        Record an upload for rate limiting.
        
        Args:
            client_ip: Client IP address
        """
        with self._lock:
            self._rate_limits[client_ip].append(time.time())
    
    def cleanup_old_statuses(self, max_age_hours: int = 24):
        """
        Remove old status entries.
        
        Args:
            max_age_hours: Maximum age in hours to keep statuses
        """
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        
        with self._lock:
            to_remove = [
                pdf_id for pdf_id, status in self._statuses.items()
                if status.updated_at.timestamp() < cutoff_time
            ]
            
            for pdf_id in to_remove:
                del self._statuses[pdf_id]
    
    def get_all_statuses(self) -> Dict[str, UploadStatus]:
        """Get all upload statuses (for debugging)."""
        with self._lock:
            return dict(self._statuses)
    
    # Query Management Methods
    
    def check_query_rate_limit(self, client_ip: str, max_queries: int = 100) -> Tuple[bool, int]:
        """
        Check if client has exceeded query rate limit.
        
        Args:
            client_ip: Client IP address
            max_queries: Maximum queries allowed (default: 100)
            
        Returns:
            Tuple of (is_allowed, remaining_queries)
        """
        current_time = time.time()
        
        # Use a separate key for queries
        query_key = f"query_{client_ip}"
        
        with self._lock:
            # Clean up old timestamps
            cutoff_time = current_time - self.window_seconds
            self._rate_limits[query_key] = [
                ts for ts in self._rate_limits[query_key]
                if ts > cutoff_time
            ]
            
            # Check limit
            query_count = len(self._rate_limits[query_key])
            remaining = max(0, max_queries - query_count)
            is_allowed = query_count < max_queries
            
            return is_allowed, remaining
    
    def record_query(self, client_ip: str):
        """
        Record a query for rate limiting.
        
        Args:
            client_ip: Client IP address
        """
        query_key = f"query_{client_ip}"
        with self._lock:
            self._rate_limits[query_key].append(time.time())
    
    def log_query(self, client_ip: str, query: str, pdf_id: Optional[str], processing_time: float):
        """
        Log a query for analytics (basic in-memory logging).
        
        Args:
            client_ip: Client IP address
            query: Query string
            pdf_id: PDF ID if filtered
            processing_time: Processing time in seconds
        """
        # In production, this would log to a database or analytics service
        logger.info(
            f"QUERY | IP: {client_ip} | PDF: {pdf_id or 'all'} | "
            f"Time: {processing_time:.2f}s | Query: {query[:50]}..."
        )


# Singleton instance
_upload_manager_instance = None


def get_upload_manager() -> UploadManager:
    """
    Get or create singleton UploadManager instance.
    
    Returns:
        UploadManager instance
    """
    global _upload_manager_instance
    if _upload_manager_instance is None:
        from app.config import settings
        _upload_manager_instance = UploadManager(
            max_uploads_per_hour=settings.RATE_LIMIT_UPLOADS,
            window_seconds=settings.RATE_LIMIT_WINDOW
        )
    return _upload_manager_instance
