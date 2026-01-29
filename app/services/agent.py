"""
GeminiAgent - Production-ready multimodal RAG agent using Google's Gemini API.

This module provides a comprehensive agent for generating summaries from multimodal
context (text + images) retrieved from PDF documents.
"""

import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, Iterator, Union
from pathlib import Path
import re

import google.generativeai as genai
from google.generativeai.types import GenerateContentResponse
from google.api_core import exceptions as google_exceptions
from PIL import Image

from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeminiAgent:
    """
    Production-ready multimodal RAG agent using Gemini API.
    
    Supports:
    - Text and image inputs
    - Streaming responses
    - Async operations
    - Retry logic with exponential backoff
    - Comprehensive error handling
    """
    
    def __init__(self, api_key: str, model: str = None):
        """
        Initialize the GeminiAgent.
        
        Args:
            api_key: Google API key for Gemini
            model: Model name (default: from settings.GEMINI_MODEL)
        """
        self.api_key = api_key
        self.model_name = model or settings.GEMINI_MODEL
        
        # Configure the API
        genai.configure(api_key=self.api_key)
        
        # Initialize the model
        self.model = genai.GenerativeModel(self.model_name)
        
        # Configuration
        self.max_retries = settings.MAX_RETRIES
        self.initial_delay = settings.INITIAL_RETRY_DELAY
        self.timeout = settings.REQUEST_TIMEOUT
        
        logger.info(f"GeminiAgent initialized with model: {self.model_name}")
    
    def generate_summary(
        self,
        query: str,
        text_context: List[str],
        visual_context: Optional[List[Path]] = None,
        stream: bool = False
    ) -> Union[str, Iterator[str]]:
        """
        Generate a comprehensive summary based on query and context.
        
        Args:
            query: User's query
            text_context: List of text chunks from retrieved documents
            visual_context: List of paths to images (charts, tables, diagrams)
            stream: Whether to stream the response
            
        Returns:
            Complete summary string or iterator of text chunks if streaming
        """
        try:
            # Encode images if provided
            images = []
            has_visuals = False
            if visual_context and len(visual_context) > 0:
                try:
                    images = self._encode_images(visual_context)
                    has_visuals = len(images) > 0
                    logger.info(f"Successfully encoded {len(images)} visual elements")
                except Exception as e:
                    logger.warning(f"Image encoding failed, falling back to text-only: {e}")
                    # Continue with text-only if image encoding fails
            
            # Build the prompt with visual context info
            prompt = self._build_prompt(query, text_context, has_visuals=has_visuals)
            
            # Prepare content for generation
            if images:
                # Multimodal: interleave prompt and images
                content = [prompt] + images
                logger.debug("Generating multimodal response")
            else:
                # Text-only
                content = prompt
                logger.debug("Generating text-only response")
            
            # Generate with retry logic
            if stream:
                return self._retry_with_backoff(
                    lambda: self._generate_streaming(content)
                )
            else:
                response = self._retry_with_backoff(
                    lambda: self.model.generate_content(content)
                )
                return response.text
                
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"Error generating summary: {str(e)}"
    
    async def generate_summary_async(
        self,
        query: str,
        text_context: List[str],
        visual_context: Optional[List[Path]] = None,
        stream: bool = False
    ) -> Union[str, Iterator[str]]:
        """
        Async version of generate_summary.
        
        Args:
            query: User's query
            text_context: List of text chunks from retrieved documents
            visual_context: List of paths to images
            stream: Whether to stream the response
            
        Returns:
            Complete summary string or async iterator of chunks if streaming
        """
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.generate_summary,
            query,
            text_context,
            visual_context,
            stream
        )
    
    def _build_prompt(
        self,
        query: str,
        text_context: List[str],
        has_visuals: bool = False
    ) -> str:
        """
        Build a sophisticated prompt for accurate summarization.
        
        Args:
            query: User's query
            text_context: List of text chunks
            has_visuals: Whether visual elements (images/charts/tables) are provided
            
        Returns:
            Formatted prompt string
        """
        # Extract page references from context
        page_refs = self._extract_page_references(text_context)
        
        # Combine text context - properly handle empty list vs None
        if text_context and len(text_context) > 0:
            combined_context = "\n\n".join(text_context)
        else:
            combined_context = "No text context available."
        
        # Build visual elements section conditionally
        visual_section = ""
        if has_visuals:
            visual_section = """
Visual Elements:
The visual elements (charts, tables, diagrams) are provided alongside this text. Please analyze them in context.
"""
        
        # Build task instructions based on available data
        task_instructions = ["1. Directly answers the query"]
        
        if has_visuals:
            task_instructions.append("2. References specific data from tables/charts when relevant")
            task_instructions.append(f"3. Maintains accuracy and cites page numbers when available{f' (available pages: {', '.join(map(str, page_refs))})' if page_refs else ''}")
            task_instructions.append("4. Explains visual elements in context")
            task_instructions.append("5. Synthesizes information from both text and visual sources")
        else:
            task_instructions.append(f"2. Maintains accuracy and cites page numbers when available{f' (available pages: {', '.join(map(str, page_refs))})' if page_refs else ''}")
            task_instructions.append("3. Provides a clear and comprehensive response based on the text context")
        
        # Add formatting instructions
        task_instructions.append("6. FORMATTING: Use Markdown. **Bold** key entities and metrics. Use bullet points for lists. Use headers for sections.")
        
        # Build structured prompt
        prompt = f"""You are an expert analyst. Based on the following context extracted from a PDF document, provide a comprehensive summary addressing the user's query.

Query: {query}

Text Context:
{combined_context}{visual_section}

Provide a detailed summary that:
{chr(10).join(task_instructions)}

Summary:"""
        
        return prompt
    
    def _encode_images(self, image_paths: List[Path]) -> List[Image.Image]:
        """
        Encode images for Gemini API.
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            List of PIL Image objects
        """
        images = []
        for path in image_paths:
            try:
                if not path.exists():
                    logger.warning(f"Image not found: {path}")
                    continue
                
                # Open and validate image
                img = Image.open(path)
                
                # Convert to RGB if necessary (Gemini prefers RGB)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                images.append(img)
                logger.debug(f"Encoded image: {path}")
                
            except Exception as e:
                logger.warning(f"Failed to encode image {path}: {e}")
                continue
        
        return images
    
    def _retry_with_backoff(
        self,
        func: callable,
        max_retries: Optional[int] = None,
        initial_delay: Optional[float] = None
    ) -> Any:
        """
        Execute a function with exponential backoff retry logic.
        
        Args:
            func: Function to execute
            max_retries: Maximum number of retries (default: from settings)
            initial_delay: Initial delay in seconds (default: from settings)
            
        Returns:
            Result from the function
            
        Raises:
            Exception: If all retries are exhausted
        """
        max_retries = max_retries or self.max_retries
        delay = initial_delay or self.initial_delay
        
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return func()
                
            except google_exceptions.ResourceExhausted as e:
                # Rate limiting (429)
                last_exception = e
                if attempt < max_retries:
                    sleep_time = delay * (2 ** attempt)
                    logger.warning(f"Rate limited. Retrying in {sleep_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(sleep_time)
                else:
                    logger.error("Rate limit retries exhausted")
                    raise
                    
            except google_exceptions.InvalidArgument as e:
                # Invalid input (400) - don't retry
                logger.error(f"Invalid argument: {e}")
                raise
                
            except google_exceptions.InternalServerError as e:
                # Server error (500) - retry
                last_exception = e
                if attempt < max_retries:
                    sleep_time = delay * (2 ** attempt)
                    logger.warning(f"Server error. Retrying in {sleep_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(sleep_time)
                else:
                    logger.error("Server error retries exhausted")
                    raise
                    
            except Exception as e:
                # Other errors - retry with caution
                last_exception = e
                if attempt < max_retries:
                    sleep_time = delay * (2 ** attempt)
                    logger.warning(f"Error occurred: {e}. Retrying in {sleep_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"All retries exhausted: {e}")
                    raise
        
        # Should not reach here, but just in case
        raise last_exception or Exception("Unknown error in retry logic")
    
    def _generate_streaming(self, content: Any) -> Iterator[str]:
        """
        Generate streaming response.
        
        Args:
            content: Content to send to the model
            
        Yields:
            Text chunks from the response
        """
        try:
            response = self.model.generate_content(content, stream=True)
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"Error during streaming: {str(e)}"
    
    def _extract_page_references(self, text_context: List[str]) -> List[int]:
        """
        Extract page numbers from text context.
        
        Args:
            text_context: List of text chunks
            
        Returns:
            List of unique page numbers found in context
        """
        page_numbers = set()
        
        # Handle None or empty list
        if not text_context:
            return []
        
        for text in text_context:
            # Look for common page reference patterns
            # Pattern 1: "Page 5", "page 5"
            matches = re.findall(r'\bpage\s+(\d+)\b', text, re.IGNORECASE)
            page_numbers.update(int(m) for m in matches)
            
            # Pattern 2: "p. 5", "p.5"
            matches = re.findall(r'\bp\.?\s*(\d+)\b', text, re.IGNORECASE)
            page_numbers.update(int(m) for m in matches)
        
        return sorted(list(page_numbers))


# Singleton instance for easy import
gemini_agent = None

def get_agent() -> GeminiAgent:
    """
    Get or create a singleton GeminiAgent instance.
    
    Returns:
        GeminiAgent instance
    """
    global gemini_agent
    if gemini_agent is None:
        if not settings.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not configured in settings")
        gemini_agent = GeminiAgent(api_key=settings.GOOGLE_API_KEY)
    return gemini_agent
