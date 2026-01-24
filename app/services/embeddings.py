import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union, Optional
from pathlib import Path
import functools

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
import warnings

# Suppress warnings from huggingface/transformers if needed
warnings.filterwarnings("ignore", category=FutureWarning)

from app.config import settings

# Configure logging
logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        self.device = settings.DEVICE
        self.text_model_name = settings.EMBEDDING_MODEL
        self.vision_model_name = settings.CLIP_MODEL
        self.batch_size = settings.EMBEDDING_BATCH_SIZE
        
        self._text_model: Optional[SentenceTransformer] = None
        self._vision_model: Optional[SentenceTransformer] = None
        
        # Simple in-memory cache for embeddings to avoid re-computation
        # Key: str (text) or str (image_path) -> Value: np.ndarray
        # In a production environment, use Redis or similar
        self._cache = {}
        
        # Executor for async operations
        self._executor = ThreadPoolExecutor(max_workers=3)

    @property
    def text_model(self) -> SentenceTransformer:
        if self._text_model is None:
            self._text_model = self._load_model_with_retry(self.text_model_name)
        return self._text_model

    @property
    def vision_model(self) -> SentenceTransformer:
        if self._vision_model is None:
            self._vision_model = self._load_model_with_retry(self.vision_model_name)
        return self._vision_model

    def _load_model_with_retry(self, model_name: str, max_retries: int = 3) -> SentenceTransformer:
        """Loads a model with exponential backoff on failure."""
        attempt = 0
        backoff = 1
        while attempt < max_retries:
            try:
                logger.info(f"Loading model {model_name} on {self.device}...")
                model = SentenceTransformer(model_name, device=self.device)
                return model
            except Exception as e:
                attempt += 1
                logger.error(f"Failed to load model {model_name} (Attempt {attempt}/{max_retries}): {e}")
                if attempt == max_retries:
                    raise RuntimeError(f"Could not load model {model_name} after {max_retries} attempts.") from e
                time.sleep(backoff)
                backoff *= 2
        raise RuntimeError("Unreachable code reached.")

    def embed_text(self, texts: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """Generates embeddings for a single text or a list of texts."""
        if isinstance(texts, str):
            texts = [texts]
            single = True
        else:
            single = False

        # Check cache for individual items? 
        # For batching efficiency, we might want to process uncached items in a batch.
        # However, typically sentence-transformers handles batching internally well.
        # Let's check cache first.
        
        uncached_texts = []
        uncached_indices = []
        results = [None] * len(texts)

        for i, text in enumerate(texts):
            cache_key = f"text:{text}"
            if cache_key in self._cache:
                results[i] = self._cache[cache_key]
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        if uncached_texts:
            try:
                embeddings = self.text_model.encode(
                    uncached_texts, 
                    batch_size=self.batch_size, 
                    show_progress_bar=False, 
                    convert_to_numpy=True,
                    normalize_embeddings=True 
                )
                
                for idx, embedding in zip(uncached_indices, embeddings):
                    results[idx] = embedding
                    # Cache the result
                    self._cache[f"text:{texts[idx]}"] = embedding
            except Exception as e:
                logger.error(f"Error embedding text: {e}")
                raise

        if single:
            return results[0]  # type: ignore
        return results # type: ignore

    def embed_image(self, image_input: Union[str, Path, Image.Image]) -> np.ndarray:
        """Generates embedding for a single image."""
        # Convert path to string for caching key
        cache_key = None
        image_obj = None

        if isinstance(image_input, (str, Path)):
             cache_key = f"image:{str(image_input)}"
             if cache_key in self._cache:
                 return self._cache[cache_key]
             try:
                image_obj = Image.open(image_input)
             except Exception as e:
                 logger.error(f"Failed to open image {image_input}: {e}")
                 raise
        elif isinstance(image_input, Image.Image):
             # Cannot reliably cache raw image objects without hashing, skip cache for raw objects for now
             # or hash them if needed. For now, just process.
             image_obj = image_input
        else:
             raise ValueError("Unsupported image input type.")

        try:
            # SentenceTransformer('clip-...') can encode images directly if passed to encode
            embedding = self.vision_model.encode(
                image_obj, 
                show_progress_bar=False, 
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            if cache_key:
                self._cache[cache_key] = embedding
                
            return embedding
        except Exception as e:
            logger.error(f"Error embedding image: {e}")
            raise

    def embed_batch(self, items: List[Union[str, Path, Image.Image]]) -> List[np.ndarray]:
        """
        Embeds a mixed batch of text and images.
        Note: This is less efficient than batching same-types because models are different.
        For best performance, caller should separate texts and images.
        """
        results = []
        for item in items:
            if isinstance(item, str) and not (Path(item).exists() and Path(item).suffix.lower() in ['.jpg', '.jpeg', '.png']):
                # Heuristic: if it looks like a path and exists, treat as image, else text
                # But 'str' could be a path.
                # Requirement signature says items: List[Union[str, Path]]
                # We need a robust way to distinguish.
                # Let's assume if it is a Path object or a file path that exists, it is an image.
                # Otherwise treat as text.
                is_file = False
                try:
                    if Path(item).is_file():
                        is_file = True
                except OSError:
                    pass
                
                if is_file:
                    results.append(self.embed_image(item))
                else:
                    results.append(self.embed_text(item)) # type: ignore
            elif isinstance(item, (Path, Image.Image)):
                results.append(self.embed_image(item))
            else:
                 # Fallback treat as text
                 results.append(self.embed_text(str(item))) # type: ignore
        return results

    async def embed_text_async(self, texts: Union[str, List[str]]) -> Union[np.ndarray, List[np.ndarray]]:
        """Async wrapper for embed_text."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, 
            functools.partial(self.embed_text, texts)
        )

    async def embed_image_async(self, image_path: Union[str, Path]) -> np.ndarray:
        """Async wrapper for embed_image."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, 
            functools.partial(self.embed_image, image_path)
        )

    async def embed_batch_async(self, items: List[Union[str, Path]]) -> List[np.ndarray]:
        """Async wrapper for embed_batch."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, 
            functools.partial(self.embed_batch, items)
        )

# Global instance
embedding_service = EmbeddingService()
