"""
RAG Pipeline Orchestrator - Coordinates all multimodal RAG components.

This module orchestrates the complete RAG workflow:
1. PDF Processing: Parse, embed, and store document content
2. Query Handling: Retrieve, re-rank, and generate responses
"""

import asyncio
import hashlib
import logging
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from app.config import settings
from app.models.schemas import (
    Document,
    DocumentMetadata,
    ProcessingResult,
    QueryResult,
    SourceReference
)
from app.services.pdf_parser import PDFParser
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore
from app.services.agent import GeminiAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Production-ready RAG pipeline orchestrator.
    
    Coordinates PDF processing, embedding generation, vector storage,
    and query handling with caching, parallel processing, and error recovery.
    """
    
    def __init__(self):
        """Initialize the RAG pipeline with all service components."""
        logger.info("Initializing RAG Pipeline...")
        
        # Initialize service components
        self.pdf_parser = PDFParser(output_dir=settings.EXTRACTED_DIR)
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        self.agent = GeminiAgent(api_key=settings.GOOGLE_API_KEY)
        
        # Initialize cache (LRU with TTL)
        self.query_cache: OrderedDict[str, Tuple[QueryResult, float]] = OrderedDict()
        self.cache_max_size = settings.CACHE_MAX_SIZE
        self.cache_ttl = settings.CACHE_TTL_SECONDS
        
        # Metrics
        self.metrics = {
            "pdfs_processed": 0,
            "queries_handled": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        logger.info("RAG Pipeline initialized successfully")
    
    def process_pdf(
        self,
        file_path: str,
        pdf_id: str,
        collection_name: str = "multimodal_rag"
    ) -> ProcessingResult:
        """
        Process a PDF document through the complete pipeline.
        
        Workflow:
        1. Parse PDF (extract text, images, tables, charts)
        2. Generate embeddings for all content
        3. Store in vector database with metadata
        
        Args:
            file_path: Path to PDF file
            pdf_id: Unique identifier for this PDF
            collection_name: Vector store collection name
            
        Returns:
            ProcessingResult with statistics and status
        """
        start_time = time.time()
        errors = []
        
        try:
            logger.info(f"Processing PDF: {pdf_id} from {file_path}")
            
            # Step 1: Parse PDF
            logger.info("Step 1/3: Parsing PDF...")
            try:
                parsed_data = self.pdf_parser.parse_pdf(file_path)
            except Exception as e:
                logger.error(f"PDF parsing failed: {e}")
                return ProcessingResult(
                    pdf_id=pdf_id,
                    total_pages=0,
                    text_chunks=0,
                    images_extracted=0,
                    tables_found=0,
                    charts_found=0,
                    processing_time=time.time() - start_time,
                    status="failed",
                    errors=[f"PDF parsing failed: {str(e)}"]
                )
            
            # Extract statistics
            total_pages = len(set(item["page"] for item in parsed_data["text_blocks"]))
            text_blocks = parsed_data["text_blocks"]
            images = parsed_data["images"]
            tables = parsed_data["tables"]
            
            # Count visual elements by type
            charts_count = sum(1 for img in images if img.get("type") == "chart")
            tables_count = len(tables) + sum(1 for img in images if img.get("type") == "table")
            images_count = len(images)
            
            logger.info(f"Parsed {total_pages} pages: {len(text_blocks)} text blocks, "
                       f"{images_count} images, {tables_count} tables")
            
            # Step 2: Generate embeddings
            logger.info("Step 2/3: Generating embeddings...")
            documents = []
            embeddings = []
            
            try:
                # Process text blocks in parallel batches
                text_docs, text_embeddings = self._process_text_blocks(
                    text_blocks, pdf_id
                )
                documents.extend(text_docs)
                embeddings.extend(text_embeddings)
                
                # Process visual elements
                visual_docs, visual_embeddings = self._process_visual_elements(
                    images, tables, pdf_id
                )
                documents.extend(visual_docs)
                embeddings.extend(visual_embeddings)
                
                logger.info(f"Generated {len(embeddings)} embeddings")
                
            except Exception as e:
                logger.error(f"Embedding generation failed: {e}")
                errors.append(f"Embedding generation error: {str(e)}")
            
            # Step 3: Store in vector database
            logger.info("Step 3/3: Storing in vector database...")
            try:
                if documents and embeddings:
                    self.vector_store.initialize_collection(
                        collection_name=collection_name,
                        reset=False
                    )
                    self.vector_store.add_documents(
                        docs=documents,
                        embeddings=embeddings,
                        collection_name=collection_name
                    )
                    logger.info(f"Stored {len(documents)} documents in vector store")
                else:
                    errors.append("No documents to store")
                    
            except Exception as e:
                logger.error(f"Vector store operation failed: {e}")
                errors.append(f"Vector store error: {str(e)}")
            
            # Update metrics
            self.metrics["pdfs_processed"] += 1
            
            processing_time = time.time() - start_time
            status = "success" if not errors else ("partial" if documents else "failed")
            
            result = ProcessingResult(
                pdf_id=pdf_id,
                total_pages=total_pages,
                text_chunks=len(text_blocks),
                images_extracted=images_count,
                tables_found=tables_count,
                charts_found=charts_count,
                processing_time=processing_time,
                status=status,
                errors=errors if errors else None
            )
            
            logger.info(f"PDF processing complete: {status} in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Unexpected error in process_pdf: {e}")
            return ProcessingResult(
                pdf_id=pdf_id,
                total_pages=0,
                text_chunks=0,
                images_extracted=0,
                tables_found=0,
                charts_found=0,
                processing_time=time.time() - start_time,
                status="failed",
                errors=[f"Unexpected error: {str(e)}"]
            )
    
    def query(
        self,
        query: str,
        pdf_id: Optional[str] = None,
        top_k: int = 5,
        include_visuals: bool = True,
        collection_name: str = "multimodal_rag"
    ) -> QueryResult:
        """
        Handle a query through the complete RAG pipeline.
        
        Workflow:
        1. Check cache for existing results
        2. Embed query
        3. Retrieve relevant text and visual content
        4. Re-rank results
        5. Aggregate context
        6. Generate summary using LLM
        7. Extract source references
        8. Cache results
        
        Args:
            query: User's query string
            pdf_id: Optional PDF ID to filter results
            top_k: Number of results to return
            include_visuals: Whether to include visual elements
            collection_name: Vector store collection name
            
        Returns:
            QueryResult with summary and source references
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing query: '{query}' (pdf_id={pdf_id}, top_k={top_k})")
            
            # Step 1: Check cache
            cache_key = self._get_cache_key(query, pdf_id, top_k, include_visuals)
            cached_result = self._get_cached_query(cache_key)
            
            if cached_result:
                logger.info("Query result found in cache")
                self.metrics["cache_hits"] += 1
                return cached_result
            
            self.metrics["cache_misses"] += 1
            
            # Step 2: Embed query
            logger.info("Embedding query...")
            query_embedding = self.embedding_service.embed_text(query)
            if isinstance(query_embedding, list):
                query_embedding = query_embedding[0]
            
            # Step 3: Retrieve relevant content
            logger.info("Retrieving relevant content...")
            
            # Build filters for pdf_id if provided
            filters = {"source": pdf_id} if pdf_id else None
            
            # Retrieve with MMR for diversity
            fetch_k = settings.RERANK_TOP_K
            all_results = self.vector_store.mmr_search(
                query_embedding=query_embedding,
                k=top_k,
                lambda_param=0.7,  # Balance relevance and diversity
                filters=filters,
                collection_name=collection_name,
                fetch_k=fetch_k
            )
            
            logger.info(f"Retrieved {len(all_results)} results")
            
            # Step 4: Separate text and visual results
            text_results = [r for r in all_results if r.metadata.type == "text"]
            visual_results = [r for r in all_results if r.metadata.type in ["image", "table", "chart"]]
            
            if not include_visuals:
                visual_results = []
            
            logger.info(f"Split into {len(text_results)} text and {len(visual_results)} visual results")
            
            # Step 5: Re-rank results
            reranked_results = self._rerank_results(query, all_results)
            
            # Step 6: Aggregate context
            text_context, visual_context = self._aggregate_context(
                text_results[:top_k],
                visual_results[:min(3, len(visual_results))]  # Limit visuals to avoid token overflow
            )
            
            # Step 7: Generate summary
            logger.info("Generating summary with LLM...")
            try:
                summary = self.agent.generate_summary(
                    query=query,
                    text_context=text_context,
                    visual_context=visual_context,
                    stream=False
                )
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                summary = f"Error generating summary: {str(e)}"
            
            # Step 8: Create source references
            sources = self._create_source_references(reranked_results[:top_k])
            
            # Calculate confidence
            confidence = self._calculate_confidence(reranked_results[:top_k])
            
            processing_time = time.time() - start_time
            
            result = QueryResult(
                query=query,
                summary=summary,
                sources=sources,
                confidence=confidence,
                processing_time=processing_time,
                cached=False
            )
            
            # Cache the result
            self._cache_query_result(cache_key, result)
            
            # Update metrics
            self.metrics["queries_handled"] += 1
            
            logger.info(f"Query complete in {processing_time:.2f}s (confidence: {confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"Error in query processing: {e}")
            return QueryResult(
                query=query,
                summary=f"Error processing query: {str(e)}",
                sources=[],
                confidence=0.0,
                processing_time=time.time() - start_time,
                cached=False
            )
    
    def _process_text_blocks(
        self,
        text_blocks: List[Dict],
        pdf_id: str
    ) -> Tuple[List[Document], List[np.ndarray]]:
        """Process text blocks in parallel batches."""
        documents = []
        
        # Create documents
        for i, block in enumerate(text_blocks):
            doc = Document(
                id=f"{pdf_id}_text_{i}",
                content=block["text"],
                metadata=DocumentMetadata(
                    type="text",
                    page=block["page"],
                    source=pdf_id,
                    bbox=block.get("bbox")
                )
            )
            documents.append(doc)
        
        # Generate embeddings in batches
        texts = [doc.content for doc in documents]
        embeddings = self.embedding_service.embed_text(texts)
        
        # Convert to list of numpy arrays if needed
        if isinstance(embeddings, np.ndarray) and len(embeddings.shape) == 2:
            embeddings = [embeddings[i] for i in range(len(embeddings))]
        
        return documents, embeddings
    
    def _process_visual_elements(
        self,
        images: List[Dict],
        tables: List[Dict],
        pdf_id: str
    ) -> Tuple[List[Document], List[np.ndarray]]:
        """Process visual elements (images, tables, charts)."""
        documents = []
        embeddings = []
        
        # Process images
        for i, img in enumerate(images):
            img_path = Path(img["image_path"])
            if not img_path.exists():
                logger.warning(f"Image not found: {img_path}")
                continue
            
            try:
                # Generate image embedding
                img_embedding = self.embedding_service.embed_image(img_path)
                
                # Create document with image description
                content = f"{img['type']} on page {img['page']}"
                if img.get("caption_context"):
                    content += f": {img['caption_context']}"
                
                doc = Document(
                    id=f"{pdf_id}_{img['type']}_{i}",
                    content=content,
                    metadata=DocumentMetadata(
                        type=img["type"],
                        page=img["page"],
                        source=pdf_id,
                        image_path=str(img_path),
                        bbox=img.get("bbox")
                    )
                )
                
                documents.append(doc)
                embeddings.append(img_embedding)
                
            except Exception as e:
                logger.warning(f"Failed to process image {img_path}: {e}")
                continue
        
        # Process tables (as text embeddings with table metadata)
        for i, table in enumerate(tables):
            nearby_text = table.get("nearby_text", "")
            content = f"Table on page {table['page']}: {nearby_text}"
            
            try:
                # Use text embedding for table context
                table_embedding = self.embedding_service.embed_text(content)
                if isinstance(table_embedding, list):
                    table_embedding = table_embedding[0]
                
                doc = Document(
                    id=f"{pdf_id}_table_{i}",
                    content=content,
                    metadata=DocumentMetadata(
                        type="table",
                        page=table["page"],
                        source=pdf_id,
                        bbox=table.get("bbox")
                    )
                )
                
                documents.append(doc)
                embeddings.append(table_embedding)
                
            except Exception as e:
                logger.warning(f"Failed to process table {i}: {e}")
                continue
        
        return documents, embeddings
    
    def _rerank_results(
        self,
        query: str,
        results: List[Document]
    ) -> List[Document]:
        """
        Re-rank results using hybrid scoring.
        
        Scoring factors:
        - Semantic similarity (already from vector store)
        - Type weighting (boost visual elements for visual queries)
        - Recency (prefer earlier pages as tie-breaker)
        """
        # Simple re-ranking: already handled by MMR in retrieval
        # Could add additional scoring here if needed
        
        # Sort by page number as secondary sort (earlier pages first)
        return sorted(results, key=lambda r: r.metadata.page)
    
    def _aggregate_context(
        self,
        text_results: List[Document],
        visual_results: List[Document]
    ) -> Tuple[List[str], List[Path]]:
        """
        Aggregate context for LLM input.
        
        Returns:
            Tuple of (text_context_list, visual_paths_list)
        """
        # Prepare text context with page attribution
        text_context = []
        for doc in text_results:
            context_str = f"[Page {doc.metadata.page}] {doc.content}"
            text_context.append(context_str)
        
        # Prepare visual paths
        visual_paths = []
        for doc in visual_results:
            if doc.metadata.image_path:
                visual_paths.append(Path(doc.metadata.image_path))
        
        return text_context, visual_paths
    
    def _create_source_references(
        self,
        results: List[Document]
    ) -> List[SourceReference]:
        """Create source references from results."""
        sources = []
        
        for doc in results:
            # Create preview (first 100 chars)
            preview = doc.content[:100] + "..." if len(doc.content) > 100 else doc.content
            
            source = SourceReference(
                type=doc.metadata.type,
                page=doc.metadata.page,
                content_preview=preview,
                image_path=doc.metadata.image_path,
                confidence=0.8  # Could be computed from similarity scores
            )
            sources.append(source)
        
        return sources
    
    def _calculate_confidence(
        self,
        results: List[Document]
    ) -> float:
        """
        Calculate overall confidence score.
        
        Based on:
        - Number of results found
        - Quality of matches (would need similarity scores from vector store)
        """
        if not results:
            return 0.0
        
        # Simple heuristic: more results = higher confidence
        base_confidence = min(len(results) / 5.0, 1.0)
        
        # Boost if we have visual elements
        has_visuals = any(r.metadata.type in ["image", "table", "chart"] for r in results)
        if has_visuals:
            base_confidence = min(base_confidence * 1.1, 1.0)
        
        return round(base_confidence, 2)
    
    def _get_cache_key(
        self,
        query: str,
        pdf_id: Optional[str],
        top_k: int,
        include_visuals: bool
    ) -> str:
        """Generate cache key from query parameters."""
        key_data = f"{query}|{pdf_id}|{top_k}|{include_visuals}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_query(self, cache_key: str) -> Optional[QueryResult]:
        """Retrieve cached query result if valid."""
        if cache_key not in self.query_cache:
            return None
        
        result, timestamp = self.query_cache[cache_key]
        
        # Check if cache entry is expired
        if time.time() - timestamp > self.cache_ttl:
            del self.query_cache[cache_key]
            return None
        
        # Move to end (LRU)
        self.query_cache.move_to_end(cache_key)
        
        # Mark as cached
        result.cached = True
        return result
    
    def _cache_query_result(self, cache_key: str, result: QueryResult):
        """Cache a query result with TTL."""
        # Evict oldest if cache is full
        if len(self.query_cache) >= self.cache_max_size:
            self.query_cache.popitem(last=False)  # Remove oldest
        
        self.query_cache[cache_key] = (result, time.time())
        logger.debug(f"Cached query result (cache size: {len(self.query_cache)})")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline metrics."""
        return {
            **self.metrics,
            "cache_size": len(self.query_cache),
            "cache_hit_rate": (
                self.metrics["cache_hits"] / max(self.metrics["queries_handled"], 1)
                if self.metrics["queries_handled"] > 0 else 0.0
            )
        }
    
    def clear_cache(self):
        """Clear the query cache."""
        self.query_cache.clear()
        logger.info("Query cache cleared")


# Singleton instance
_pipeline_instance = None

def get_pipeline() -> RAGPipeline:
    """
    Get or create a singleton RAGPipeline instance.
    
    Returns:
        RAGPipeline instance
    """
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = RAGPipeline()
    return _pipeline_instance
