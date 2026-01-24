"""
Unit tests for RAG Pipeline orchestrator.
"""

import pytest
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from app.services.rag_pipeline import RAGPipeline, get_pipeline
from app.models.schemas import (
    Document,
    DocumentMetadata,
    ProcessingResult,
    QueryResult,
    SourceReference
)


@pytest.fixture
def pipeline():
    """Create a RAGPipeline instance with mocked services."""
    with patch('app.services.rag_pipeline.PDFParser'), \
         patch('app.services.rag_pipeline.EmbeddingService'), \
         patch('app.services.rag_pipeline.VectorStore'), \
         patch('app.services.rag_pipeline.GeminiAgent'):
        pipeline = RAGPipeline()
        return pipeline


@pytest.fixture
def mock_parsed_pdf():
    """Mock parsed PDF data."""
    return {
        "text_blocks": [
            {"page": 1, "text": "Introduction to AI", "bbox": [0, 0, 100, 20]},
            {"page": 1, "text": "Machine learning basics", "bbox": [0, 30, 100, 50]},
            {"page": 2, "text": "Deep learning advances", "bbox": [0, 0, 100, 20]},
        ],
        "images": [
            {
                "page": 1,
                "image_path": "data/extracted/test_p1_i0.png",
                "type": "chart",
                "bbox": [0, 60, 100, 160],
                "caption_context": "Revenue growth chart"
            },
            {
                "page": 2,
                "image_path": "data/extracted/test_p2_i0.png",
                "type": "image",
                "bbox": [0, 60, 100, 160]
            }
        ],
        "tables": [
            {
                "page": 2,
                "bbox": [0, 180, 100, 240],
                "nearby_text": "Quarterly results"
            }
        ],
        "metadata": {}
    }


@pytest.fixture
def mock_documents():
    """Mock document results from vector store."""
    return [
        Document(
            id="doc1",
            content="Introduction to artificial intelligence",
            metadata=DocumentMetadata(
                type="text",
                page=1,
                source="test_pdf"
            )
        ),
        Document(
            id="doc2",
            content="Chart on page 1: Revenue growth",
            metadata=DocumentMetadata(
                type="chart",
                page=1,
                source="test_pdf",
                image_path="data/extracted/test_p1_i0.png"
            )
        ),
        Document(
            id="doc3",
            content="Deep learning advances",
            metadata=DocumentMetadata(
                type="text",
                page=2,
                source="test_pdf"
            )
        )
    ]


class TestPipelineInitialization:
    """Test RAGPipeline initialization."""
    
    def test_init(self, pipeline):
        """Test successful initialization."""
        assert pipeline.pdf_parser is not None
        assert pipeline.embedding_service is not None
        assert pipeline.vector_store is not None
        assert pipeline.agent is not None
        assert len(pipeline.query_cache) == 0
        assert pipeline.metrics["pdfs_processed"] == 0
    
    def test_singleton_pattern(self):
        """Test get_pipeline returns singleton."""
        with patch('app.services.rag_pipeline.RAGPipeline'):
            pipeline1 = get_pipeline()
            pipeline2 = get_pipeline()
            assert pipeline1 is pipeline2


class TestPDFProcessing:
    """Test PDF processing workflow."""
    
    def test_process_pdf_success(self, pipeline, mock_parsed_pdf):
        """Test successful PDF processing."""
        # Mock PDF parser
        pipeline.pdf_parser.parse_pdf = Mock(return_value=mock_parsed_pdf)
        
        # Mock embedding service
        pipeline.embedding_service.embed_text = Mock(
            return_value=np.random.rand(3, 384)  # 3 text blocks
        )
        pipeline.embedding_service.embed_image = Mock(
            return_value=np.random.rand(384)
        )
        
        # Mock vector store
        pipeline.vector_store.initialize_collection = Mock()
        pipeline.vector_store.add_documents = Mock()
        
        # Process PDF
        result = pipeline.process_pdf(
            file_path="test.pdf",
            pdf_id="test_001"
        )
        
        assert result.status == "success"
        assert result.pdf_id == "test_001"
        assert result.total_pages == 2
        assert result.text_chunks == 3
        assert result.images_extracted == 2
        assert result.tables_found >= 1
        assert result.processing_time > 0
        assert result.errors is None
    
    def test_process_pdf_parsing_failure(self, pipeline):
        """Test PDF processing with parsing failure."""
        pipeline.pdf_parser.parse_pdf = Mock(
            side_effect=Exception("Parse error")
        )
        
        result = pipeline.process_pdf(
            file_path="test.pdf",
            pdf_id="test_002"
        )
        
        assert result.status == "failed"
        assert result.total_pages == 0
        assert result.errors is not None
        assert "parsing failed" in result.errors[0].lower()
    
    def test_process_text_blocks(self, pipeline):
        """Test text block processing."""
        text_blocks = [
            {"page": 1, "text": "Block 1", "bbox": [0, 0, 100, 20]},
            {"page": 1, "text": "Block 2", "bbox": [0, 30, 100, 50]}
        ]
        
        pipeline.embedding_service.embed_text = Mock(
            return_value=np.random.rand(2, 384)
        )
        
        docs, embeddings = pipeline._process_text_blocks(text_blocks, "test_pdf")
        
        assert len(docs) == 2
        assert len(embeddings) == 2
        assert docs[0].content == "Block 1"
        assert docs[0].metadata.type == "text"
        assert docs[0].metadata.page == 1


class TestQueryHandling:
    """Test query handling workflow."""
    
    def test_query_success(self, pipeline, mock_documents):
        """Test successful query processing."""
        # Mock embedding service
        pipeline.embedding_service.embed_text = Mock(
            return_value=np.random.rand(384)
        )
        
        # Mock vector store
        pipeline.vector_store.mmr_search = Mock(
            return_value=mock_documents
        )
        
        # Mock agent
        pipeline.agent.generate_summary = Mock(
            return_value="This is a summary about AI and machine learning."
        )
        
        result = pipeline.query(
            query="What is AI?",
            pdf_id="test_pdf",
            top_k=3
        )
        
        assert result.query == "What is AI?"
        assert len(result.summary) > 0
        assert len(result.sources) > 0
        assert result.confidence > 0
        assert result.processing_time > 0
        assert result.cached is False
    
    def test_query_caching(self, pipeline, mock_documents):
        """Test query result caching."""
        # Setup mocks
        pipeline.embedding_service.embed_text = Mock(
            return_value=np.random.rand(384)
        )
        pipeline.vector_store.mmr_search = Mock(
            return_value=mock_documents
        )
        pipeline.agent.generate_summary = Mock(
            return_value="Cached summary"
        )
        
        # First query (not cached)
        result1 = pipeline.query(
            query="What is AI?",
            pdf_id="test_pdf",
            top_k=3
        )
        
        assert result1.cached is False
        assert pipeline.metrics["cache_misses"] == 1
        assert pipeline.metrics["cache_hits"] == 0
        
        # Second query (should be cached)
        result2 = pipeline.query(
            query="What is AI?",
            pdf_id="test_pdf",
            top_k=3
        )
        
        assert result2.cached is True
        assert pipeline.metrics["cache_hits"] == 1
        assert len(pipeline.query_cache) == 1
    
    def test_query_without_visuals(self, pipeline, mock_documents):
        """Test query with include_visuals=False."""
        pipeline.embedding_service.embed_text = Mock(
            return_value=np.random.rand(384)
        )
        pipeline.vector_store.mmr_search = Mock(
            return_value=mock_documents
        )
        pipeline.agent.generate_summary = Mock(
            return_value="Text-only summary"
        )
        
        result = pipeline.query(
            query="Test query",
            include_visuals=False
        )
        
        # Should still succeed
        assert result.summary == "Text-only summary"
        assert len(result.sources) > 0
    
    def test_query_error_handling(self, pipeline):
        """Test query with error in processing."""
        pipeline.embedding_service.embed_text = Mock(
            side_effect=Exception("Embedding error")
        )
        
        result = pipeline.query(query="Test query")
        
        assert "Error" in result.summary
        assert len(result.sources) == 0
        assert result.confidence == 0.0


class TestContextAggregation:
    """Test context aggregation."""
    
    def test_aggregate_context(self, pipeline, mock_documents):
        """Test context aggregation from results."""
        text_docs = [d for d in mock_documents if d.metadata.type == "text"]
        visual_docs = [d for d in mock_documents if d.metadata.type != "text"]
        
        text_context, visual_paths = pipeline._aggregate_context(
            text_docs,
            visual_docs
        )
        
        assert len(text_context) == len(text_docs)
        assert all("[Page" in ctx for ctx in text_context)
        assert len(visual_paths) == len([d for d in visual_docs if d.metadata.image_path])


class TestSourceReferences:
    """Test source reference creation."""
    
    def test_create_source_references(self, pipeline, mock_documents):
        """Test source reference extraction."""
        sources = pipeline._create_source_references(mock_documents)
        
        assert len(sources) == len(mock_documents)
        assert all(isinstance(s, SourceReference) for s in sources)
        assert all(s.page > 0 for s in sources)
        assert all(s.type in ["text", "chart", "image", "table"] for s in sources)
        assert all(len(s.content_preview) > 0 for s in sources)


class TestCaching:
    """Test caching mechanisms."""
    
    def test_cache_key_generation(self, pipeline):
        """Test cache key generation is consistent."""
        key1 = pipeline._get_cache_key("query1", "pdf1", 5, True)
        key2 = pipeline._get_cache_key("query1", "pdf1", 5, True)
        key3 = pipeline._get_cache_key("query2", "pdf1", 5, True)
        
        assert key1 == key2
        assert key1 != key3
    
    def test_cache_expiration(self, pipeline):
        """Test cache TTL expiration."""
        # Create a mock result
        mock_result = QueryResult(
            query="test",
            summary="test summary",
            sources=[],
            confidence=0.8,
            processing_time=1.0
        )
        
        # Cache with short TTL
        pipeline.cache_ttl = 1  # 1 second
        cache_key = "test_key"
        pipeline._cache_query_result(cache_key, mock_result)
        
        # Should be cached immediately
        cached = pipeline._get_cached_query(cache_key)
        assert cached is not None
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be expired
        cached = pipeline._get_cached_query(cache_key)
        assert cached is None
    
    def test_cache_lru_eviction(self, pipeline):
        """Test LRU cache eviction."""
        pipeline.cache_max_size = 2
        
        # Add 3 items (should evict oldest)
        for i in range(3):
            mock_result = QueryResult(
                query=f"query{i}",
                summary="summary",
                sources=[],
                confidence=0.8,
                processing_time=1.0
            )
            pipeline._cache_query_result(f"key{i}", mock_result)
        
        # Cache should only have 2 items
        assert len(pipeline.query_cache) == 2
        # First item should be evicted
        assert "key0" not in pipeline.query_cache
        assert "key1" in pipeline.query_cache
        assert "key2" in pipeline.query_cache
    
    def test_clear_cache(self, pipeline):
        """Test cache clearing."""
        # Add some cached items
        for i in range(3):
            mock_result = QueryResult(
                query=f"query{i}",
                summary="summary",
                sources=[],
                confidence=0.8,
                processing_time=1.0
            )
            pipeline._cache_query_result(f"key{i}", mock_result)
        
        assert len(pipeline.query_cache) > 0
        
        pipeline.clear_cache()
        
        assert len(pipeline.query_cache) == 0


class TestMetrics:
    """Test metrics collection."""
    
    def test_metrics_tracking(self, pipeline, mock_parsed_pdf, mock_documents):
        """Test that metrics are properly tracked."""
        # Process a PDF
        pipeline.pdf_parser.parse_pdf = Mock(return_value=mock_parsed_pdf)
        pipeline.embedding_service.embed_text = Mock(
            return_value=np.random.rand(3, 384)
        )
        pipeline.embedding_service.embed_image = Mock(
            return_value=np.random.rand(384)
        )
        pipeline.vector_store.initialize_collection = Mock()
        pipeline.vector_store.add_documents = Mock()
        
        pipeline.process_pdf("test.pdf", "test_001")
        
        assert pipeline.metrics["pdfs_processed"] == 1
        
        # Handle a query
        pipeline.embedding_service.embed_text = Mock(
            return_value=np.random.rand(384)
        )
        pipeline.vector_store.mmr_search = Mock(return_value=mock_documents)
        pipeline.agent.generate_summary = Mock(return_value="Summary")
        
        pipeline.query("test query")
        
        assert pipeline.metrics["queries_handled"] == 1
        assert pipeline.metrics["cache_misses"] == 1
    
    def test_get_metrics(self, pipeline):
        """Test metrics retrieval."""
        metrics = pipeline.get_metrics()
        
        assert "pdfs_processed" in metrics
        assert "queries_handled" in metrics
        assert "cache_hits" in metrics
        assert "cache_misses" in metrics
        assert "cache_size" in metrics
        assert "cache_hit_rate" in metrics


class TestReranking:
    """Test result re-ranking."""
    
    def test_rerank_results(self, pipeline, mock_documents):
        """Test result re-ranking logic."""
        reranked = pipeline._rerank_results("test query", mock_documents)
        
        # Should preserve all documents
        assert len(reranked) == len(mock_documents)
        
        # Should be sorted by page (earlier pages first)
        pages = [doc.metadata.page for doc in reranked]
        assert pages == sorted(pages)


class TestConfidence:
    """Test confidence calculation."""
    
    def test_calculate_confidence_with_results(self, pipeline, mock_documents):
        """Test confidence with good results."""
        confidence = pipeline._calculate_confidence(mock_documents)
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be reasonably confident with 3 results
    
    def test_calculate_confidence_no_results(self, pipeline):
        """Test confidence with no results."""
        confidence = pipeline._calculate_confidence([])
        
        assert confidence == 0.0
    
    def test_calculate_confidence_with_visuals(self, pipeline, mock_documents):
        """Test confidence boost with visual elements."""
        # Calculate with visuals
        conf_with_visuals = pipeline._calculate_confidence(mock_documents)
        
        # Calculate without visuals
        text_only = [d for d in mock_documents if d.metadata.type == "text"]
        conf_without_visuals = pipeline._calculate_confidence(text_only)
        
        # Should be higher (or equal) with visuals
        assert conf_with_visuals >= conf_without_visuals
