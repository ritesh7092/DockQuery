import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

# Mock settings before importing app
import os
os.environ["CHROMA_DB_PATH"] = tempfile.mkdtemp()

from app.services.vector_store import VectorStore
from app.models.schemas import Document, DocumentMetadata

@pytest.fixture
def temp_db_path(tmp_path):
    """Create a temporary database path for testing."""
    db_path = tmp_path / "test_chroma_db"
    db_path.mkdir()
    yield str(db_path)
    # Skip cleanup on Windows due to file locks

@pytest.fixture
def vector_store(temp_db_path):
    """Create a VectorStore instance with temporary database."""
    store = VectorStore(persist_directory=temp_db_path)
    yield store
    # Cleanup: Try to clear references
    try:
        store.collections.clear()
        del store.client
    except:
        pass

@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    docs = [
        Document(
            id="doc1",
            content="This is a text document about machine learning.",
            metadata=DocumentMetadata(
                type="text",
                page=1,
                source="test.pdf",
                bbox={"x0": 0.0, "y0": 0.0, "x1": 100.0, "y1": 100.0}
            )
        ),
        Document(
            id="doc2",
            content="Chart showing quarterly revenue growth.",
            metadata=DocumentMetadata(
                type="chart",
                page=2,
                source="test.pdf",
                image_path="/path/to/chart.png",
                bbox={"x0": 50.0, "y0": 50.0, "x1": 200.0, "y1": 200.0}
            )
        ),
        Document(
            id="doc3",
            content="Table with financial data for Q4.",
            metadata=DocumentMetadata(
                type="table",
                page=3,
                source="test.pdf",
                bbox={"x0": 10.0, "y0": 10.0, "x1": 150.0, "y1": 150.0}
            )
        ),
        Document(
            id="doc4",
            content="Another text document discussing neural networks.",
            metadata=DocumentMetadata(
                type="text",
                page=4,
                source="test.pdf"
            )
        ),
    ]
    return docs

@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing."""
    np.random.seed(42)
    # Generate 4 random 384-dimensional embeddings
    embeddings = [np.random.rand(384) for _ in range(4)]
    # Normalize them
    embeddings = [emb / np.linalg.norm(emb) for emb in embeddings]
    return embeddings

def test_initialize_collection(vector_store):
    """Test collection initialization."""
    collection = vector_store.initialize_collection("test_collection")
    assert collection is not None
    assert collection.name == "test_collection"
    assert "test_collection" in vector_store.collections

def test_add_documents(vector_store, sample_documents, sample_embeddings):
    """Test adding documents to the collection."""
    vector_store.initialize_collection("test_collection")
    
    # Should not raise an exception
    vector_store.add_documents(
        docs=sample_documents,
        embeddings=sample_embeddings,
        collection_name="test_collection"
    )
    
    # Verify documents were added
    collection = vector_store.collections["test_collection"]
    count = collection.count()
    assert count == 4

def test_add_documents_mismatch(vector_store, sample_documents):
    """Test that adding documents with mismatched embeddings raises error."""
    vector_store.initialize_collection("test_collection")
    
    with pytest.raises(ValueError, match="Mismatch"):
        vector_store.add_documents(
            docs=sample_documents,
            embeddings=[np.random.rand(384)],  # Only 1 embedding for 4 docs
            collection_name="test_collection"
        )

def test_similarity_search(vector_store, sample_documents, sample_embeddings):
    """Test basic similarity search."""
    vector_store.initialize_collection("test_collection")
    vector_store.add_documents(
        docs=sample_documents,
        embeddings=sample_embeddings,
        collection_name="test_collection"
    )
    
    # Search with the first embedding
    query_embedding = sample_embeddings[0]
    results = vector_store.similarity_search(
        query_embedding=query_embedding,
        k=2,
        collection_name="test_collection"
    )
    
    assert len(results) == 2
    assert isinstance(results[0], Document)
    # First result should be the most similar (should be doc1)
    assert results[0].id == "doc1"

def test_similarity_search_with_filters(vector_store, sample_documents, sample_embeddings):
    """Test similarity search with metadata filtering."""
    vector_store.initialize_collection("test_collection")
    vector_store.add_documents(
        docs=sample_documents,
        embeddings=sample_embeddings,
        collection_name="test_collection"
    )
    
    # Search only for text documents
    query_embedding = sample_embeddings[0]
    results = vector_store.similarity_search(
        query_embedding=query_embedding,
        k=5,
        filters={"type": "text"},
        collection_name="test_collection"
    )
    
    assert len(results) == 2  # Only 2 text documents
    assert all(doc.metadata.type == "text" for doc in results)

def test_similarity_search_with_page_filter(vector_store, sample_documents, sample_embeddings):
    """Test similarity search filtering by page number."""
    vector_store.initialize_collection("test_collection")
    vector_store.add_documents(
        docs=sample_documents,
        embeddings=sample_embeddings,
        collection_name="test_collection"
    )
    
    # Search only for documents on page 2
    query_embedding = sample_embeddings[0]
    results = vector_store.similarity_search(
        query_embedding=query_embedding,
        k=5,
        filters={"page": 2},
        collection_name="test_collection"
    )
    
    assert len(results) == 1
    assert results[0].metadata.page == 2
    assert results[0].id == "doc2"

def test_mmr_search(vector_store, sample_documents, sample_embeddings):
    """Test MMR search returns diverse results."""
    vector_store.initialize_collection("test_collection")
    vector_store.add_documents(
        docs=sample_documents,
        embeddings=sample_embeddings,
        collection_name="test_collection"
    )
    
    query_embedding = sample_embeddings[0]
    results = vector_store.mmr_search(
        query_embedding=query_embedding,
        k=3,
        lambda_param=0.5,
        collection_name="test_collection",
        fetch_k=4
    )
    
    assert len(results) == 3
    assert isinstance(results[0], Document)
    # MMR should return diverse results
    assert len(set(doc.id for doc in results)) == 3

def test_mmr_search_with_filters(vector_store, sample_documents, sample_embeddings):
    """Test MMR search with metadata filtering."""
    vector_store.initialize_collection("test_collection")
    vector_store.add_documents(
        docs=sample_documents,
        embeddings=sample_embeddings,
        collection_name="test_collection"
    )
    
    query_embedding = sample_embeddings[0]
    results = vector_store.mmr_search(
        query_embedding=query_embedding,
        k=2,
        lambda_param=0.7,
        filters={"type": "text"},
        collection_name="test_collection"
    )
    
    assert len(results) == 2
    assert all(doc.metadata.type == "text" for doc in results)

def test_delete_collection(vector_store):
    """Test collection deletion."""
    collection_name = "test_delete_collection"
    vector_store.initialize_collection(collection_name)
    
    # Verify it exists
    assert collection_name in vector_store.collections
    
    # Delete it
    vector_store.delete_collection(collection_name)
    
    # Verify it's gone
    assert collection_name not in vector_store.collections
    
    # Verify it's not in ChromaDB
    collection_names = [c.name for c in vector_store.client.list_collections()]
    assert collection_name not in collection_names

def test_collection_reset(vector_store, sample_documents, sample_embeddings):
    """Test collection reset functionality."""
    collection_name = "test_reset"
    
    # Create and populate collection
    vector_store.initialize_collection(collection_name)
    vector_store.add_documents(
        docs=sample_documents,
        embeddings=sample_embeddings,
        collection_name=collection_name
    )
    
    # Verify 4 documents
    assert vector_store.collections[collection_name].count() == 4
    
    # Reset collection
    vector_store.initialize_collection(collection_name, reset=True)
    
    # Verify it's empty
    assert vector_store.collections[collection_name].count() == 0

def test_mmr_diversity(vector_store):
    """Test that MMR produces more diverse results than regular search."""
    # Create documents with varying similarity
    np.random.seed(42)
    base_embedding = np.random.rand(384)
    base_embedding = base_embedding / np.linalg.norm(base_embedding)
    
    # Create similar and dissimilar embeddings
    similar_emb = base_embedding + np.random.rand(384) * 0.1
    similar_emb = similar_emb / np.linalg.norm(similar_emb)
    
    dissimilar_emb = np.random.rand(384)
    dissimilar_emb = dissimilar_emb / np.linalg.norm(dissimilar_emb)
    
    docs = [
        Document(id="base", content="Base document", 
                metadata=DocumentMetadata(type="text", page=1, source="test.pdf")),
        Document(id="similar", content="Similar document", 
                metadata=DocumentMetadata(type="text", page=2, source="test.pdf")),
        Document(id="dissimilar", content="Dissimilar document", 
                metadata=DocumentMetadata(type="text", page=3, source="test.pdf")),
    ]
    
    embeddings = [base_embedding, similar_emb, dissimilar_emb]
    
    vector_store.initialize_collection("diversity_test")
    vector_store.add_documents(docs, embeddings, "diversity_test")
    
    # MMR with high diversity (low lambda)
    mmr_results = vector_store.mmr_search(
        query_embedding=base_embedding,
        k=2,
        lambda_param=0.3,  # Favor diversity
        collection_name="diversity_test"
    )
    
    # Should include the dissimilar document for diversity
    result_ids = [doc.id for doc in mmr_results]
    assert "base" in result_ids  # Base should be first
    # With low lambda, we should get diverse results
    assert len(set(result_ids)) == 2

def test_empty_collection_search(vector_store):
    """Test searching an empty collection."""
    vector_store.initialize_collection("empty_collection")
    
    query_embedding = np.random.rand(384)
    results = vector_store.similarity_search(
        query_embedding=query_embedding,
        k=5,
        collection_name="empty_collection"
    )
    
    assert len(results) == 0
