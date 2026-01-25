"""
Deep debug script to trace the entire query flow
"""
from app.services.rag_pipeline import get_pipeline
from app.services.vector_store import VectorStore
from app.services.embeddings import EmbeddingService
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

print("=" * 70)
print("DEEP DEBUGGING - QUERY FLOW")
print("=" * 70)

# Step 1: Check vector store
print("\n1. Checking Vector Store...")
vs = VectorStore()
vs.initialize_collection("multimodal_rag", reset=False)
collection = vs.collections.get("multimodal_rag")
count = collection.count() if collection else 0
print(f"   Total documents: {count}")

if count == 0:
    print("   ❌ PROBLEM: Vector store is empty!")
    print("   You need to upload a PDF first")
    exit(1)

# Step 2: Test embedding generation
print("\n2. Testing Embedding Generation...")
embedding_service = EmbeddingService()
query = "What are the main topics covered?"

try:
    query_embedding = embedding_service.embed_text(query)
    if isinstance(query_embedding, list):
        query_embedding = query_embedding[0]
    print(f"   ✓ Query embedding generated")
    print(f"   Embedding shape: {query_embedding.shape if hasattr(query_embedding, 'shape') else len(query_embedding)}")
except Exception as e:
    print(f"   ❌ Embedding generation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 3: Test vector store retrieval
print("\n3. Testing Vector Store Retrieval...")
try:
    # Try direct query with the embedding
    results = vs.search(
        query_embedding=query_embedding,
        k=5,
        collection_name="multimodal_rag"
    )
    print(f"   ✓ Retrieval executed")
    print(f"   Results returned: {len(results)}")
    
    if len(results) == 0:
        print("   ❌ PROBLEM: No results returned from vector store!")
        print("   This could mean:")
        print("   - Embedding dimension mismatch")
        print("   - Collection is empty")
        print("   - Query filtering too strict")
        
        # Check document embeddings dimension
        print("\n   Checking sample document...")
        sample = collection.peek(1)
        if sample and sample['embeddings']:
            print(f"   Sample embedding shape: {len(sample['embeddings'][0])}")
    else:
        print(f"\n   Sample results:")
        for i, doc in enumerate(results[:3]):
            print(f"   {i+1}. Type: {doc.metadata.type}, Page: {doc.metadata.page}")
            print(f"      Content: {doc.content[:80]}...")
            
except Exception as e:
    print(f"   ❌ Retrieval failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 4: Test full pipeline query
print("\n4. Testing Full Pipeline Query...")
try:
    pipeline = get_pipeline()
    result = pipeline.query(
        query=query,
        pdf_id=None,
        top_k=5,
        include_visuals=True
    )
    
    print(f"   ✓ Pipeline query executed")
    print(f"   Processing time: {result.processing_time:.2f}s")
    print(f"   Sources returned: {len(result.sources)}")
    print(f"   Summary length: {len(result.summary)} chars")
    
    # Check if we got the error
    if "No text context available" in result.summary:
        print(f"\n   ❌ PROBLEM FOUND: 'No text context available' in summary")
        print(f"   This means the retrieval returned empty results to the LLM")
    else:
        print(f"\n   ✓ SUCCESS: Got proper summary")
        print(f"\n   Summary preview:")
        print(f"   {result.summary[:200]}...")
        
except Exception as e:
    print(f"   ❌ Pipeline query failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
