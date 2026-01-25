# -*- coding: utf-8 -*-
"""
Comprehensive diagnostic to find why text context is empty despite retrieval working
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from app.services.rag_pipeline import get_pipeline
from app.services.vector_store import VectorStore
from app.services.embeddings import EmbeddingService
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

print("=" * 70)
print("COMPREHENSIVE DIAGNOSTIC - WHY IS TEXT CONTEXT EMPTY?")
print("=" * 70)

query = "What are the main topics covered?"

# Get pipeline
pipeline = get_pipeline()

# Step 1: Check embedding dimensions
print("\n[1] CHECKING EMBEDDING DIMENSIONS")
print("-" * 70)

embedding_service = EmbeddingService()
query_embedding = embedding_service.embed_text(query)
if isinstance(query_embedding, list):
    query_embedding = query_embedding[0]

print(f"Query embedding shape: {query_embedding.shape}")
print(f"Query embedding dimension: {query_embedding.shape[0] if hasattr(query_embedding, 'shape') else len(query_embedding)}")

# Check stored document embedding dimension
vs = VectorStore()
vs.initialize_collection("multimodal_rag", reset=False)
collection = vs.collections.get("multimodal_rag")

sample = collection.peek(1)
if sample is not None and 'embeddings' in sample and len(sample['embeddings']) > 0:
    stored_dim = len(sample['embeddings'][0])
    print(f"Stored embedding dimension: {stored_dim}")
    
    query_dim = query_embedding.shape[0] if hasattr(query_embedding, 'shape') else len(query_embedding)
    if stored_dim != query_dim:
        print(f"\nERROR: DIMENSION MISMATCH!")
        print(f"  Query: {query_dim}, Stored: {stored_dim}")
        print(f"  This is why retrieval returns empty results!")
    else:
        print(f"OK: Dimensions match!")

# Step 2: Test retrieval directly
print(f"\n[2] TESTING DIRECT RETRIEVAL")
print("-" * 70)

results = vs.similarity_search(
    query_embedding=query_embedding,
    k=5,
    collection_name="multimodal_rag"
)

print(f"Results retrieved: {len(results)}")

if len(results) > 0:
    print(f"\nSample results:")
    for i, doc in enumerate(results[:3]):
        print(f"\n  [{i+1}] Type: {doc.metadata.type}, Page: {doc.metadata.page}")
        print(f"      Source: {doc.metadata.source}")
        print(f"      Content length: {len(doc.content)} chars")
        content_preview = doc.content[:150] if doc.content else "[EMPTY]"
        print(f"      Content preview: {content_preview}...")
        
        # Check if content is actually empty
        if not doc.content or len(doc.content.strip()) == 0:
            print(f"      *** WARNING: Content is EMPTY! ***")
else:
    print("*** NO RESULTS - This is the problem! ***")

# Step 3: Test context aggregation
print(f"\n[3] TESTING CONTEXT AGGREGATION")
print("-" * 70)

if len(results) > 0:
    text_results = [r for r in results if r.metadata.type == "text"]
    visual_results = [r for r in results if r.metadata.type in ["image", "table", "chart"]]
    
    print(f"Text results: {len(text_results)}")
    print(f"Visual results: {len(visual_results)}")
    
    # Manually aggregate
    text_context, visual_paths = pipeline._aggregate_context(text_results, visual_results)
    
    print(f"\nAggregated text context items: {len(text_context)}")
    print(f"Aggregated visual paths: {len(visual_paths)}")
    
    if len(text_context) > 0:
        print(f"\nSample text context:")
        for i, ctx in enumerate(text_context[:2]):
            print(f"  [{i+1}] Length: {len(ctx)} chars")
            print(f"      Content: {ctx[:150]}...")
            
            if not ctx or len(ctx.strip()) <= 10:  # Just page number?
                print(f"      WARNING: Context is essentially empty!")
    else:
        print("WARNING: text_context list is empty after aggregation!")

# Step 4: Test prompt building
print(f"\n[4] TESTING PROMPT BUILDING")
print("-" * 70)

if len(text_context) > 0:
    prompt = pipeline.agent._build_prompt(
        query=query,
        text_context=text_context,
        has_visuals=(len(visual_paths) > 0)
    )
    
    print(f"Prompt length: {len(prompt)} chars")
    
    if "No text context available" in prompt:
        print("ERROR: Prompt contains 'No text context available'!")
        print("This should not happen if we have text_context!")
    else:
        print("OK: Prompt was built correctly")
    
    # Check what's in the prompt
    if text_context[0][:50] in prompt:
        print("OK: Text context IS in the prompt")
    else:
        print("WARNING: Text context might not be in prompt correctly")
else:
    print("SKIPPED: No text context to test with")

# Step 5: Full pipeline test
print(f"\n[5] TESTING FULL PIPELINE")
print("-" * 70)

result = pipeline.query(
    query=query,
    pdf_id=None,  # No filter
    top_k=5,
    include_visuals=True
)

print(f"Pipeline executed in {result.processing_time:.2f}s")
print(f"Confidence: {result.confidence}")
print(f"Sources returned: {len(result.sources)}")
print(f"Summary length: {len(result.summary)} chars")

if "No text context available" in result.summary:
    print(f"\nERROR FOUND IN SUMMARY!")
    print("The LLM received 'No text context available' message")
else:
    print(f"\nSUCCESS: Proper summary generated")

print(f"\nSummary preview:")
print("-" * 70)
print(result.summary[:500])
print("-" * 70)

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
