"""
Test query after fixing the vector store
"""
from app.services.rag_pipeline import get_pipeline
import logging

logging.basicConfig(level=logging.WARNING)

pdf_id = "7c329583-b845-4b12-ac62-4a0952af5184"
query = "What are the main topics covered?"

print("Testing query after fix:")
print(f"  PDF ID: {pdf_id}")
print(f"  Query: {query}")
print("-" * 80)

pipeline = get_pipeline()

result = pipeline.query(
    query=query,
    pdf_id=pdf_id,
    top_k=5,
    include_visuals=True
)

print(f"\nQuery Result:")
print(f"  Summary: {result.summary}")
print(f"  Sources: {len(result.sources)}")
print(f"  Confidence: {result.confidence}")
print(f"  Processing Time: {result.processing_time:.2f}s")

if result.sources:
    print(f"\n  Source References:")
    for i, source in enumerate(result.sources[:3], 1):
        print(f"    {i}. Page {source.page}: {source.content_preview[:80]}...")
