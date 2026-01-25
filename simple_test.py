"""Simple diagnostic - check what's happening"""
from app.services.rag_pipeline import get_pipeline

print("Testing full pipeline query...")

pipeline = get_pipeline()

result = pipeline.query(
    query="What are the main topics covered?",
    pdf_id=None,  # No filter!
    top_k=5,
    include_visuals=True
)

print(f"\n====== RESULT ======")
print(f"Sources: {len(result.sources)}")
print(f"Confidence: {result.confidence}")
print(f"Time: {result.processing_time:.2f}s")

print(f"\n====== SUMMARY ======")
print(result.summary)

print(f"\n====== DIAGNOSIS ======")
if "No text context available" in result.summary:
    print("ERROR: Still getting 'No text context available'")
    print("\nThis means the retrieval is working BUT something is wrong")
    print("between retrieval and prompt building.")
else:
    print("SUCCESS: Proper summary generated!")
