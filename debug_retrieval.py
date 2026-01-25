"""
Debug script to check RAG pipeline and vector store
"""

from app.services.vector_store import VectorStore
from app.services.rag_pipeline import get_pipeline

print("=" * 60)
print("RAG PIPELINE DEBUG")
print("=" *60)

# Check vector store
print("\n1. Checking Vector Store...")
try:
    vs = VectorStore()
    count = vs.collection.count()
    print(f"   ✅ Total documents in vector store: {count}")
    
    if count == 0:
        print("   ⚠️  WARNING: Vector store is EMPTY!")
        print("   This means no PDFs have been successfully processed.")
    else:
        # Try to peek at some documents
        results = vs.collection.peek(5)
        print(f"\n   Sample documents:")
        for i, doc_id in enumerate(results['ids'][:3]):
            print(f"   - ID: {doc_id}")
            if results['metadatas']:
                print(f"     Metadata: {results['metadatas'][i]}")
            
except Exception as e:
    print(f"   ❌ Error accessing vector store: {e}")

# Check pipeline
print("\n2. Checking RAG Pipeline...")
try:
    pipeline = get_pipeline()
    metrics = pipeline.get_metrics()
    print(f"   PDFs processed: {metrics['pdfs_processed']}")
    print(f"   Queries handled: {metrics['queries_handled']}")
    print(f"   Cache hits: {metrics['cache_hits']}")
    print(f"   Cache misses: {metrics['cache_misses']}")
except Exception as e:
    print(f"   ❌ Error accessing pipeline: {e}")

# Check data directories
print("\n3. Checking Data Directories...")
import os

uploads_dir = "./data/uploads"
extracted_dir = "./data/extracted"
chroma_db_dir = "./data/chroma_db"

for dir_path in [uploads_dir, extracted_dir, chroma_db_dir]:
    if os.path.exists(dir_path):
        files = os.listdir(dir_path)
        print(f"   {dir_path}: {len(files)} items")
        if dir_path == uploads_dir and files:
            print(f"     Files: {files[:5]}")
    else:
        print(f"   {dir_path}: DOES NOT EXIST")

print("\n" + "=" * 60)
print("DIAGNOSIS:")
print("=" * 60)

if count == 0:
    print("""
⚠️  ISSUE FOUND: Vector store is empty!

This means:
1. Either no PDF was uploaded
2. Or PDF processing failed silently
3. Or embeddings weren't stored

Troubleshooting steps:
1. Check server logs for processing errors
2. Upload a PDF and check processing status
3. Look for errors in PDF parsing or embedding generation
""")
else:
    print(f"""
✅ Vector store has {count} documents.

If queries are still failing, the issue might be:
1. Query embedding generation
2. Context aggregation
3. LLM prompt formatting
""")

print("=" * 60)
