"""
Detailed check of what went wrong
"""
from app.services.vector_store import VectorStore
import logging

logging.basicConfig(level=logging.INFO)

pdf_id = "7c329583-b845-4b12-ac62-4a0952af5184"

print("=" * 80)
print("DETAILED VECTOR STORE CHECK")
print("=" * 80)

vs = VectorStore()
collection = vs.initialize_collection("multimodal_rag")

print(f"\nTotal documents in collection: {collection.count()}")

# Try to get documents for this PDF
print(f"\nSearching for documents with source='{pdf_id}'...")
results = collection.get(
    where={"source": pdf_id},
    limit=10
)

if results and results['ids']:
    print(f"\nFOUND {len(results['ids'])} documents!")
    print("\nFirst 3 documents:")
    for i in range(min(3, len(results['ids']))):
        print(f"\n  Document {i+1}:")
        print(f"    ID: {results['ids'][i]}")
        if results['metadatas'] and i < len(results['metadatas']):
            meta = results['metadatas'][i]
            print(f"    Type: {meta.get('type')}")
            print(f"    Page: {meta.get('page')}")
            print(f"    Source: {meta.get('source')}")
        if results['documents'] and i < len(results['documents']):
            content = results['documents'][i]
            print(f"    Content (first 100 chars): {content[:100]}")
else:
    print("\nNO DOCUMENTS FOUND!")
    
    # Check what sources ARE in the vector store
    print("\nLet's see what sources ARE in the vector store...")
    all_results = collection.get(limit=100)
    
    if all_results and all_results['metadatas']:
        sources = set()
        for meta in all_results['metadatas']:
            if meta and 'source' in meta:
                sources.add(meta['source'])
        
        print(f"\nFound {len(sources)} unique sources:")
        for source in sorted(sources):
            # Count documents for this source
            source_results = collection.get(where={"source": source})
            count = len(source_results['ids']) if source_results and source_results['ids'] else 0
            print(f"  - {source}: {count} documents")
    else:
        print("\nVector store appears to be completely empty!")

print("\n" + "=" * 80)
