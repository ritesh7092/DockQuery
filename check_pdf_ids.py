# -*- coding: utf-8 -*-
"""Check what PDF IDs are in the vector store"""
from app.services.vector_store import VectorStore

vs = VectorStore()
vs.initialize_collection("multimodal_rag", reset=False)
collection = vs.collections.get("multimodal_rag")

if collection:
    # Get all documents
    count = collection.count()
    print(f"Total documents: {count}")
    
    # Peek at documents to see what sources exist
    results = collection.peek(50)  # Get more samples
    
    if results and results['metadatas']:
        # Extract unique PDF sources
        sources = set()
        for metadata in results['metadatas']:
            if 'source' in metadata:
                sources.add(metadata['source'])
        
        print(f"\nUnique PDF IDs in vector store ({len(sources)}):")
        for source in sorted(sources):
            print(f"  - {source}")
        
        # Count documents per source
        print(f"\nDocuments per PDF ID:")
        for source in sorted(sources):
            source_count = sum(1 for m in results['metadatas'] if m.get('source') == source)
            print(f"  - {source}: {source_count}+ documents (sample)")
    else:
        print("No metadata found")
else:
    print("Collection not found")
