"""
Debug script to check vector store contents
"""
from app.services.vector_store import VectorStore
import logging

logging.basicConfig(level=logging.INFO)

print("=" * 60)
print("VECTOR STORE DIAGNOSTIC")
print("=" * 60)

try:
    vs = VectorStore()
    
    # Check collection
    print(f"\n1. Collection Info:")
    print(f"   Name: {vs.collection.name}")
    
    # Count documents
    count = vs.collection.count()
    print(f"   Total documents: {count}")
    
    if count == 0:
        print("\n⚠️  ISSUE: Vector store is EMPTY!")
        print("   This means no PDF has been successfully processed.")
        print("\n   You need to:")
        print("   1. Upload your PDF via the web interface (http://localhost:8000)")
        print("   2. Wait for processing to complete")
        print("   3. Then try querying again")
    else:
        print(f"\n✓ Vector store has {count} documents")
        
        # Peek at first few documents
        print("\n2. Sample Documents:")
        results = vs.collection.peek(3)
        
        for i, doc_id in enumerate(results['ids']):
            print(f"\n   Document {i+1}:")
            print(f"   - ID: {doc_id}")
            if results['metadatas'] and i < len(results['metadatas']):
                metadata = results['metadatas'][i]
                print(f"   - Type: {metadata.get('type', 'N/A')}")
                print(f"   - Page: {metadata.get('page', 'N/A')}")
                print(f"   - Source: {metadata.get('source', 'N/A')}")
            if results['documents'] and i < len(results['documents']):
                doc_preview = results['documents'][i][:100]
                print(f"   - Content: {doc_preview}...")
        
        # Try a test query
        print("\n3. Test Query:")
        import numpy as np
        test_embedding = np.random.rand(384).astype(np.float32)  # Random embedding for testing
        
        try:
            test_results = vs.collection.query(
                query_embeddings=[test_embedding.tolist()],
                n_results=1
            )
            print(f"   ✓ Query executed successfully")
            print(f"   Results returned: {len(test_results['ids'][0]) if test_results['ids'] else 0}")
        except Exception as e:
            print(f"   ❌ Query failed: {e}")
            
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
