"""Simple vector store check"""
from app.services.vector_store import VectorStore

vs = VectorStore()
vs.initialize_collection("multimodal_rag", reset=False)
collection = vs.collections.get("multimodal_rag")

if collection:
    count = collection.count()
    print(f"Documents in vector store: {count}")

    if count > 0:
        print("\n✅ Vector store has documents!")
        results = collection.peek(1)
        if results and results['ids']:
            print(f"Sample ID: {results['ids'][0]}")
            if results['metadatas']:
                print(f"Sample metadata: {results['metadatas'][0]}")
    else:
        print("\n❌ Vector store is still empty - PDF not processed yet")
else:
    print("\n❌ Collection not found")
