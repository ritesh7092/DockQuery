"""
Simple diagnostic script to check the RAG pipeline
"""
from app.services.vector_store import VectorStore
from app.config import settings
import logging
import os

logging.basicConfig(level=logging.WARNING)

print("=" * 80)
print("RAG PIPELINE DIAGNOSTIC")
print("=" * 80)

# Step 1: Check Vector Store
print("\nSTEP 1: CHECKING VECTOR STORE")
print("-" * 80)

try:
    vs = VectorStore()
    collection = vs.initialize_collection("multimodal_rag")
    count = collection.count()
    
    print(f"Collection Name: {collection.name}")
    print(f"Total Documents: {count}")
    
    if count == 0:
        print("\n** PROBLEM: Vector store is EMPTY! **")
        print("\nROOT CAUSE:")
        print("   PDF documents are NOT being stored in the vector database.")
        print("\nPOSSIBLE REASONS:")
        print("   1. PDF processing failed")
        print("   2. Embedding generation failed")
        print("   3. Documents were processed but not added to vector store")
        
        # Check data directory
        data_dir = settings.EXTRACTED_DIR
        print(f"\nChecking data directory: {data_dir}")
        
        if os.path.exists(data_dir):
            subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
            print(f"   Found {len(subdirs)} PDF directories")
            
            if subdirs:
                # Show details of first PDF
                for i, subdir in enumerate(subdirs[:3]):
                    pdf_dir = os.path.join(data_dir, subdir)
                    files = os.listdir(pdf_dir)
                    print(f"   PDF {i+1}: {subdir}/")
                    print(f"      Files extracted: {len(files)}")
                    
                print("\n** DIAGNOSIS: PDFs WERE processed, but NOT stored in vector DB **")
                print("** This means the RAG pipeline is not completing successfully **")
            else:
                print("\n** DIAGNOSIS: NO PDFs were processed at all **")
        else:
            print(f"   Data directory doesn't exist!")
    else:
        print(f"\n** Vector store has {count} documents **")
        
        # Check for specific PDF ID
        pdf_id_to_check = "7c329583-b845-4b12-ac62-4a0952af5184"
        print(f"\nLooking for PDF ID: {pdf_id_to_check}")
        
        all_docs = collection.get(where={"source": pdf_id_to_check})
        
        if all_docs and all_docs['ids']:
            print(f"   Found {len(all_docs['ids'])} documents for this PDF")
            print("\n** THIS PDF IS IN THE VECTOR STORE **")
        else:
            print(f"   NO documents found for this PDF ID")
            
            # Show what's actually in there
            all_results = collection.get(limit=10)
            if all_results and all_results['metadatas']:
                sources = set(m.get('source', 'unknown') for m in all_results['metadatas'])
                print(f"\n   Available PDF IDs in vector store:")
                for source in sources:
                    source_count = sum(1 for m in all_results['metadatas'] if m.get('source') == source)
                    print(f"      - {source} ({source_count} docs)")
                    
            print("\n** DIAGNOSIS: Wrong PDF ID or PDF not processed **")
            
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("END DIAGNOSTIC")
print("=" * 80)
