"""
Fix and reprocess PDF script - diagnoses and fixes vector store issues
"""
from app.services.vector_store import VectorStore
from app.services.rag_pipeline import get_pipeline
from app.config import settings
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("=" * 80)
print("FIX AND REPROCESS SCRIPT")
print("=" * 80)

# PDF ID to check
pdf_id = "7c329583-b845-4b12-ac62-4a0952af5184"

# Step 1: Check vector store
print(f"\nStep 1: Checking vector store for PDF: {pdf_id}")
print("-" * 80)

try:
    vs = VectorStore()
    collection = vs.initialize_collection("multimodal_rag")
    total_count = collection.count()
    
    print(f"Total documents in vector store: {total_count}")
    
    # Check for this specific PDF
    results = collection.get(where={"source": pdf_id})
    
    if results and results['ids']:
        print(f"*** Documents found for this PDF: {len(results['ids'])}")
        print("The documents ARE in the vector store!")
        print("\nThis means the issue might be with the query embedding or retrieval logic.")
    else:
        print(f"*** NO documents found for PDF: {pdf_id}")
        print("\n** ROOT CAUSE IDENTIFIED **")
        print("   PDF was parsed (files extracted) but documents were NOT added to vector store")
        print("\n** SOLUTION: Manually reprocess the PDF **")
        
        # Find the original PDF file
        pdf_file = f"d:\\CodeBase\\DockQuery\\Annual Report_2022-23_English1.pdf"
        
        if not os.path.exists(pdf_file):
            print(f"\n   ERROR: PDF file not found at: {pdf_file}")
        else:
            print(f"\n   Found PDF file: {pdf_file}")
            print("   Size: {:.2f} MB".format(os.path.getsize(pdf_file) / 1024 / 1024))
            
            # Reprocess the PDF
            print("\n   Reprocessing PDF through RAG pipeline...")
            pipeline = get_pipeline()
            
            result = pipeline.process_pdf(
                file_path=pdf_file,
 pdf_id=pdf_id,
                collection_name="multimodal_rag"
            )
            
            print(f"\n   Processing result:")
            print(f"      Status: {result.status}")
            print(f"      Pages: {result.total_pages}")
            print(f"      Text chunks: {result.text_chunks}")
            print(f"      Images: {result.images_extracted}")
            print(f"      Tables: {result.tables_found}")
            print(f"      Time: {result.processing_time:.2f}s")
            
            if result.errors:
                print(f"      Errors: {result.errors}")
            
            # Verify documents were added
            print("\n   Verifying documents were added...")
            results = collection.get(where={"source": pdf_id})
            
            if results and results['ids']:
                print(f"      SUCCESS! Added{len(results['ids'])} documents to vector store")
            else:
                print(f"      FAILED! Documents still not found in vector store")
                
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("SCRIPT COMPLETE")
print("=" * 80)
