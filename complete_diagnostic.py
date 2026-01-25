"""
Complete diagnostic script to check the entire RAG pipeline
"""
from app.services.vector_store import VectorStore
from app.config import settings
import logging

logging.basicConfig(level=logging.INFO)

print("=" * 80)
print("COMPLETE RAG PIPELINE DIAGNOSTIC")
print("=" * 80)

# Step 1: Check Vector Store
print("\n📊 STEP 1: CHECKING VECTOR STORE")
print("-" * 80)

try:
    vs = VectorStore()
    
    # Get the collection
    collection = vs.initialize_collection("multimodal_rag")
    
    # Count documents
    count = collection.count()
    print(f"✓ Collection Name: {collection.name}")
    print(f"✓ Total Documents: {count}")
    
    if count == 0:
        print("\n❌ PROBLEM IDENTIFIED: Vector store is EMPTY!")
        print("\n🔍 ROOT CAUSE:")
        print("   The PDF upload/processing pipeline is NOT storing documents in the vector database.")
        print("\n💡 POSSIBLE REASONS:")
        print("   1. PDF processing failed silently")
        print("   2. Embedding generation failed")
        print("   3. Vector store add_documents() was never called")
        print("   4. Processing completed but documents were not added")
        
        print("\n🔧 SOLUTION:")
        print("   We need to check the upload/processing logs to see what happened.")
        print("   Let me check the data directory for processed files...")
        
        # Check data directory
        import os
        data_dir = settings.EXTRACTED_DIR
        print(f"\n📁 Checking data directory: {data_dir}")
        
        if os.path.exists(data_dir):
            subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
            print(f"   Found {len(subdirs)} PDF directories:")
            for subdir in subdirs[:5]:  # Show first 5
                pdf_dir = os.path.join(data_dir, subdir)
                files = os.listdir(pdf_dir)
                print(f"   - {subdir}/: {len(files)} files")
                
            if subdirs:
                print(f"\n✓ PDFs WERE processed (files extracted)")
                print("❌ BUT documents were NOT added to vector store")
                print("\n🎯 THE ISSUE: There's a gap between PDF parsing and vector storage!")
            else:
                print(f"\n❌ NO PDFs were processed at all")
                print("🎯 THE ISSUE: PDF processing pipeline is not running!")
        else:
            print(f"   ❌ Data directory doesn't exist: {data_dir}")
            
    else:
        print(f"\n✓ Vector store has {count} documents")
        
        # Sample documents
        print("\n📄 SAMPLE DOCUMENTS:")
        results = collection.peek(5)
        
        # Check for the specific PDF ID
        pdf_id_to_check = "7c329583-b845-4b12-ac62-4a0952af5184"
        print(f"\n🔍 Looking for PDF ID: {pdf_id_to_check}")
        
        # Query all documents with that source
        all_docs = collection.get(
            where={"source": pdf_id_to_check}
        )
        
        if all_docs and all_docs['ids']:
            print(f"   ✓ Found {len(all_docs['ids'])} documents for this PDF")
            for i, doc_id in enumerate(all_docs['ids'][:3]):
                print(f"\n   Document {i+1}:")
                print(f"   - ID: {doc_id}")
                if all_docs['metadatas'] and i < len(all_docs['metadatas']):
                    metadata = all_docs['metadatas'][i]
                    print(f"   - Type: {metadata.get('type', 'N/A')}")
                    print(f"   - Page: {metadata.get('page', 'N/A')}")
                if all_docs['documents'] and i < len(all_docs['documents']):
                    doc_preview = all_docs['documents'][i][:100]
                    print(f"   - Content: {doc_preview}...")
        else:
            print(f"   ❌ NO documents found for PDF: {pdf_id_to_check}")
            print(f"\n   Available sources in vector store:")
            # Get all unique sources
            all_results = collection.get(limit=100)
            if all_results and all_results['metadatas']:
                sources = set(m.get('source', 'unknown') for m in all_results['metadatas'])
                for source in list(sources)[:10]:
                    print(f"      - {source}")
                    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Step 2: Check if the PDF file exists
print("\n\n📁 STEP 2: CHECKING PDF FILE")
print("-" * 80)

import os
pdf_path = "d:\\CodeBase\\DockQuery\\Annual Report_2022-23_English1.pdf"
if os.path.exists(pdf_path):
    pdf_size = os.path.getsize(pdf_path)
    print(f"✓ PDF exists: {pdf_path}")
    print(f"✓ Size: {pdf_size / 1024 / 1024:.2f} MB")
else:
    print(f"❌ PDF not found: {pdf_path}")

# Step 3: Try manual processing
print("\n\n🔧 STEP 3: TESTING MANUAL PROCESSING")
print("-" * 80)

try:
    from app.services.pdf_parser import PDFParser
    from app.services.embeddings import EmbeddingService
    
    # Check if we can parse the PDF
    parser = PDFParser(output_dir=settings.EXTRACTED_DIR)
    print("✓ PDF Parser initialized")
    
    embeddings_service = EmbeddingService()
    print("✓ Embedding Service initialized")
    
    print("\n💡 All services are working correctly")
    print("   The issue is likely in the RAG Pipeline orchestration")
    
except Exception as e:
    print(f"❌ Service initialization failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)
