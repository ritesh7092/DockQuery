"""
Upload a PDF to the RAG system via API
"""
import requests
from pathlib import Path
import time

API_URL = "http://localhost:8000/api/v1"

def upload_pdf(pdf_path):
    """Upload a PDF and wait for processing"""
    
    pdf_file = Path(pdf_path)
    
    if not pdf_file.exists():
        print(f"❌ Error: File not found: {pdf_path}")
        return None
    
    print(f"📄 Uploading: {pdf_file.name}")
    print(f"   Size: {pdf_file.stat().st_size / 1024:.2f} KB")
    
    try:
        # Upload the PDF
        with open(pdf_file, 'rb') as f:
            files = {'file': (pdf_file.name, f, 'application/pdf')}
            response = requests.post(f"{API_URL}/upload", files=files)
        
        if response.status_code != 200:
            print(f"❌ Upload failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
        
        result = response.json()
        pdf_id = result['pdf_id']
        
        print(f"✅ Upload successful!")
        print(f"   PDF ID: {pdf_id}")
        print(f"   Status: {result['status']}")
        
        # Poll for processing status
        print("\n⏳ Processing PDF...")
        max_wait = 300  # 5 minutes
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            status_response = requests.get(f"{API_URL}/status/{pdf_id}")
            
            if status_response.status_code != 200:
                print(f"❌ Status check failed: {status_response.status_code}")
                break
            
            status_data = status_response.json()
            current_status = status_data['status']
            
            if current_status == 'completed':
                print(f"\n✅ Processing complete!")
                print(f"   Total pages: {status_data.get('total_pages', 'N/A')}")
                print(f"   Text chunks: {status_data.get('text_chunks', 'N/A')}")
                print(f"   Images: {status_data.get('images_extracted', 'N/A')}")
                print(f"   Processing time: {status_data.get('processing_time', 'N/A'):.2f}s")
                return pdf_id
            
            elif current_status == 'failed':
                print(f"\n❌ Processing failed!")
                if 'errors' in status_data:
                    print(f"   Errors: {status_data['errors']}")
                return None
            
            elif current_status in ['pending', 'processing']:
                print(f"   Status: {current_status}...", end='\r')
                time.sleep(2)
            
        print(f"\n⚠️  Processing timeout after {max_wait}s")
        return None
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def query_pdf(pdf_id, query):
    """Query the uploaded PDF"""
    
    print(f"\n🔍 Querying: {query}")
    
    try:
        payload = {
            "query": query,
            "pdf_id": pdf_id,
            "top_k": 5,
            "include_images": True,
            "stream": False
        }
        
        response = requests.post(f"{API_URL}/query", json=payload)
        
        if response.status_code != 200:
            print(f"❌ Query failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return
        
        result = response.json()
        
        print(f"\n📝 Summary:")
        print(f"   {result['summary']}")
        
        print(f"\n📚 Sources ({len(result.get('sources', []))}):")
        for i, source in enumerate(result.get('sources', [])[:3], 1):
            print(f"   {i}. Page {source['page']} ({source['type']}): {source['content_preview'][:80]}...")
        
        print(f"\n⏱️  Processing time: {result['processing_time_ms']:.0f}ms")
        print(f"   Cached: {result.get('cached', False)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Upload the PDF
    pdf_path = r"d:\CodeBase\DockQuery\Annual Report_2022-23_English1.pdf"
    
    print("=" * 70)
    print("PDF UPLOAD & QUERY TEST")
    print("=" * 70)
    
    pdf_id = upload_pdf(pdf_path)
    
    if pdf_id:
        # Query the PDF
        query_pdf(pdf_id, "What are the main topics covered?")
    
    print("\n" + "=" * 70)
