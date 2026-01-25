"""
End-to-End Workflow Test

Tests the complete workflow:
1. System health check
2. PDF upload
3. Processing status tracking
4. Multiple queries
5. Image retrieval

Usage:
    python tests/test_e2e_workflow.py
    
    # With custom PDF
    python tests/test_e2e_workflow.py --pdf path/to/your/file.pdf
"""

import requests
import time
import sys
import argparse
from pathlib import Path
from typing import Dict, Any

BASE_URL = "http://localhost:8000"

class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_step(step_num: int, message: str):
    """Print a test step header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}")
    print(f"Step {step_num}: {message}")
    print(f"{'='*60}{Colors.RESET}")

def print_success(message: str):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {message}{Colors.RESET}")

def print_error(message: str):
    """Print error message"""
    print(f"{Colors.RED}❌ {message}{Colors.RESET}")

def print_info(message: str):
    """Print info message"""
    print(f"   {message}")

def test_health_check() -> bool:
    """Test 1: System health check"""
    print_step(1, "System Health Check")
    
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        
        if resp.status_code != 200:
            print_error(f"Health check failed with status {resp.status_code}")
            return False
        
        data = resp.json()
        print_info(f"Status: {data.get('status')}")
        print_info(f"Version: {data.get('version')}")
        print_success("System is healthy")
        return True
        
    except requests.RequestException as e:
        print_error(f"Failed to connect to server: {e}")
        print_info("Make sure the server is running:")
        print_info("  uvicorn app.main:app --reload --port 8000")
        return False

def test_system_metrics() -> bool:
    """Test 2: System metrics check"""
    print_step(2, "System Metrics")
    
    try:
        resp = requests.get(f"{BASE_URL}/metrics", timeout=5)
        
        if resp.status_code != 200:
            print_error(f"Metrics check failed with status {resp.status_code}")
            return False
        
        data = resp.json()
        
        # Check services
        services = data.get('services', {})
        print_info("Services:")
        for service, status in services.items():
            print_info(f"  {service}: {status}")
        
        # Check storage
        storage = data.get('storage', {})
        print_info("Storage:")
        for store, status in storage.items():
            print_info(f"  {store}: {status}")
        
        print_success("System metrics retrieved")
        return True
        
    except requests.RequestException as e:
        print_error(f"Failed to get metrics: {e}")
        return False

def test_upload_pdf(pdf_path: str) -> str:
    """Test 3: Upload PDF"""
    print_step(3, f"Upload PDF: {pdf_path}")
    
    pdf_file = Path(pdf_path)
    
    if not pdf_file.exists():
        print_error(f"PDF file not found: {pdf_path}")
        return None
    
    file_size_mb = pdf_file.stat().st_size / (1024 * 1024)
    print_info(f"File size: {file_size_mb:.2f} MB")
    
    try:
        with open(pdf_file, 'rb') as f:
            files = {'file': (pdf_file.name, f, 'application/pdf')}
            
            print_info("Uploading...")
            start_time = time.time()
            resp = requests.post(
                f"{BASE_URL}/api/v1/upload",
                files=files,
                timeout=30
            )
            upload_time = time.time() - start_time
        
        if resp.status_code != 200:
            print_error(f"Upload failed with status {resp.status_code}")
            print_info(f"Response: {resp.text}")
            return None
        
        data = resp.json()
        pdf_id = data.get('pdf_id')
        
        print_success(f"Upload successful in {upload_time:.2f}s")
        print_info(f"PDF ID: {pdf_id}")
        print_info(f"Filename: {data.get('filename')}")
        print_info(f"Status: {data.get('status')}")
        
        return pdf_id
        
    except requests.RequestException as e:
        print_error(f"Upload failed: {e}")
        return None

def test_processing_status(pdf_id: str, max_wait: int = 120) -> Dict[str, Any]:
    """Test 4: Track processing status"""
    print_step(4, "Track Processing Status")
    
    print_info(f"Waiting for processing (max {max_wait}s)...")
    start_time = time.time()
    last_progress = -1
    
    while time.time() - start_time < max_wait:
        try:
            resp = requests.get(
                f"{BASE_URL}/api/v1/upload/{pdf_id}/status",
                timeout=5
            )
            
            if resp.status_code != 200:
                print_error(f"Status check failed: {resp.status_code}")
                return None
            
            data = resp.json()
            status = data.get('status')
            progress = data.get('progress', 0)
            
            # Only print when progress changes
            if progress != last_progress:
                print_info(f"Status: {status} - Progress: {progress*100:.0f}%")
                last_progress = progress
            
            if status == 'completed':
                elapsed = time.time() - start_time
                print_success(f"Processing completed in {elapsed:.2f}s")
                
                result = data.get('result', {})
                print_info(f"Total pages: {result.get('total_pages')}")
                print_info(f"Text blocks: {result.get('text_blocks_extracted')}")
                print_info(f"Images extracted: {result.get('images_extracted')}")
                print_info(f"Tables extracted: {result.get('tables_extracted')}")
                print_info(f"Total embeddings: {result.get('total_embeddings')}")
                
                return result
            
            elif status == 'failed':
                print_error(f"Processing failed: {data.get('error')}")
                return None
            
            time.sleep(2)
            
        except requests.RequestException as e:
            print_error(f"Status check error: {e}")
            time.sleep(2)
    
    print_error(f"Processing timed out after {max_wait}s")
    return None

def test_queries(pdf_id: str = None) -> bool:
    """Test 5: Execute queries"""
    print_step(5, "Execute Queries")
    
    queries = [
        {
            "query": "Summarize this document in 3 bullet points",
            "description": "Summary query"
        },
        {
            "query": "What are the main topics discussed?",
            "description": "Topic extraction"
        },
        {
            "query": "List any key findings or conclusions",
            "description": "Key findings"
        }
    ]
    
    if pdf_id:
        queries.append({
            "query": "What is on the first page?",
            "pdf_id": pdf_id,
            "description": "Specific page query"
        })
    
    all_success = True
    
    for i, query_data in enumerate(queries, 1):
        query = query_data.pop('query')
        description = query_data.pop('description')
        
        print(f"\n{Colors.YELLOW}Query {i}: {description}{Colors.RESET}")
        print_info(f'"{query}"')
        
        try:
            payload = {"query": query, **query_data}
            
            start_time = time.time()
            resp = requests.post(
                f"{BASE_URL}/api/v1/query",
                json=payload,
                timeout=30
            )
            query_time = time.time() - start_time
            
            if resp.status_code != 200:
                print_error(f"Query failed with status {resp.status_code}")
                print_info(f"Response: {resp.text}")
                all_success = False
                continue
            
            data = resp.json()
            
            answer = data.get('answer', '')
            sources = data.get('sources', [])
            images = data.get('images', [])
            
            print_success(f"Query completed in {query_time:.2f}s")
            print_info(f"Answer preview: {answer[:150]}...")
            print_info(f"Sources found: {len(sources)}")
            print_info(f"Images found: {len(images)}")
            
            if sources:
                print_info("Top source:")
                top_source = sources[0]
                print_info(f"  Page: {top_source.get('page')}")
                print_info(f"  Confidence: {top_source.get('confidence', 0):.2f}")
            
        except requests.RequestException as e:
            print_error(f"Query error: {e}")
            all_success = False
    
    return all_success

def test_image_retrieval(pdf_id: str) -> bool:
    """Test 6: Image retrieval"""
    print_step(6, "Test Image Retrieval")
    
    # First, query to get image IDs
    try:
        resp = requests.post(
            f"{BASE_URL}/api/v1/query",
            json={
                "query": "Show me any diagrams or charts",
                "pdf_id": pdf_id,
                "include_images": True
            },
            timeout=30
        )
        
        if resp.status_code != 200:
            print_info("No images to test retrieval")
            return True
        
        data = resp.json()
        images = data.get('images', [])
        
        if not images:
            print_info("No images found in document")
            return True
        
        # Try to retrieve first image
        first_image = images[0]
        image_url = first_image.get('url')
        
        print_info(f"Retrieving image: {image_url}")
        
        resp = requests.get(f"{BASE_URL}{image_url}", timeout=10)
        
        if resp.status_code != 200:
            print_error(f"Image retrieval failed: {resp.status_code}")
            return False
        
        print_success(f"Image retrieved ({len(resp.content)} bytes)")
        print_info(f"Content-Type: {resp.headers.get('Content-Type')}")
        
        return True
        
    except requests.RequestException as e:
        print_error(f"Image retrieval error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='End-to-End RAG Pipeline Test')
    parser.add_argument(
        '--pdf',
        default='tests/fixtures/sample.pdf',
        help='Path to PDF file for testing'
    )
    parser.add_argument(
        '--skip-upload',
        action='store_true',
        help='Skip upload test (use existing PDFs)'
    )
    
    args = parser.parse_args()
    
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║         Multimodal RAG - End-to-End Test Suite            ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(Colors.RESET)
    
    results = {}
    
    # Test 1: Health Check
    results['health'] = test_health_check()
    if not results['health']:
        print_error("\nFailed at health check. Aborting tests.")
        sys.exit(1)
    
    # Test 2: System Metrics
    results['metrics'] = test_system_metrics()
    
    pdf_id = None
    
    if not args.skip_upload:
        # Test 3: Upload PDF
        pdf_id = test_upload_pdf(args.pdf)
        results['upload'] = pdf_id is not None
        
        if not pdf_id:
            print_error("\nFailed at upload. Skipping remaining tests.")
            print_summary(results)
            sys.exit(1)
        
        # Test 4: Processing Status
        processing_result = test_processing_status(pdf_id)
        results['processing'] = processing_result is not None
        
        if not processing_result:
            print_error("\nFailed at processing. Skipping query tests.")
            print_summary(results)
            sys.exit(1)
    
    # Test 5: Queries
    results['queries'] = test_queries(pdf_id)
    
    # Test 6: Image Retrieval
    if pdf_id:
        results['images'] = test_image_retrieval(pdf_id)
    
    # Print summary
    print_summary(results)
    
    # Exit with appropriate code
    if all(results.values()):
        sys.exit(0)
    else:
        sys.exit(1)

def print_summary(results: Dict[str, bool]):
    """Print test summary"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}")
    print("Test Summary")
    print(f"{'='*60}{Colors.RESET}\n")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    for test, passed_status in results.items():
        status_icon = "✅" if passed_status else "❌"
        status_text = f"{Colors.GREEN}PASSED{Colors.RESET}" if passed_status else f"{Colors.RED}FAILED{Colors.RESET}"
        print(f"{status_icon} {test.upper()}: {status_text}")
    
    print(f"\n{Colors.BOLD}Total: {passed}/{total} tests passed{Colors.RESET}")
    
    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 All tests passed!{Colors.RESET}")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}⚠️  Some tests failed{Colors.RESET}")

if __name__ == "__main__":
    main()
