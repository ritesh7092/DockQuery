"""
Interactive Testing Tool for Multimodal RAG

A user-friendly interactive script to test your RAG pipeline step by step.

Usage:
    python tests/interactive_test.py
"""

import requests
import time
import sys
from pathlib import Path
from typing import Optional

BASE_URL = "http://localhost:8000"

def clear_screen():
    """Clear the terminal screen"""
    print("\n" * 2)

def print_header(title: str):
    """Print a formatted header"""
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)

def print_menu(title: str, options: list):
    """Print a menu with options"""
    print(f"\n{title}")
    print("-" * 40)
    for i, option in enumerate(options, 1):
        print(f"  {i}. {option}")
    print(f"  {len(options) + 1}. Back")
    print(f"  0. Exit")

def get_choice(max_choice: int) -> int:
    """Get user's menu choice"""
    while True:
        try:
            choice = input(f"\nEnter your choice (0-{max_choice}): ").strip()
            choice = int(choice)
            if 0 <= choice <= max_choice:
                return choice
            print(f"Please enter a number between 0 and {max_choice}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            sys.exit(0)

def test_connection():
    """Test connection to the server"""
    print_header("Testing Server Connection")
    
    try:
        print("\nConnecting to server...")
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        
        if resp.status_code == 200:
            data = resp.json()
            print(f"✅ Server is running!")
            print(f"   Status: {data.get('status')}")
            print(f"   Version: {data.get('version')}")
            return True
        else:
            print(f"❌ Server returned status code: {resp.status_code}")
            return False
            
    except requests.RequestException as e:
        print(f"❌ Cannot connect to server: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure the server is running:")
        print("     uvicorn app.main:app --reload --port 8000")
        print("  2. Check if port 8000 is available")
        print("  3. Verify your .env file is configured")
        return False

def upload_pdf_interactive():
    """Interactive PDF upload"""
    print_header("Upload PDF")
    
    print("\nEnter the path to your PDF file:")
    print("(or press Enter for a test file)")
    
    pdf_path = input("Path: ").strip()
    
    if not pdf_path:
        # Check common test locations
        test_paths = [
            "tests/fixtures/sample.pdf",
            "data/sample.pdf",
            "sample.pdf"
        ]
        for path in test_paths:
            if Path(path).exists():
                pdf_path = path
                break
        
        if not pdf_path or not Path(pdf_path).exists():
            print("\n❌ No test file found. Please provide a PDF path.")
            return None
    
    pdf_file = Path(pdf_path)
    
    if not pdf_file.exists():
        print(f"\n❌ File not found: {pdf_path}")
        return None
    
    file_size_mb = pdf_file.stat().st_size / (1024 * 1024)
    
    if file_size_mb > 50:
        print(f"\n❌ File too large: {file_size_mb:.2f} MB (max 50 MB)")
        return None
    
    print(f"\n📄 File: {pdf_file.name}")
    print(f"📊 Size: {file_size_mb:.2f} MB")
    
    confirm = input("\nUpload this file? (y/n): ").strip().lower()
    
    if confirm != 'y':
        print("Upload cancelled")
        return None
    
    try:
        print("\n⏳ Uploading...")
        with open(pdf_file, 'rb') as f:
            files = {'file': (pdf_file.name, f, 'application/pdf')}
            resp = requests.post(f"{BASE_URL}/api/v1/upload", files=files, timeout=30)
        
        if resp.status_code != 200:
            print(f"\n❌ Upload failed: {resp.status_code}")
            print(f"Response: {resp.text}")
            return None
        
        data = resp.json()
        pdf_id = data.get('pdf_id')
        
        print(f"\n✅ Upload successful!")
        print(f"   PDF ID: {pdf_id}")
        print(f"   Status: {data.get('status')}")
        
        # Wait for processing
        print("\n⏳ Processing PDF...")
        processing_result = wait_for_processing(pdf_id)
        
        if processing_result:
            print("\n✅ Processing complete!")
            print(f"   Pages: {processing_result.get('total_pages')}")
            print(f"   Images: {processing_result.get('images_extracted')}")
            print(f"   Tables: {processing_result.get('tables_extracted')}")
        
        return pdf_id
        
    except requests.RequestException as e:
        print(f"\n❌ Upload error: {e}")
        return None

def wait_for_processing(pdf_id: str, max_wait: int = 120) -> Optional[dict]:
    """Wait for PDF processing to complete"""
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            resp = requests.get(f"{BASE_URL}/api/v1/upload/{pdf_id}/status", timeout=5)
            
            if resp.status_code != 200:
                return None
            
            data = resp.json()
            status = data.get('status')
            progress = data.get('progress', 0)
            
            print(f"\r   Progress: {progress*100:.0f}%", end='', flush=True)
            
            if status == 'completed':
                print()  # New line
                return data.get('result', {})
            elif status == 'failed':
                print(f"\n❌ Processing failed: {data.get('error')}")
                return None
            
            time.sleep(2)
            
        except requests.RequestException:
            time.sleep(2)
    
    print(f"\n❌ Processing timed out")
    return None

def query_interactive(pdf_id: Optional[str] = None):
    """Interactive query execution"""
    print_header("Execute Query")
    
    # Pre-defined query templates
    templates = [
        "Summarize this document in 3 bullet points",
        "What are the main topics discussed?",
        "List any key findings or conclusions",
        "What methodology was used?",
        "Describe the charts and diagrams (with images)",
        "Custom query (you type)"
    ]
    
    print("\nSelect a query template:")
    for i, template in enumerate(templates, 1):
        print(f"  {i}. {template}")
    
    choice = get_choice(len(templates))
    
    if choice == 0:
        return
    
    if choice == len(templates):
        # Custom query
        query = input("\nEnter your query: ").strip()
        if not query:
            print("Query cannot be empty")
            return
        include_images = input("Include images? (y/n): ").strip().lower() == 'y'
    else:
        query = templates[choice - 1]
        include_images = "images" in query.lower()
        query = query.replace(" (with images)", "")
    
    # Build request
    payload = {"query": query}
    
    if pdf_id:
        use_pdf_id = input(f"\nQuery specific PDF {pdf_id[:8]}...? (y/n): ").strip().lower()
        if use_pdf_id == 'y':
            payload["pdf_id"] = pdf_id
    
    if include_images:
        payload["include_images"] = True
    
    # Advanced options
    advanced = input("\nConfigure advanced options? (y/n): ").strip().lower()
    if advanced == 'y':
        try:
            top_k = input("Number of results to retrieve (default 5): ").strip()
            if top_k:
                payload["top_k"] = int(top_k)
        except ValueError:
            print("Invalid number, using default")
    
    # Execute query
    print(f"\n⏳ Executing query...")
    print(f"   Query: {query}")
    
    try:
        start_time = time.time()
        resp = requests.post(f"{BASE_URL}/api/v1/query", json=payload, timeout=30)
        query_time = time.time() - start_time
        
        if resp.status_code != 200:
            print(f"\n❌ Query failed: {resp.status_code}")
            print(f"Response: {resp.text}")
            return
        
        data = resp.json()
        
        print(f"\n✅ Query completed in {query_time:.2f}s")
        print("\n" + "=" * 60)
        print("ANSWER:")
        print("=" * 60)
        print(data.get('answer', 'No answer'))
        
        sources = data.get('sources', [])
        if sources:
            print(f"\n📚 Sources ({len(sources)} found):")
            for i, source in enumerate(sources[:3], 1):  # Show top 3
                print(f"\n  {i}. Page {source.get('page')} (confidence: {source.get('confidence', 0):.2f})")
                content = source.get('content', '')
                print(f"     {content[:100]}...")
        
        images = data.get('images', [])
        if images:
            print(f"\n🖼️  Images ({len(images)} found):")
            for i, img in enumerate(images[:3], 1):  # Show top 3
                print(f"  {i}. {img.get('caption', 'No caption')} (page {img.get('page')})")
                print(f"     URL: {img.get('url')}")
        
        print("\n" + "=" * 60)
        
        # Option to save answer
        save = input("\nSave answer to file? (y/n): ").strip().lower()
        if save == 'y':
            filename = f"query_result_{int(time.time())}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Query: {query}\n\n")
                f.write(f"Answer:\n{data.get('answer', '')}\n\n")
                f.write(f"Sources: {len(sources)}\n")
                f.write(f"Images: {len(images)}\n")
            print(f"✅ Saved to {filename}")
        
    except requests.RequestException as e:
        print(f"\n❌ Query error: {e}")

def view_metrics():
    """View system metrics"""
    print_header("System Metrics")
    
    try:
        resp = requests.get(f"{BASE_URL}/metrics", timeout=5)
        
        if resp.status_code != 200:
            print(f"❌ Failed to get metrics: {resp.status_code}")
            return
        
        data = resp.json()
        
        print("\n📊 Services:")
        for service, status in data.get('services', {}).items():
            print(f"   {service}: {status}")
        
        print("\n💾 Storage:")
        for store, status in data.get('storage', {}).items():
            print(f"   {store}: {status}")
        
        print("\n⚙️  Configuration:")
        for key, value in data.get('config', {}).items():
            print(f"   {key}: {value}")
        
    except requests.RequestException as e:
        print(f"❌ Error: {e}")

def main():
    """Main interactive loop"""
    pdf_id = None
    
    while True:
        clear_screen()
        print_header("Multimodal RAG Interactive Testing Tool")
        
        if pdf_id:
            print(f"\n📄 Current PDF: {pdf_id[:8]}...")
        
        options = [
            "Test Server Connection",
            "View System Metrics",
            "Upload PDF",
            "Execute Query",
            "Clear Current PDF"
        ]
        
        print_menu("Main Menu", options)
        choice = get_choice(len(options) + 1)
        
        if choice == 0:
            print("\nGoodbye!")
            break
        
        elif choice == 1:
            test_connection()
            input("\nPress Enter to continue...")
        
        elif choice == 2:
            view_metrics()
            input("\nPress Enter to continue...")
        
        elif choice == 3:
            new_pdf_id = upload_pdf_interactive()
            if new_pdf_id:
                pdf_id = new_pdf_id
            input("\nPress Enter to continue...")
        
        elif choice == 4:
            query_interactive(pdf_id)
            input("\nPress Enter to continue...")
        
        elif choice == 5:
            pdf_id = None
            print("\n✅ PDF cleared")
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting...")
        sys.exit(0)
