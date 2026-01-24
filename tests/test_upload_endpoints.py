"""
Test script for upload endpoints.

This script demonstrates how to test the upload and status endpoints.
"""

import requests
import time
import json
from pathlib import Path


BASE_URL = "http://localhost:8000/api/v1"


def test_upload_pdf(pdf_path: str):
    """Test uploading a PDF file."""
    print(f"\n{'='*60}")
    print("TEST 1: Upload Valid PDF")
    print(f"{'='*60}")
    
    if not Path(pdf_path).exists():
        print(f"❌ File not found: {pdf_path}")
        return None
    
    with open(pdf_path, 'rb') as f:
        files = {'file': (Path(pdf_path).name, f, 'application/pdf')}
        
        print(f"📤 Uploading: {pdf_path}")
        response = requests.post(f"{BASE_URL}/upload", files=files)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            pdf_id = response.json()['pdf_id']
            print(f"✅ Upload successful! PDF ID: {pdf_id}")
            return pdf_id
        else:
            print(f"❌ Upload failed!")
            return None


def test_get_status(pdf_id: str):
    """Test getting upload status."""
    print(f"\n{'='*60}")
    print("TEST 2: Get Upload Status")
    print(f"{'='*60}")
    
    print(f"📊 Checking status for: {pdf_id}")
    response = requests.get(f"{BASE_URL}/upload/{pdf_id}/status")
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    if response.status_code == 200:
        status_data = response.json()
        print(f"✅ Status: {status_data['status']} - Progress: {status_data['progress']*100:.1f}%")
        return status_data
    else:
        print(f"❌ Failed to get status!")
        return None


def test_invalid_file():
    """Test uploading a non-PDF file."""
    print(f"\n{'='*60}")
    print("TEST 3: Upload Invalid File (Should Fail)")
    print(f"{'='*60}")
    
    # Create a fake PDF
    fake_content = b"This is not a PDF file"
    files = {'file': ('fake.pdf', fake_content, 'application/pdf')}
    
    print("📤 Uploading fake PDF...")
    response = requests.post(f"{BASE_URL}/upload", files=files)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    if response.status_code == 400:
        print("✅ Correctly rejected invalid PDF!")
    else:
        print("❌ Should have rejected invalid PDF!")


def test_rate_limit(pdf_path: str, count: int = 11):
    """Test rate limiting by uploading multiple files."""
    print(f"\n{'='*60}")
    print(f"TEST 4: Rate Limiting ({count} uploads)")
    print(f"{'='*60}")
    
    if not Path(pdf_path).exists():
        print(f"❌ File not found: {pdf_path}")
        return
    
    for i in range(count):
        with open(pdf_path, 'rb') as f:
            files = {'file': (f"test_{i}.pdf", f, 'application/pdf')}
            response = requests.post(f"{BASE_URL}/upload", files=files)
            
            if response.status_code == 429:
                print(f"📛 Upload {i+1}/{count}: Rate limit hit! (expected)")
                print(f"   Response: {json.dumps(response.json(), indent=2)}")
                return
            elif response.status_code == 200:
                print(f"✓ Upload {i+1}/{count}: Success")
            else:
                print(f"✗ Upload {i+1}/{count}: Failed with {response.status_code}")
    
    print("❌ Rate limit was not triggered!")


def test_oversized_file():
    """Test uploading a file that's too large."""
    print(f"\n{'='*60}")
    print("TEST 5: Upload Oversized File (Should Fail)")
    print(f"{'='*60}")
    
    # Create 51MB of data (exceeds 50MB limit)
    print("📦 Creating 51MB file...")
    large_content = b"%PDF-1.4\n" + (b"x" * (51 * 1024 * 1024))
    files = {'file': ('large.pdf', large_content, 'application/pdf')}
    
    print("📤 Uploading oversized file...")
    response = requests.post(f"{BASE_URL}/upload", files=files)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    if response.status_code == 413:
        print("✅ Correctly rejected oversized file!")
    else:
        print("❌ Should have rejected oversized file!")


def monitor_processing(pdf_id: str, max_wait: int = 60):
    """Monitor processing status until completion."""
    print(f"\n{'='*60}")
    print("Monitoring Processing Status")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        response = requests.get(f"{BASE_URL}/upload/{pdf_id}/status")
        
        if response.status_code == 200:
            status_data = response.json()
            status = status_data['status']
            progress = status_data['progress']
            
            print(f"⏳ Status: {status} - Progress: {progress*100:.1f}%", end='\r')
            
            if status in ['completed', 'failed']:
                print()  # New line
                print(f"\n{'='*60}")
                print(f"🎉 Processing {status.upper()}!")
                print(f"{'='*60}")
                print(f"Final Result: {json.dumps(status_data, indent=2)}")
                return status_data
        
        time.sleep(2)
    
    print(f"\n⏰ Timeout after {max_wait}s")
    return None


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 UPLOAD ENDPOINT TEST SUITE")
    print("="*60)
    print(f"Base URL: {BASE_URL}")
    print(f"Make sure the server is running!")
    print("="*60)
    
    # Get PDF path from user
    print("\n📁 Enter path to a test PDF file:")
    pdf_path = input("> ").strip().strip('"')
    
    if not Path(pdf_path).exists():
        print(f"❌ File not found: {pdf_path}")
        exit(1)
    
    # Run tests
    try:
        # Test 1: Valid upload
        pdf_id = test_upload_pdf(pdf_path)
        
        if pdf_id:
            # Test 2: Status check
            time.sleep(1)
            test_get_status(pdf_id)
            
            # Monitor until completion
            monitor_processing(pdf_id, max_wait=120)
        
        # Test 3: Invalid file
        test_invalid_file()
        
        # Test 4: Rate limiting
        test_rate_limit(pdf_path, count=11)
        
        # Test 5: Oversized file
        test_oversized_file()
        
        print(f"\n{'='*60}")
        print("✅ ALL TESTS COMPLETED!")
        print(f"{'='*60}\n")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to server!")
        print("Make sure the server is running:")
        print("  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
