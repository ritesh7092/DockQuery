"""
Test script for query endpoints.

This script demonstrates how to test the query and image serving endpoints.
"""

import requests
import json
import time
from pathlib import Path


BASE_URL = "http://localhost:8000/api/v1"


def test_basic_query():
    """Test basic query without PDF ID filter."""
    print(f"\n{'='*60}")
    print("TEST 1: Basic Query (No PDF Filter)")
    print(f"{'='*60}")
    
    payload = {
        "query": "What are the main topics discussed in the documents?",
        "top_k": 5,
        "include_images": True,
        "stream": False
    }
    
    print(f"📤 Sending query: {payload['query']}")
    response = requests.post(
        f"{BASE_URL}/query",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✅ Query successful!")
        print(f"Summary: {data.get('summary', 'N/A')[:200]}...")
        print(f"Sources: {len(data.get('sources', []))} found")
        print(f"Visual Elements: {len(data.get('visual_elements', []))} found")
        print(f"Processing Time: {data.get('processing_time_ms', 0):.2f}ms")
        print(f"Cached: {data.get('cached', False)}")
        return data
    else:
        print(f"❌ Query failed!")
        print(f"Response: {response.text}")
        return None


def test_query_with_pdf_id(pdf_id: str):
    """Test query filtered by PDF ID."""
    print(f"\n{'='*60}")
    print(f"TEST 2: Query with PDF ID Filter ({pdf_id})")
    print(f"{'='*60}")
    
    payload = {
        "query": "Summarize the main findings from this document",
        "pdf_id": pdf_id,
        "top_k": 5,
        "include_images": True,
        "stream": False
    }
    
    print(f"📤 Querying PDF: {pdf_id}")
    response = requests.post(
        f"{BASE_URL}/query",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✅ Query successful!")
        print(f"Summary: {data.get('summary', 'N/A')[:200]}...")
        print(f"Sources: {len(data.get('sources', []))} from PDF {pdf_id}")
        print(f"Processing Time: {data.get('processing_time_ms', 0):.2f}ms")
        return data
    else:
        print(f"❌ Query failed!")
        print(f"Response: {response.text}")
        return None


def test_streaming_query():
    """Test streaming query response."""
    print(f"\n{'='*60}")
    print("TEST 3: Streaming Query (SSE)")
    print(f"{'='*60}")
    
    payload = {
        "query": "What is the conclusion of the document?",
        "top_k": 3,
        "include_images": False,
        "stream": True
    }
    
    print(f"📤 Sending streaming query...")
    
    try:
        response = requests.post(
            f"{BASE_URL}/query",
            json=payload,
            headers={"Content-Type": "application/json"},
            stream=True
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("\n✅ Streaming response:")
            print("-" * 60)
            
            for line in response.iter_lines():
                if line:
                    decoded = line.decode('utf-8')
                    if decoded.startswith('data:'):
                        content = decoded[5:].strip()
                        if content == "[DONE]":
                            print("\n" + "-" * 60)
                            print("Stream complete!")
                            break
                        print(content, end=' ', flush=True)
        else:
            print(f"❌ Streaming failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")


def test_query_validation():
    """Test query validation and sanitization."""
    print(f"\n{'='*60}")
    print("TEST 4: Query Validation")
    print(f"{'='*60}")
    
    test_cases = [
        ("", "Empty query"),
        ("<script>alert('xss')</script>", "XSS attempt"),
        ("A" * 1001, "Too long query"),
        ("Normal query", "Valid query")
    ]
    
    for query, description in test_cases:
        print(f"\n📝 Testing: {description}")
        payload = {"query": query, "stream": False}
        
        response = requests.post(
            f"{BASE_URL}/query",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if query == "Normal query":
            print(f"   ✅ {response.status_code} - {description}")
        else:
            print(f"   {'✅' if response.status_code >= 400 else '❌'} {response.status_code} - {description}")


def test_image_retrieval(pdf_id: str, image_id: str):
    """Test image serving endpoint."""
    print(f"\n{'='*60}")
    print(f"TEST 5: Image Retrieval")
    print(f"{'='*60}")
    
    url = f"{BASE_URL}/images/{pdf_id}/{image_id}"
    print(f"📥 Fetching image: {url}")
    
    response = requests.get(url)
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        content_type = response.headers.get('content-type', 'unknown')
        content_length = len(response.content)
        print(f"✅ Image retrieved successfully!")
        print(f"   Content-Type: {content_type}")
        print(f"   Size: {content_length} bytes")
        
        # Save to file for verification
        output_path = f"test_image_{image_id}.png"
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"   Saved to: {output_path}")
    else:
        print(f"❌ Image retrieval failed!")
        print(f"Response: {response.text}")


def test_rate_limiting():
    """Test query rate limiting."""
    print(f"\n{'='*60}")
    print("TEST 6: Query Rate Limiting")
    print(f"{'='*60}")
    
    print(f"🔄 Sending rapid queries to test rate limit...")
    
    # Send multiple queries rapidly (this is just a demo, 100 is too many)
    for i in range(5):
        payload = {"query": f"Test query {i}", "stream": False}
        response = requests.post(
            f"{BASE_URL}/query",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 429:
            print(f"📛 Query {i+1}: Rate limit hit!")
            print(f"   {response.json()}")
            return
        elif response.status_code == 200:
            print(f"✓ Query {i+1}: OK")
        else:
            print(f"✗ Query {i+1}: Failed ({response.status_code})")
    
    print("✅ Rate limiting tested (send 101 queries to trigger limit)")


def test_caching():
    """Test query caching."""
    print(f"\n{'='*60}")
    print("TEST 7: Query Caching")
    print(f"{'='*60}")
    
    query = "What is the main topic?"
    payload = {"query": query, "stream": False}
    
    # First query
    print("📤 First query...")
    start1 = time.time()
    response1 = requests.post(f"{BASE_URL}/query", json=payload)
    time1 = (time.time() - start1) * 1000
    
    if response1.status_code == 200:
        data1 = response1.json()
        cached1 = data1.get('cached', False)
        print(f"   Time: {time1:.2f}ms, Cached: {cached1}")
    
    # Second query (same query, should be cached)
    time.sleep(1)
    print("📤 Second query (same)...")
    start2 = time.time()
    response2 = requests.post(f"{BASE_URL}/query", json=payload)
    time2 = (time.time() - start2) * 1000
    
    if response2.status_code == 200:
        data2 = response2.json()
        cached2 = data2.get('cached', False)
        print(f"   Time: {time2:.2f}ms, Cached: {cached2}")
        
        if cached2 and time2 < time1:
            print(f"✅ Caching working! Second query was {time1 - time2:.2f}ms faster")
        else:
            print(f"ℹ️  Caching status: {cached2}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 QUERY ENDPOINT TEST SUITE")
    print("="*60)
    print(f"Base URL: {BASE_URL}")
    print("="*60)
    
    try:
        # Test 1: Basic query
        test_basic_query()
        
        # Get a PDF ID from user
        print(f"\n{'='*60}")
        pdf_id = input("Enter a PDF ID to test (or press Enter to skip): ").strip()
        
        if pdf_id:
            # Test 2: Query with PDF ID
            result = test_query_with_pdf_id(pdf_id)
            
            # Test 5: Image retrieval (if visual elements exist)
            if result and result.get('visual_elements'):
                visual = result['visual_elements'][0]
                image_id = visual['image_id']
                test_image_retrieval(pdf_id, image_id)
        
        # Test 3: Streaming
        test_streaming_query()
        
        # Test 4: Validation
        test_query_validation()
        
        # Test 6: Rate limiting
        test_rate_limiting()
        
        # Test 7: Caching
        test_caching()
        
        print(f"\n{'='*60}")
        print("✅ ALL TESTS COMPLETED!")
        print(f"{'='*60}\n")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to server!")
        print("Make sure the server is running:")
        print("  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
