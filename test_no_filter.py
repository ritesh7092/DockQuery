"""Test query without PDF ID filter"""
import requests
import json

API_URL = "http://localhost:8000/api/v1"

# Query WITHOUT PDF ID filter - search all documents
payload = {
    "query": "What are the main topics covered?",
    "top_k": 5,
    "include_images": True,
    "stream": False
    # No pdf_id - this is the key change!
}

print("=" * 70)
print("TESTING QUERY WITHOUT PDF ID FILTER")
print("=" * 70)
print(f"\nThis should retrieve from ANY of the 115 documents in the database")
print(f"\nQuery: {payload['query']}")

try:
    response = requests.post(f"{API_URL}/query", json=payload, timeout=60)
    
    if response.status_code == 200:
        result = response.json()
        
        print(f"\n{'='*70}")
        print("RESULT:")
        print(f"{'='*70}")
        print(f"\nProcessing time: {result.get('processing_time_ms', 0):.0f}ms")
        print(f"Number of sources: {len(result.get('sources', []))}")
        
        print(f"\n{'='*70}")
        print("SUMMARY:")
        print(f"{'='*70}")
        summary = result.get('summary', 'No summary')
        print(summary)
        
        # Check for the error
        if "No text context available" in summary:
            print(f"\nSTILL HAS ERROR - This means embedding/retrieval is failing")
        else:
            print(f"\nSUCCESS! Query worked without PDF ID filter")
        
        print(f"\n{'='*70}")
        print("TOP SOURCES:")
        print(f"{'='*70}")
        for i, source in enumerate(result.get('sources', [])[:3], 1):
            print(f"\n{i}. Page {source['page']} - {source['type']}")
            print(f"   {source['content_preview'][:100]}...")
            
    else:
        print(f"\nERROR: {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"\nERROR: {e}")

print(f"\n{'='*70}")
