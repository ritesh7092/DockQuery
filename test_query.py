"""Test query to verify the fixes"""
import requests
import json

API_URL = "http://localhost:8000/api/v1"

# Query without PDF ID filter (search all documents)
payload = {
    "query": "What are the main topics covered?",
    "top_k": 5,
    "include_images": True,
    "stream": False
}

print("=" * 70)
print("TESTING QUERY WITH FIXED PROMPT BUILDING")
print("=" * 70)
print(f"\nQuery: {payload['query']}")
print(f"Parameters: top_k={payload['top_k']}, include_images={payload['include_images']}")

try:
    response = requests.post(f"{API_URL}/query", json=payload, timeout=60)
    
    if response.status_code == 200:
        result = response.json()
        
        print(f"\n✅ Query successful!")
        print(f"\nProcessing time: {result.get('processing_time_ms', 0):.0f}ms")
        print(f"Cached: {result.get('cached', False)}")
        print(f"\n{'='*70}")
        print("SUMMARY:")
        print(f"{'='*70}")
        print(result.get('summary', 'No summary'))
        
        print(f"\n{'='*70}")
        print(f"SOURCES ({len(result.get('sources', []))}):")
        print(f"{'='*70}")
        for i, source in enumerate(result.get('sources', [])[:5], 1):
            print(f"\n{i}. Page {source['page']} - Type: {source['type']}")
            print(f"   Preview: {source['content_preview'][:150]}...")
        
        visual_elements = result.get('visual_elements', [])
        if visual_elements:
            print(f"\n{'='*70}")
            print(f"VISUAL ELEMENTS ({len(visual_elements)}):")
            print(f"{'='*70}")
            for i, visual in enumerate(visual_elements[:3], 1):
                print(f"{i}. {visual['type'].upper()} on page {visual['page']}")
                print(f"   URL: {visual.get('url', 'N/A')}")
        
        # Check if we still get the "No text context available" error
        summary = result.get('summary', '')
        if "No text context available" in summary:
            print(f"\n❌ ISSUE FOUND: Still getting 'No text context available' error!")
            print(f"   This might mean retrieval is failing.")
        elif "I apologize" in summary or "cannot provide" in summary:
            print(f"\n⚠️  WARNING: LLM is still apologizing, might indicate an issue")
        else:
            print(f"\n✅ SUCCESS: Got a proper summary without 'No text context' error!")
            
    else:
        print(f"\n❌ Query failed with status code: {response.status_code}")
        print(f"Response: {response.text}")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print(f"\n{'='*70}")
