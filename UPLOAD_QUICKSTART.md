# FastAPI Upload Endpoint - Quick Start Guide

## 🚀 Starting the Server

```bash
# 1. Activate virtual environment
.\venv\Scripts\activate

# 2. Start the FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Server will be available at: `http://localhost:8000`

---

## 📤 Upload a PDF

### Using cURL

```bash
curl -X POST http://localhost:8000/api/v1/upload \
  -F "file=@path/to/your/document.pdf" \
  -H "Content-Type: multipart/form-data"
```

### Using Python

```python
import requests

# Upload file
with open("document.pdf", "rb") as f:
    files = {"file": ("document.pdf", f, "application/pdf")}
    response = requests.post("http://localhost:8000/api/v1/upload", files=files)
    
    if response.status_code == 200:
        data = response.json()
        pdf_id = data["pdf_id"]
        print(f"✅ Upload successful! PDF ID: {pdf_id}")
    else:
        print(f"❌ Upload failed: {response.json()}")
```

### Expected Response

```json
{
  "pdf_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "filename": "document.pdf",
  "status": "processing",
  "message": "File uploaded successfully and processing has started"
}
```

---

## 📊 Check Processing Status

### Using cURL

```bash
curl http://localhost:8000/api/v1/upload/a1b2c3d4-e5f6-7890-abcd-ef1234567890/status
```

### Using Python

```python
import requests
import time

pdf_id = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

# Poll until complete
while True:
    response = requests.get(f"http://localhost:8000/api/v1/upload/{pdf_id}/status")
    data = response.json()
    
    status = data["status"]
    progress = data["progress"]
    
    print(f"Status: {status} - Progress: {progress*100:.1f}%")
    
    if status in ["completed", "failed"]:
        print(f"Final result: {data}")
        break
    
    time.sleep(2)
```

---

## 🧪 Run Test Suite

```bash
# Install requests if needed
pip install requests

# Run test script
python tests/test_upload_endpoints.py
```

The test script will guide you through testing all features including:
- Valid PDF upload
- Invalid file rejection
- File size validation
- Rate limiting
- Status tracking

---

## 🔒 Security Features

- ✅ **PDF Magic Bytes**: Validates real PDFs (blocks fake/renamed files)
- ✅ **File Size Limit**: Maximum 50MB
- ✅ **Rate Limiting**: 10 uploads per hour per IP
- ✅ **Filename Sanitization**: Prevents path traversal attacks
- ✅ **MIME Type Validation**: Only accepts `application/pdf`

---

## 📖 API Documentation

Interactive API docs available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## ⚠️ Common Errors

| Error Code | Cause | Solution |
|------------|-------|----------|
| 400 | Invalid file type | Use a real PDF file |
| 400 | Invalid PDF magic bytes | File is not a valid PDF |
| 413 | File too large | File must be ≤ 50MB |
| 429 | Rate limit exceeded | Wait 1 hour or use different IP |
| 404 | PDF ID not found | Check the pdf_id is correct |

---

## 🎯 Example: Complete Workflow

```python
import requests
import time

# 1. Upload PDF
with open("report.pdf", "rb") as f:
    files = {"file": f}
    resp = requests.post("http://localhost:8000/api/v1/upload", files=files)
    pdf_id = resp.json()["pdf_id"]
    print(f"Uploaded: {pdf_id}")

# 2. Wait for processing
while True:
    resp = requests.get(f"http://localhost:8000/api/v1/upload/{pdf_id}/status")
    data = resp.json()
    
    if data["status"] == "completed":
        print(f"✅ Processing complete!")
        print(f"   Pages: {data['result']['total_pages']}")
        print(f"   Images: {data['result']['images_extracted']}")
        break
    elif data["status"] == "failed":
        print(f"❌ Failed: {data['error']}")
        break
    
    time.sleep(2)

# 3. Query the document (using query endpoint)
query_resp = requests.post(
    "http://localhost:8000/api/v1/query",
    json={"query": "What is this document about?"}
)
print(f"Answer: {query_resp.json()['answer']}")
```

---

For full documentation, see [walkthrough.md](file:///C:/Users/rites/.gemini/antigravity/brain/92472883-b42e-4896-9072-2afe32dee3dd/walkthrough.md)
