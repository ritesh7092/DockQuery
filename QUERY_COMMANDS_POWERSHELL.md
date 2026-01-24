# PowerShell Query Command Examples

## Issue with curl in PowerShell
In PowerShell, `curl` is an alias for `Invoke-WebRequest` which has different syntax.

## ✅ Working PowerShell Commands

### Option 1: Use curl.exe (Simplest)

**Basic Query:**
```powershell
curl.exe -X POST "http://localhost:8000/api/v1/query" `
  -H "Content-Type: application/json" `
  -d '{\"query\":\"Summarize the document\",\"top_k\":5}'
```

**Query with PDF ID:**
```powershell
$pdfId = "YOUR_PDF_ID_HERE"
curl.exe -X POST "http://localhost:8000/api/v1/query" `
  -H "Content-Type: application/json" `
  -d "{`\"query`\":`\"What is on page 1?`\",`\"pdf_id`\":`\"$pdfId`\"}"
```

### Option 2: Use PowerShell Native

```powershell
$body = @{
    query = "Summarize the document"
    top_k = 5
    include_images = $true
    stream = $false
} | ConvertTo-Json

$response = Invoke-WebRequest `
    -Uri "http://localhost:8000/api/v1/query" `
    -Method POST `
    -ContentType "application/json" `
    -Body $body

$result = $response.Content | ConvertFrom-Json
$result | ConvertTo-Json
```

### Option 3: Use Python Test Script (RECOMMENDED)

```powershell
python tests/test_query_endpoints.py
```

## 🖼️ Get Image

```powershell
curl.exe http://localhost:8000/api/v1/images/pdf_id/image_id -o output.png
```

## 📊 Use Swagger UI (EASIEST)

Open your browser: **http://localhost:8000/docs**

1. Click on `POST /api/v1/query`
2. Click "Try it out"
3. Enter your query JSON
4. Click "Execute"
