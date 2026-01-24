# PowerShell Commands for Testing Upload Endpoint

## The Issue
In PowerShell, `curl` is an alias for `Invoke-WebRequest` which has different syntax.

## Solutions

### Option 1: Use PowerShell's Invoke-WebRequest
```powershell
$response = Invoke-WebRequest -Uri "http://localhost:8000/api/v1/upload" `
    -Method POST `
    -Form @{file = Get-Item -Path "test.pdf"}
$response.Content | ConvertFrom-Json | ConvertTo-Json
```

### Option 2: Use curl.exe Explicitly
```powershell
curl.exe -X POST http://localhost:8000/api/v1/upload -F "file=@test.pdf"
```

### Option 3: Use the Python Test Script (RECOMMENDED)
```powershell
python tests/test_upload_endpoints.py
```

## Quick Upload Test (PowerShell)

```powershell
# Upload a PDF
$file = Get-Item "test.pdf"
$uri = "http://localhost:8000/api/v1/upload"

$response = Invoke-WebRequest -Uri $uri -Method POST -Form @{file = $file}
$data = $response.Content | ConvertFrom-Json

Write-Host "✅ Upload successful!"
Write-Host "PDF ID: $($data.pdf_id)"
Write-Host "Status: $($data.status)"

# Check status
$statusUri = "http://localhost:8000/api/v1/upload/$($data.pdf_id)/status"
$statusResponse = Invoke-WebRequest -Uri $statusUri
$statusData = $statusResponse.Content | ConvertFrom-Json

Write-Host "`n📊 Processing Status:"
Write-Host "Status: $($statusData.status)"
Write-Host "Progress: $([math]::Round($statusData.progress * 100, 1))%"
```

## Alternative: Use Postman, Insomnia, or Browser

You can also use API testing tools or the auto-generated Swagger UI at:
- http://localhost:8000/docs
