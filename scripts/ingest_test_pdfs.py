from pathlib import Path
import requests

# Configuration
API_URL = "http://localhost:8000"
TENANT_ID = "8fa00d66-ad8d-44cc-baf6-a71c7540dc84"  # Replace with tenant UUID
PDF_DIR = Path("/mnt/c/Users/khiza/Documents/test-data")

print(f"Uploading PDFs from {PDF_DIR} to tenant {TENANT_ID}...\n")

for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
    print(f"Uploading: {pdf_path.name}...")
    with open(pdf_path, "rb") as f:
        response = requests.post(
            f"{API_URL}/api/v1/documents/upload?tenant_id={TENANT_ID}",
            files={"file": (pdf_path.name, f, "application/pdf")},
            timeout=120.0,
        )
    print(f"Status: {response.status_code} -> {response.json()}\n")