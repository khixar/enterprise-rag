import argparse
from pathlib import Path

import requests

# Configuration
API_URL = "http://localhost:8000"
TENANT_ID = "8fa00d66-ad8d-44cc-baf6-a71c7540dc84"  # Replace with tenant UUID
PDF_DIR = Path("/mnt/c/Users/khiza/Documents/test-data")

parser = argparse.ArgumentParser(description="Upload every PDF in a folder to a tenant")
parser.add_argument("--tenant-id", default=TENANT_ID)
parser.add_argument("--pdf-dir", type=Path, default=PDF_DIR)
args = parser.parse_args()

print(f"Uploading PDFs from {args.pdf_dir} to tenant {args.tenant_id}...\n")

for pdf_path in sorted(args.pdf_dir.glob("*.pdf")):
    print(f"Uploading: {pdf_path.name}...")
    with open(pdf_path, "rb") as f:
        response = requests.post(
            f"{API_URL}/api/v1/documents/upload?tenant_id={args.tenant_id}",
            files={"file": (pdf_path.name, f, "application/pdf")},
            timeout=600.0,
        )
    print(f"Status: {response.status_code} -> {response.json()}\n")
