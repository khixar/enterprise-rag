import os
import sys
import json
import time
import random
import urllib.request
import urllib.parse
from pathlib import Path

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

TARGET_DIRS = [
    Path("/mnt/c/Users/khiza/Documents/test-data"),
    Path.home() / "documents" / "test-data",
    Path.home() / "Documents" / "test-data",
]

INDEX_FILE = Path(__file__).parent / "legal_docs_index.json"

def main():
    for d in TARGET_DIRS:
        d.mkdir(parents=True, exist_ok=True)
    
    primary_dir = TARGET_DIRS[0]
    
    # Load index
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        index_data = json.load(f)
    
    print(f"Loaded {len(index_data)} entries from legal_docs_index.json")
    
    existing_pdfs = {p.name for p in primary_dir.glob("*.pdf")}
    print(f"Currently existing PDFs in {primary_dir}: {len(existing_pdfs)}")
    for name in sorted(existing_pdfs):
        print(f"  - {name}")
    
    target_count = 50
    downloaded_count = len(existing_pdfs)
    
    for item in index_data:
        if downloaded_count >= target_count:
            break
            
        pdf_url = item.get("pdf_url", "")
        if not pdf_url:
            continue
            
        filename = os.path.basename(urllib.parse.urlparse(pdf_url).path)
        if not filename.lower().endswith(".pdf"):
            filename = f"{filename}.pdf"
            
        target_path = primary_dir / filename
        if target_path.exists() and target_path.stat().st_size > 1000:
            continue
            
        print(f"[{downloaded_count + 1}/{target_count}] Downloading: {pdf_url} -> {filename}...")
        try:
            req = urllib.request.Request(pdf_url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=25) as resp:
                pdf_bytes = resp.read()
            
            # Check magic bytes for PDF
            if not pdf_bytes.startswith(b'%PDF'):
                print(f"   -> Skipped (not a valid PDF header)")
                continue
                
            if len(pdf_bytes) < 2000:
                print(f"   -> Skipped (file too small: {len(pdf_bytes)} bytes)")
                continue
                
            # Write to primary dir
            target_path.write_bytes(pdf_bytes)
            
            # Sync to all other target dirs
            for other_dir in TARGET_DIRS[1:]:
                (other_dir / filename).write_bytes(pdf_bytes)
                
            downloaded_count += 1
            print(f"   -> Saved {filename} ({len(pdf_bytes):,} bytes)")
            
            time.sleep(random.uniform(0.8, 1.5))
        except Exception as e:
            print(f"   -> Failed {pdf_url}: {e}")
            time.sleep(0.5)
            
    # Sync all PDFs across all target directories
    for p in primary_dir.glob("*.pdf"):
        data = p.read_bytes()
        for other_dir in TARGET_DIRS[1:]:
            dest = other_dir / p.name
            if not dest.exists() or dest.stat().st_size != len(data):
                dest.write_bytes(data)

    print("\n" + "="*60)
    print(f"DOWNLOAD COMPLETE! Total PDFs in {primary_dir}: {len(list(primary_dir.glob('*.pdf')))}")
    for d in TARGET_DIRS:
        print(f" - {d} : {len(list(d.glob('*.pdf')))} PDFs")
    print("="*60)

if __name__ == "__main__":
    main()
