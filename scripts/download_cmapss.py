"""Download official NASA C-MAPSS (FD001-FD004) with streaming, progress and SHA-256 record.
Usage: python scripts/download_cmapss.py
"""
import hashlib
import io
import sys
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "CMAPSS"
URLS = ["https://data.nasa.gov/docs/legacy/CMAPSSData.zip",
        "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"]


def fetch(url):
    buf = io.BytesIO()
    with requests.get(url, stream=True, timeout=(15, 30), headers={"User-Agent": "Mozilla/5.0"}) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        for i, chunk in enumerate(r.iter_content(1 << 16)):
            buf.write(chunk)
            if i % 40 == 0:
                print(f"  {buf.tell()/1e6:.1f}/{total/1e6:.1f} MB", flush=True)
    return buf.getvalue()


def extract(b, dest):
    z = zipfile.ZipFile(io.BytesIO(b))
    for n in z.namelist():
        if n.lower().endswith(".zip"):  # S3 mirror nests the zip
            extract(z.read(n), dest)
    z.extractall(dest)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for u in URLS:
        try:
            print("Downloading", u, flush=True)
            b = fetch(u)
            break
        except Exception as e:  # try next mirror
            print("  failed:", e, flush=True)
    else:
        sys.exit("All mirrors failed")
    sha = hashlib.sha256(b).hexdigest()
    (OUT / "source.zip").write_bytes(b)
    extract(b, OUT)
    (ROOT / "audit").mkdir(exist_ok=True)
    (ROOT / "audit" / "cmapss_sha256.txt").write_text(f"{u}\n{sha}  {len(b)} bytes\n")
    print("SHA256", sha, "| txt files:", sorted(p.name for p in OUT.rglob("*_FD00*.txt")))


if __name__ == "__main__":
    main()
