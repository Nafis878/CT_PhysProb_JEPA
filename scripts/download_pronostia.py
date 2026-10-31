"""Download PRONOSTIA (FEMTO-ST / IEEE PHM 2012 Bearing Dataset).
Usage: python scripts/download_pronostia.py
"""
import io
import os
import sys
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "PRONOSTIA"
URL = "https://github.com/wkzs111/phm-ieee-2012-data-challenge-dataset/archive/refs/heads/master.zip"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    zip_target = OUT / "pronostia_raw.zip"
    
    if not zip_target.exists() or zip_target.stat().st_size < 1000000:
        print(f"Downloading PRONOSTIA dataset from {URL}...", flush=True)
        with requests.get(URL, stream=True, timeout=(15, 60), headers={"User-Agent": "Mozilla/5.0"}) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            downloaded = 0
            with open(zip_target, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            print(f"  {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB", flush=True)
                        else:
                            print(f"  {downloaded / 1e6:.1f} MB", flush=True)

    print("Extracting PRONOSTIA dataset...", flush=True)
    with zipfile.ZipFile(zip_target, "r") as z:
        # Strip the top-level repo folder when extracting
        for member in z.infolist():
            parts = Path(member.filename).parts
            if len(parts) > 1:
                rel_path = Path(*parts[1:])
                target_path = OUT / rel_path
                if member.is_dir():
                    target_path.mkdir(parents=True, exist_ok=True)
                else:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(member) as src, open(target_path, "wb") as dst:
                        dst.write(src.read())

    print("PRONOSTIA extraction complete! Summary:", flush=True)
    dirs = [d.name for d in OUT.iterdir() if d.is_dir()]
    print("  Subdirectories:", dirs, flush=True)


if __name__ == "__main__":
    main()
