#!/usr/bin/env python3
"""
Packages the complete source code, audit documents, experimental scripts,
and unsimulated 30-seed evaluation results into a single comprehensive ZIP archive.
"""

import os
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = Path(r"C:\Users\UseR\.gemini\antigravity\brain\91deb9fc-5063-4abb-b6eb-057d869da051")

ZIP_NAME = "CT_PhysProb_JEPA_Full_Source_and_Audit_v12.zip"
LOCAL_ZIP_PATH = ROOT / ZIP_NAME
ARTIFACT_ZIP_PATH = ARTIFACT_DIR / ZIP_NAME

EXCLUDE_DIRS = {
    '__pycache__', '.pytest_cache', '.git', '.idea', '.vscode'
}

EXCLUDE_EXTS = {
    '.zip', '.tar', '.gz'
}

def create_archive():
    print(f"Packaging codebase from {ROOT}...")
    
    with zipfile.ZipFile(LOCAL_ZIP_PATH, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(ROOT):
            # Prune excluded directories
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            
            for file in files:
                file_p = Path(root) / file
                
                # Exclude zip archives and temporary files
                if file_p.suffix.lower() in EXCLUDE_EXTS:
                    continue
                if file.startswith('.~') or file.endswith('.tmp'):
                    continue
                    
                rel_p = file_p.relative_to(ROOT)
                zipf.write(file_p, arcname=str(rel_p))
                
    zip_size_mb = LOCAL_ZIP_PATH.stat().st_size / (1024 * 1024)
    print(f"Created local ZIP: {LOCAL_ZIP_PATH} ({zip_size_mb:.2f} MB)")
    
    # Copy to artifacts directory
    if ARTIFACT_DIR.exists():
        shutil.copy2(LOCAL_ZIP_PATH, ARTIFACT_ZIP_PATH)
        print(f"Copied to Artifact Directory: {ARTIFACT_ZIP_PATH}")

if __name__ == '__main__':
    create_archive()
