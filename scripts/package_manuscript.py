import zipfile
import os
from pathlib import Path

manuscript_dir = Path("c:/Users/UseR/Downloads/Continous/manuscript")
zip_out1 = Path("c:/Users/UseR/Downloads/Continous/manuscript/CT_PhysProb_JEPA_Manuscript_Overleaf.zip")
zip_out2 = Path("c:/Users/UseR/Downloads/CT_PhysProb_JEPA_Manuscript_Overleaf.zip")

for zip_out in [zip_out1, zip_out2]:
    with zipfile.ZipFile(zip_out, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(manuscript_dir):
            for file in files:
                if file.endswith(".zip"):
                    continue
                full_path = Path(root) / file
                rel_path = full_path.relative_to(manuscript_dir)
                zf.write(full_path, rel_path)
    print(f"Created Overleaf ZIP: {zip_out} ({os.path.getsize(zip_out)} bytes)")
