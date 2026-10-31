import urllib.request
import tarfile
import io
import os

tar_buf = io.BytesIO()
with tarfile.open(fileobj=tar_buf, mode="w") as tar:
    for root, dirs, files in os.walk("manuscript"):
        for file in files:
            if file.endswith(".png") or file.endswith(".zip") or file.endswith(".tar") or file.endswith(".tar.gz"):
                continue
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, "manuscript")
            tar.add(full_path, arcname=rel_path)

tar_bytes = tar_buf.getvalue()
print(f"Created uncompressed tar: {len(tar_bytes)} bytes")

url = "https://latexonline.cc/data?target=main.tex&command=pdflatex"
req = urllib.request.Request(
    url,
    data=tar_bytes,
    headers={
        "Content-Type": "application/x-tar",
        "User-Agent": "Mozilla/5.0"
    }
)

try:
    print("Uploading tar to https://latexonline.cc/data...")
    with urllib.request.urlopen(req, timeout=90) as resp:
        content = resp.read()
        print(f"Response status: {resp.status}, length: {len(content)}")
        if content.startswith(b"%PDF"):
            pdf_path = "C:/Users/UseR/Downloads/CT_PhysProb_JEPA_Manuscript.pdf"
            with open(pdf_path, "wb") as f:
                f.write(content)
            with open("manuscript/CT_PhysProb_JEPA_Manuscript.pdf", "wb") as f:
                f.write(content)
            print(f"[SUCCESS] Compiled PDF saved to {pdf_path} ({len(content)} bytes)!")
        else:
            print("Response text preview:")
            print(content[:1500].decode("utf-8", errors="ignore"))
except urllib.error.HTTPError as he:
    print(f"HTTP Error {he.code}: {he.reason}")
    print(he.read().decode("utf-8", errors="ignore")[:3000])
except Exception as e:
    print("Error:", e)
