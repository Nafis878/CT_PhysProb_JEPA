import urllib.request
import tarfile
import io
import os

tar_buf = io.BytesIO()
with tarfile.open(fileobj=tar_buf, mode="w:gz") as tar:
    for root, dirs, files in os.walk("manuscript"):
        for file in files:
            # Skip large pngs and archives for online compiler
            if file.endswith(".png") or file.endswith(".zip") or file.endswith(".tar.gz") or file.endswith(".pdf"):
                if not root.endswith("figures"):
                    continue
                # in figures, include only .pdf figures!
                if not file.endswith(".pdf"):
                    continue
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, "manuscript")
            tar.add(full_path, arcname=rel_path)

tar_bytes = tar_buf.getvalue()
print(f"Created slim tar.gz: {len(tar_bytes)} bytes")

boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
body = (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="file"; filename="manuscript.tar.gz"\r\n'
    f"Content-Type: application/gzip\r\n\r\n"
).encode("utf-8") + tar_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

req = urllib.request.Request(
    "https://latexonline.cc/compile?tarball=true&command=pdflatex",
    data=body,
    headers={
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "User-Agent": "Mozilla/5.0"
    }
)

try:
    print("Sending compilation request...")
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
    print(he.read().decode("utf-8", errors="ignore")[:2000])
except Exception as e:
    print("Error:", e)
