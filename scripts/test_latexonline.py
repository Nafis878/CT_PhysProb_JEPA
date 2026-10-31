import urllib.request
import tarfile
import io
import os

# Create tar.gz of manuscript
tar_buf = io.BytesIO()
with tarfile.open(fileobj=tar_buf, mode="w:gz") as tar:
    for root, dirs, files in os.walk("manuscript"):
        for file in files:
            if file.endswith(".zip") or file.endswith(".tar.gz"):
                continue
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, "manuscript")
            tar.add(full_path, arcname=rel_path)

tar_bytes = tar_buf.getvalue()
print(f"Created tar.gz of manuscript: {len(tar_bytes)} bytes")

# Check latexonline.cc
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
    print("Sending compilation request to latexonline.cc...")
    with urllib.request.urlopen(req, timeout=60) as resp:
        content = resp.read()
        print(f"Response status: {resp.status}, content length: {len(content)}")
        if content.startswith(b"%PDF"):
            with open("manuscript/CT_PhysProb_JEPA_Manuscript.pdf", "wb") as f:
                f.write(content)
            with open("C:/Users/UseR/Downloads/CT_PhysProb_JEPA_Manuscript.pdf", "wb") as f:
                f.write(content)
            print("SUCCESS! Saved PDF to C:/Users/UseR/Downloads/CT_PhysProb_JEPA_Manuscript.pdf!")
        else:
            print("Response was not PDF:")
            print(content[:500].decode("utf-8", errors="ignore"))
except Exception as e:
    print("Error:", e)
