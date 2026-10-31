import urllib.request
import tarfile
import io
import os

# Create tar.bz2
tar_buf = io.BytesIO()
with tarfile.open(fileobj=tar_buf, mode="w:bz2") as tar:
    # Add main.tex, references.bib, and figures
    tar.add("manuscript/main.tex", arcname="main.tex")
    tar.add("manuscript/references.bib", arcname="references.bib")
    for fig in os.listdir("manuscript/figures"):
        if fig.endswith(".pdf"):
            tar.add(os.path.join("manuscript/figures", fig), arcname=os.path.join("figures", fig))

tar_bytes = tar_buf.getvalue()
print(f"Created bzip2 tarball: {len(tar_bytes)} bytes")

boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
body = (
    f"--{boundary}\r\n"
    f'Content-Disposition: form-data; name="file"; filename="latexTarball.tar.bz2"\r\n'
    f"Content-Type: application/x-bzip2\r\n\r\n"
).encode("latin1") + tar_bytes + f"\r\n--{boundary}--\r\n".encode("latin1")

hosts = ["https://texlive2020.latexonline.cc", "https://latexonline.cc"]

for host in hosts:
    url = f"{host}/data?target=main.tex&command=pdflatex"
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "Mozilla/5.0"
        }
    )

    print(f"\nTrying host: {host} ...")
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            content = resp.read()
            print(f"Response status: {resp.status}, length: {len(content)}")
            if content.startswith(b"%PDF"):
                pdf_path = "C:/Users/UseR/Downloads/CT_PhysProb_JEPA_Manuscript.pdf"
                with open(pdf_path, "wb") as f:
                    f.write(content)
                with open("manuscript/CT_PhysProb_JEPA_Manuscript.pdf", "wb") as f:
                    f.write(content)
                print(f"[SUCCESS] Compiled PDF saved to: {pdf_path} ({len(content)} bytes)!")
                break
            else:
                print("Response preview:")
                print(content[:1000].decode("utf-8", errors="ignore"))
    except urllib.error.HTTPError as he:
        print(f"HTTP Error {he.code}: {he.reason}")
        print(he.read().decode("utf-8", errors="ignore")[:1000])
    except Exception as e:
        print("Error:", e)
