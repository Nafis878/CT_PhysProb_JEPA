import urllib.request
import re

fid = "1ROBW3D9kL9hYzuinOUIii1HM33-0Zmmp"
url = f"https://drive.google.com/drive/folders/{fid}?usp=sharing"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
html = urllib.request.urlopen(req).read().decode("utf-8", errors="ignore")

for folder in ["Metrics", "Figures", "Predictions", "Checkpoints"]:
    idx = 0
    while True:
        pos = html.find(f'aria-label="{folder}', idx)
        if pos == -1:
            break
        snippet = html[pos:pos+300]
        # Look for ssk or data-id or 28-35 char ID
        ids = re.findall(r'1[a-zA-Z0-9_-]{32}', snippet)
        print(f"Folder {folder}: found potential IDs: {ids}")
        idx = pos + len(folder) + 1
