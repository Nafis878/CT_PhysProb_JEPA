import urllib.request
import re

fid = "1gVREQ2n1Gh8JSmtgp3nNF2NRaJvenVLd"
url = f"https://drive.google.com/drive/folders/{fid}?usp=sharing"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
html = urllib.request.urlopen(req).read().decode("utf-8", errors="ignore")

print("MultiSeed folder size:", len(html))

# Look for subfolders
labels = re.findall(r'aria-label="([^"]+)"', html)
for l in sorted(set(labels)):
    print("Label:", l.encode("ascii", errors="replace").decode("ascii"))

for name in ["Metrics", "Figures", "Predictions", "Checkpoints"]:
    idx = html.find(name)
    if idx != -1:
        snippet = html[idx:idx+300]
        ids = re.findall(r'1[a-zA-Z0-9_-]{32}', snippet)
        print(f"Subfolder {name}: IDs={ids}")
