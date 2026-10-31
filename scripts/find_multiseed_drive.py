import urllib.request
import re

url = "https://drive.google.com/drive/folders/1BsP7l4V9EHuevOK0Xp-LuAv48DMqpspI?usp=sharing"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
html = urllib.request.urlopen(req).read().decode("utf-8", errors="ignore")

print("Drive root fetched. Size:", len(html))

# Look for MultiSeed in html
target = "MultiSeed"
idx = 0
found = []
while True:
    pos = html.find(target, idx)
    if pos == -1:
        break
    snippet = html[max(0, pos-200):min(len(html), pos+200)].replace("\n", " ")
    print(f"Match at {pos}:")
    print(snippet)
    ids = re.findall(r'1[a-zA-Z0-9_-]{32}', snippet)
    print("Found IDs:", set(ids))
    found.extend(ids)
    idx = pos + len(target)

# Check all labels
labels = re.findall(r'aria-label="([^"]+)"', html)
for l in sorted(set(labels)):
    if any(k in l.lower() for k in ["multi", "seed", "wilcoxon", "results", "metrics"]):
        print("Label match:", l)
