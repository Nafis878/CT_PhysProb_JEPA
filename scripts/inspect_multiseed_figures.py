import urllib.request
import re

fid = "1L9PFKgY4pI3i1BjuAh5KBeeDr9MOv2tG"
url = f"https://drive.google.com/drive/folders/{fid}?usp=sharing"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
html = urllib.request.urlopen(req).read().decode("utf-8", errors="ignore")

print("Figures folder fetched. Size:", len(html))
labels = re.findall(r'aria-label="([^"]+)"', html)
for l in sorted(set(labels)):
    if any(k in l.lower() for k in ["png", "pdf", "fig"]):
        print("  *", l.encode("ascii", errors="replace").decode("ascii"))

# Find files and their IDs
for name in ["Fig1_MultiSeed_TriBenchmark_ConfidenceCurves.png", "Fig1_MultiSeed_TriBenchmark_ConfidenceCurves.pdf",
              "Fig2_MultiSeed_BoxPlot_70pct_Dropout.png", "Fig2_MultiSeed_BoxPlot_70pct_Dropout.pdf"]:
    idx = html.find(name)
    if idx != -1:
        snippet = html[max(0, idx-300):min(len(html), idx+300)]
        ids = re.findall(r'1[a-zA-Z0-9_-]{32}', snippet)
        print(f"File {name}: IDs={ids}")
