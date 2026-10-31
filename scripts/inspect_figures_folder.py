import urllib.request
import re

fid = "1X1tLdnUzF3S5f68DA5FM93AoqZn-FIv6"
url = f"https://drive.google.com/drive/folders/{fid}?usp=sharing"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
html = urllib.request.urlopen(req).read().decode("utf-8", errors="ignore")

print("Figures folder fetched. Size:", len(html))
labels = re.findall(r'aria-label="([^"]+)"', html)
for l in sorted(set(labels)):
    if any(k in l.lower() for k in ["png", "pdf", "fig"]):
        print("  *", l)

# Find files and their IDs
pattern = r'data-id="([a-zA-Z0-9_-]{28,35})"'
data_ids = re.findall(pattern, html)
print("Data IDs found in Figures folder:", set(data_ids))
