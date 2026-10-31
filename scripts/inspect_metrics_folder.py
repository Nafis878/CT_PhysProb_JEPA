import urllib.request
import re

fid = "1BQoH3jxaRQvvLxAizKOA3xJcLJaFn8hW"
url = f"https://drive.google.com/drive/folders/{fid}?usp=sharing"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
html = urllib.request.urlopen(req).read().decode("utf-8", errors="ignore")

print("Metrics folder fetched. Size:", len(html))
labels = re.findall(r'aria-label="([^"]+)"', html)
for l in set(labels):
    if any(k in l.lower() for k in ["csv", "metric", "q1", "result"]):
        print("  *", l)

# Look for file IDs of CSV files
pattern = r'data-id="([a-zA-Z0-9_-]{28,35})"'
data_ids = re.findall(pattern, html)
print("Data IDs found in Metrics folder:", set(data_ids))
