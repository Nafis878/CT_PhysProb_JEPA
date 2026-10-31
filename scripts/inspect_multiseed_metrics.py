import urllib.request
import re

fid = "1R8vY55qUHZbr9b7iv8sQSVVSJMb9K8q1"
url = f"https://drive.google.com/drive/folders/{fid}?usp=sharing"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
html = urllib.request.urlopen(req).read().decode("utf-8", errors="ignore")

print("Metrics folder fetched. Size:", len(html))
labels = re.findall(r'aria-label="([^"]+)"', html)
for l in sorted(set(labels)):
    if any(k in l.lower() for k in ["csv", "raw", "metric", "wilcoxon", "agg"]):
        print("  *", l.encode("ascii", errors="replace").decode("ascii"))

# Find all CSV files and their IDs
for name in ["Q1_MultiSeed_Raw_Logs.csv", "Q1_MultiSeed_Aggregated_Metrics.csv", "Q1_Wilcoxon_Significance_Table.csv"]:
    idx = html.find(name)
    if idx != -1:
        snippet = html[max(0, idx-300):min(len(html), idx+300)]
        ids = re.findall(r'1[a-zA-Z0-9_-]{32}', snippet)
        print(f"File {name}: IDs={ids}")
