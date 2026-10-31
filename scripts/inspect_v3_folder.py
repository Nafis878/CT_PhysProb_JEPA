import urllib.request
import re

fid = "1ROBW3D9kL9hYzuinOUIii1HM33-0Zmmp"
url = f"https://drive.google.com/drive/folders/{fid}?usp=sharing"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
try:
    html = urllib.request.urlopen(req).read().decode("utf-8", errors="ignore")
    print("Page fetched. Size:", len(html))

    # Look for items
    labels = re.findall(r'aria-label="([^"]+)"', html)
    print("Aria labels found:")
    for l in set(labels):
        if any(k in l.lower() for k in ["folder", "file", "csv", "png", "pdf", "pt", "metrics", "checkpoint", "figure", "pred"]):
            print("  *", l.encode("ascii", errors="replace").decode("ascii"))

    # Also look for subfolder IDs
    # Checkpoints, Figures, Metrics, Predictions
    pattern = rf'\["([a-zA-Z0-9_-]{{28,35}})",\["{fid}"\],"([^"]+)"'
    items = re.findall(pattern, html)
    print("\nItems in v3 folder:")
    for sub_id, name in items:
        print(f"  {name} -> {sub_id}")
except Exception as e:
    print("Error:", e)
