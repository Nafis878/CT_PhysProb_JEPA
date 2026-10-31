import urllib.request
import re

url = "https://drive.google.com/drive/folders/1BsP7l4V9EHuevOK0Xp-LuAv48DMqpspI?usp=sharing"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
try:
    html = urllib.request.urlopen(req).read().decode("utf-8", errors="ignore")
    print("Page fetched. Size:", len(html))

    # Look for results folders or recently modified items
    keywords = ["Results", "Metrics", "Figure", "v3", "v4", "Q1", "TriBenchmark"]
    for kw in keywords:
        matches = [m.start() for m in re.finditer(kw, html, re.IGNORECASE)]
        print(f"Keyword '{kw}': {len(matches)} occurrences")
        for idx in matches[:3]:
            snippet = html[max(0, idx - 40):min(len(html), idx + 100)].replace("\n", " ")
            print(f"    snippet: ...{snippet}...")

    # Find all items in the folder
    pattern = r'\["([a-zA-Z0-9_-]{28,35})",\["1BsP7l4V9EHuevOK0Xp-LuAv48DMqpspI"\],"([^"]+)"'
    items = re.findall(pattern, html)
    print("\nItems in root folder:")
    for fid, name in items:
        print(f"  {name} -> {fid}")
except Exception as e:
    print("Error:", e)
