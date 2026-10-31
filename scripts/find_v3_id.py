import urllib.request
import re

url = "https://drive.google.com/drive/folders/1BsP7l4V9EHuevOK0Xp-LuAv48DMqpspI?usp=sharing"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
html = urllib.request.urlopen(req).read().decode("utf-8", errors="ignore")

# Find ID of CT_PhysProb_JEPA_Results_v3
target = "CT_PhysProb_JEPA_Results_v3"
idx = 0
while True:
    pos = html.find(target, idx)
    if pos == -1:
        break
    snippet = html[max(0, pos-200):min(len(html), pos+200)]
    print(f"Match at {pos}:")
    print(snippet.replace("\n", " "))
    # Find all 28-35 char IDs in snippet
    ids = re.findall(r'[a-zA-Z0-9_-]{28,35}', snippet)
    print("Found IDs:", set(ids))
    idx = pos + len(target)
