import urllib.request
import re

fid = "12ii13bsBe0diIlFfyRK4wectmvsQFRJA"
url = f"https://drive.google.com/drive/folders/{fid}?usp=sharing"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
html = urllib.request.urlopen(req).read().decode("utf-8", errors="ignore")

labels = re.findall(r'aria-label="([^"]+)"', html)
print("Checkpoints files in Google Drive:")
for l in sorted(set(labels)):
    if ".pt" in l:
        print("  *", l)
