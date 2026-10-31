import urllib.request
import re

fid = "1X1tLdnUzF3S5f68DA5FM93AoqZn-FIv6"
url = f"https://drive.google.com/drive/folders/{fid}?usp=sharing"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
html = urllib.request.urlopen(req).read().decode("utf-8", errors="ignore")

for f in ["Fig1_TriBenchmark_StressTest_Curves.png", "Fig1_TriBenchmark_StressTest_Curves.pdf",
          "Fig2_Calibrated_Uncertainty_Trajectory.png", "Fig2_Calibrated_Uncertainty_Trajectory.pdf"]:
    idx = html.find(f)
    if idx != -1:
        snippet = html[max(0, idx-300):min(len(html), idx+300)]
        ids = re.findall(r'1[a-zA-Z0-9_-]{32}', snippet)
        print(f"File {f}: IDs = {ids}")
