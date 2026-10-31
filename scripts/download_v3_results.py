import urllib.request
import os

files_to_download = [
    ("Fig1_TriBenchmark_StressTest_Curves.png", "1BYnXVEXAXVeMLsgTmWF5OwDKuKYzQN_z"),
    ("Fig2_Calibrated_Uncertainty_Trajectory.png", "1iqYux7U6sgiuft00YLue4V3Qf53ZA1XI"),
    ("Q1_TriBenchmark_Master_Metrics.csv", "1HbBrdj4rY2rTtHtlva255eh7Ib4ye_5j")
]

os.makedirs("results", exist_ok=True)

for name, fid in files_to_download:
    url = f"https://drive.usercontent.google.com/download?id={fid}&export=download"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    out_path = os.path.join("results", name)
    try:
        data = urllib.request.urlopen(req).read()
        with open(out_path, "wb") as f:
            f.write(data)
        print(f"Downloaded {name}: {len(data)} bytes")
    except Exception as e:
        print(f"Error downloading {name}: {e}")
