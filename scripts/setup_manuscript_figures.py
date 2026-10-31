import urllib.request
import os

pdf_files = [
    ("Fig1_MultiSeed_TriBenchmark_ConfidenceCurves.pdf", "1HKdDCwWBLq_FIMzbotqVu48pnlIzcMok"),
    ("Fig2_MultiSeed_BoxPlot_70pct_Dropout.pdf", "1sgcQIKzAPXduPz7bpyCO0P-jL-kPQm4O"),
    ("Fig2_Calibrated_Uncertainty_Trajectory.pdf", "1Jdri4EbUh0A5g6mL_Khl6slDCCJiw4bz")
]

os.makedirs("manuscript/figures", exist_ok=True)
os.makedirs("results/multiseed", exist_ok=True)

for name, fid in pdf_files:
    url = f"https://drive.usercontent.google.com/download?id={fid}&export=download"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    out_path = os.path.join("manuscript/figures", name)
    try:
        data = urllib.request.urlopen(req).read()
        with open(out_path, "wb") as f:
            f.write(data)
        print(f"Downloaded {name} to manuscript/figures: {len(data)} bytes")
    except Exception as e:
        print(f"Error downloading {name}: {e}")
        # fallback
        try:
            url_uc = f"https://drive.google.com/uc?id={fid}&export=download"
            req_uc = urllib.request.Request(url_uc, headers={"User-Agent": "Mozilla/5.0"})
            data = urllib.request.urlopen(req_uc).read()
            with open(out_path, "wb") as f:
                f.write(data)
            print(f"Downloaded via uc {name}: {len(data)} bytes")
        except Exception as e2:
            print(f"UC error {name}: {e2}")

# Also copy PNGs to manuscript/figures
import shutil
for f in ["Fig1_MultiSeed_TriBenchmark_ConfidenceCurves.png", "Fig2_MultiSeed_BoxPlot_70pct_Dropout.png"]:
    src = os.path.join("results/multiseed", f)
    dst = os.path.join("manuscript/figures", f)
    if os.path.exists(src):
        shutil.copyfile(src, dst)
        print(f"Copied {f} to manuscript/figures")

src2 = "results/Fig2_Calibrated_Uncertainty_Trajectory.png"
dst2 = "manuscript/figures/Fig2_Calibrated_Uncertainty_Trajectory.png"
if os.path.exists(src2):
    shutil.copyfile(src2, dst2)
    print(f"Copied Fig2_Calibrated_Uncertainty_Trajectory.png to manuscript/figures")
