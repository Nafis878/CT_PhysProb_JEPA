import urllib.request
import os

multiseed_files = [
    ("Q1_MultiSeed_Raw_Logs.csv", "1ne2MA-dcjMtJF8wllMXRA1JSB2AY7ruz"),
    ("Q1_MultiSeed_Aggregated_Metrics.csv", "1KIYGmTCjIT3c2Fg2P6B20Ymzze5RH1J0"),
    ("Q1_Wilcoxon_Significance_Table.csv", "1H4HxgXGhe0c-S6avYLz_rkYsERB_3H72"),
    ("Fig1_MultiSeed_TriBenchmark_ConfidenceCurves.png", "1yOERuzUSPuS961ezmwqC2D-Ruy4VDMWe"),
    ("Fig2_MultiSeed_BoxPlot_70pct_Dropout.png", "1VDkZ5RKXJdLOSK3PmRsHtmsyNeGbMUr_")
]

os.makedirs("results/multiseed", exist_ok=True)

for name, fid in multiseed_files:
    url = f"https://drive.usercontent.google.com/download?id={fid}&export=download"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    out_path = os.path.join("results/multiseed", name)
    try:
        data = urllib.request.urlopen(req).read()
        with open(out_path, "wb") as f:
            f.write(data)
        print(f"Downloaded {name}: {len(data)} bytes")
    except Exception as e:
        print(f"Error downloading {name}: {e}")
        # fallback to uc endpoint
        url_uc = f"https://drive.google.com/uc?id={fid}&export=download"
        try:
            req_uc = urllib.request.Request(url_uc, headers={"User-Agent": "Mozilla/5.0"})
            data = urllib.request.urlopen(req_uc).read()
            with open(out_path, "wb") as f:
                f.write(data)
            print(f"Downloaded via uc {name}: {len(data)} bytes")
        except Exception as e2:
            print(f"UC error {name}: {e2}")
