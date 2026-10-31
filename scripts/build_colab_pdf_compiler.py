"""Generates Compile_Manuscript_PDF_Colab.ipynb
A fully self-contained Google Colab notebook that compiles main.tex + references.bib
+ all vector PDF figures into the publication PDF and automatically triggers a browser download.
"""
import json
import base64
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Read main.tex
with open(ROOT / "manuscript" / "main.tex", "r", encoding="utf-8") as f:
    main_tex = f.read()

# Read references.bib
with open(ROOT / "manuscript" / "references.bib", "r", encoding="utf-8") as f:
    references_bib = f.read()

# Read and base64 encode vector PDF figures
figures_b64 = {}
fig_dir = ROOT / "manuscript" / "figures"
for fig_name in [
    "Fig1_MultiSeed_TriBenchmark_ConfidenceCurves.pdf",
    "Fig2_MultiSeed_BoxPlot_70pct_Dropout.pdf",
    "Fig2_Calibrated_Uncertainty_Trajectory.pdf"
]:
    with open(fig_dir / fig_name, "rb") as f:
        figures_b64[fig_name] = base64.b64encode(f.read()).decode("ascii")

cells = []

# Cell 1: Markdown
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# 1-Click LaTeX PDF Compiler for CT-PhysProb-JEPA Manuscript\n",
        "**Target Venues:** IEEE TNNLS / Elsevier RESS / EAAI.\n",
        "\n",
        "This notebook installs TeX Live in Google Colab, unpacks the complete self-contained paper sources (source code, references, and vector PDF figures), compiles the document with BibTeX bibliography resolution, and automatically downloads `CT_PhysProb_JEPA_Manuscript.pdf` to your local machine.\n",
        "\n",
        "**Instructions:** Simply click **Runtime -> Run all** (takes ~45 seconds)."
    ]
})

# Cell 2: Code for Setup & File Unpacking
code_unpack = f'''import os
import base64

print("[1/4] Unpacking manuscript source files and vector PDF figures...")
os.makedirs("/content/manuscript/figures", exist_ok=True)

# Write main.tex
with open("/content/manuscript/main.tex", "w", encoding="utf-8") as f:
    f.write({repr(main_tex)})

# Write references.bib
with open("/content/manuscript/references.bib", "w", encoding="utf-8") as f:
    f.write({repr(references_bib)})

# Unpack vector figures
figures_data = {repr(figures_b64)}

for fname, b64_str in figures_data.items():
    fpath = os.path.join("/content/manuscript/figures", fname)
    with open(fpath, "wb") as f:
        f.write(base64.b64decode(b64_str))
    print(f"  * Extracted figure: {{fname}} ({{os.path.getsize(fpath)}} bytes)")

print("[OK] Source files unpacked successfully!")
'''

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [s + "\n" for s in code_unpack.split("\n")]
})

# Cell 3: Code for TeX Live Installation and PDF Compilation
code_compile = '''# Step 2: Install TeX Live and Compile PDF
import os
import sys
import subprocess
from google.colab import files

print("[2/4] Installing TeX Live compiler in Colab (~35-45 seconds)...")
os.system("apt-get update -qq > /dev/null 2>&1")
os.system("apt-get install -y -qq texlive-latex-base texlive-latex-recommended texlive-fonts-recommended texlive-latex-extra texlive-bibtex-extra > /dev/null 2>&1")
print("[OK] TeX Live installed successfully!")

print("\\n[3/4] Compiling IEEEtran LaTeX document with BibTeX...")
work_dir = "/content/manuscript"

# Pass 1: pdflatex
print("  * Running pdflatex (Pass 1)...")
subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"], cwd=work_dir, stdout=subprocess.DEVNULL)

# Pass 2: bibtex
print("  * Running bibtex citations resolver...")
subprocess.run(["bibtex", "main"], cwd=work_dir, stdout=subprocess.DEVNULL)

# Pass 3: pdflatex
print("  * Running pdflatex (Pass 2)...")
subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"], cwd=work_dir, stdout=subprocess.DEVNULL)

# Pass 4: pdflatex final cross-references
print("  * Running pdflatex (Pass 3 - Final)...")
subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"], cwd=work_dir, stdout=subprocess.DEVNULL)

compiled_pdf = os.path.join(work_dir, "main.pdf")
output_pdf = "/content/CT_PhysProb_JEPA_Manuscript.pdf"

if os.path.exists(compiled_pdf):
    os.system(f"cp {compiled_pdf} {output_pdf}")
    pdf_size = os.path.getsize(output_pdf)
    print(f"\\n[SUCCESS] PDF compiled successfully! Size: {pdf_size} bytes ({pdf_size/1024:.1f} KB)")
    
    # Also save copy to Google Drive if mounted
    drive_target = "/content/drive/MyDrive/CT_PhysProb_JEPA_Datasets/CT_PhysProb_JEPA_Manuscript.pdf"
    if os.path.exists("/content/drive/MyDrive"):
        os.system(f"cp {output_pdf} {drive_target}")
        print(f"  * Copied to Google Drive: {drive_target}")

    # Trigger browser direct download
    print("\\n[4/4] Triggering browser direct download...")
    files.download(output_pdf)
else:
    print("[ERROR] Compilation log preview:")
    log_file = os.path.join(work_dir, "main.log")
    if os.path.exists(log_file):
        with open(log_file, "r") as lf:
            print("".join(lf.readlines()[-60:]))
'''

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [s + "\n" for s in code_compile.split("\n")]
})

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "provenance": []
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

out1 = ROOT / "Compile_Manuscript_PDF_Colab.ipynb"
out2 = Path("C:/Users/UseR/Downloads/Compile_Manuscript_PDF_Colab.ipynb")

for p in [out1, out2]:
    with open(p, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Generated Colab PDF Compiler: {p} ({os.path.getsize(p)} bytes)")
