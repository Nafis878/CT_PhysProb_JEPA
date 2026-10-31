import os
import zipfile
import pandas as pd

base_dir = "/working_dir/c_345ec2334c102f49/extracted_archive/Continous"
out_dir = "/working_dir/c_345ec2334c102f49/artifacts/file_generation/ttl=63d/output"
os.makedirs(out_dir, exist_ok=True)

# 1. Concatenate the core experimental source scripts into master_all_experiments_source_code.py
core_scripts = [
    ("CORE PACKAGE: ctpj/repro.py", "ctpj/repro.py"),
    ("CORE PACKAGE: ctpj/splits.py", "ctpj/splits.py"),
    ("CLI ENTRYPOINT: run.py", "run.py"),
    ("EXPERIMENT SUITE 1 (30-Seed 20-Model OOD & Real-Log Benchmark, SpecNorm & Integrator Audit): scripts/run_v10_all_real_experiments.py", "scripts/run_v10_all_real_experiments.py"),
    ("EXPERIMENT SUITE 2 (Multi-Dataset Table 3, 5-Fold CV, Ablations Table 4, Conformal Table 6-7, & 500-Traj CBM Table 8): scripts/run_v10_part2_table3_table4_and_claim3.py", "scripts/run_v10_part2_table3_table4_and_claim3.py"),
    ("EXPERIMENT SUITE 3 (Clamp Saturation Curve, Cubic Hermite C2 Spline, Energy-Norm Duhamel Bound, Jury Stability, & ODE Convergence Order): scripts/run_v11_math_overhaul_verification.py", "scripts/run_v11_math_overhaul_verification.py"),
    ("EXPERIMENT SUITE 4 (End-to-End PyTorch Training, 17 Competitors, Autograd Symplecticity, & Stiffness/NFE Audit): scripts/run_ress_must_haves_suite.py", "scripts/run_ress_must_haves_suite.py"),
    ("EXPERIMENT SUITE 5 (N-CMAPSS DS02 9-Unit LOUO & PRONOSTIA 17-Bearing LOBO Cross-Validation): scripts/run_ds02_and_pronostia_lobo.py", "scripts/run_ds02_and_pronostia_lobo.py"),
    ("PUBLICATION FIGURE GENERATOR (Figs. 1-4): scripts/generate_publication_figures.py", "scripts/generate_publication_figures.py"),
]

master_py_path = os.path.join(out_dir, "master_all_experiments_source_code.py")
with open(master_py_path, "w") as out_f:
    out_f.write('"""\n')
    out_f.write("MASTER CONSOLIDATED SOURCE CODE FOR ALL CT-PhysProb-JEPA EXPERIMENTS\n")
    out_f.write("====================================================================\n")
    out_f.write("Contains the complete, executable Python implementation of:\n")
    for title, rel_p in core_scripts:
        out_f.write(f"  - {title} ({rel_p})\n")
    out_f.write('"""\n\n')
    for title, rel_p in core_scripts:
        full_p = os.path.join(base_dir, rel_p)
        if os.path.exists(full_p):
            out_f.write("# " + "=" * 78 + "\n")
            out_f.write(f"# MODULE: {title}\n")
            out_f.write(f"# PATH:   {rel_p}\n")
            out_f.write("# " + "=" * 78 + "\n\n")
            with open(full_p, "r") as in_f:
                out_f.write(in_f.read())
            out_f.write("\n\n")

print(f"Wrote {master_py_path} ({os.path.getsize(master_py_path)/1024:.1f} KB)")

# 2. Build EXPERIMENTS_SOURCE_AND_RESULTS_CATALOG.md with full tables of every experiment
res_dir = os.path.join(base_dir, "results/ress_revision")

def df_to_md(csv_name, max_rows=None):
    p = os.path.join(res_dir, csv_name)
    if not os.path.exists(p):
        return f"*(File {csv_name} not found)*"
    df = pd.read_csv(p)
    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows)
    return df.to_markdown(index=False)

catalog_md = f"""# Complete Source Code & Experimental Results Catalog (CT-PhysProb-JEPA)

This document catalogs **every experiment** conducted for the paper, mapping each experiment to its **exact Python source script**, **output CSV/JSON files**, and **complete numerical results**.

---

## 1. Index of Experimental Source Scripts & Output Result Files

| Exp # | Experiment Name & Manuscript Section/Table | Primary Python Source Script(s) | Output Result CSV / JSON File(s) |
| :---: | :--- | :--- | :--- |
| **Exp 1** | **30-Seed 20-Model FD001 Benchmark (ID, OOD 5x–10x, OOD 10x–20x, OOD 15x–25x, Markov ACARS, Cruise Throttle) + Causal Control (`Plain_2ndOrder_NODE_Kin`)** (`Table 1`, `Fig. 2(a)`, `Fig. 3(a)`) | `scripts/run_v10_all_real_experiments.py` <br>`scripts/run_ress_must_haves_suite.py` | `results/ress_revision/30Seed_20Models_OOD_and_RealLogs.csv` <br>`results/ress_revision/30Seed_HolmBonferroni_Summary.csv` <br>`results/ress_revision/real_telemetry_gap_distributions.json` |
| **Exp 2** | **Clamp Saturation Curve $\\widehat{{\\mathrm{{RUL}}}}(\\Delta t)$ Across $\\Delta t \\in [1\\tau_0, 25\\tau_0]$ & True-RUL Degradation Slope Breakdown (`Slope = -1`, `Knee`, `Plateau`)** (`Prop. 1`, `Table 2`, `Fig. 3(b)`) | `scripts/run_v11_math_overhaul_verification.py` <br>`scripts/run_v10_part2_table3_table4_and_claim3.py` | `results/ress_revision/RUL_vs_Blackout_Duration_Curve.csv` <br>`results/ress_revision/Adversarial_OOD_Slope_Breakdown.csv` <br>`results/ress_revision/PerEngine_Blackout_Trajectory.csv` |
| **Exp 3** | **5-Dataset Generalization (FD001 5-Fold Unit CV, N-CMAPSS DS02 9-Unit LOUO, FD004, FD002, PRONOSTIA 17-Bearing LOBO)** (`Table 3`, `Fig. 2(b–d)`) | `scripts/run_v10_part2_table3_table4_and_claim3.py` <br>`scripts/run_ds02_and_pronostia_lobo.py` | `results/ress_revision/FD001_5Fold_UnitCV_and_Complete_Table3.csv` <br>`results/ress_revision/Complete_Table3_MultiDataset_20Models.csv` <br>`results/ress_revision/NCMAPSS_9Unit_LOUO_CV.csv` <br>`results/ress_revision/PRONOSTIA_17Bearings_LOBO.csv` <br>`results/ress_revision/5Seed_FD004_MultiFault_Benchmark.csv` <br>`results/ress_revision/5Seed_FD002_MultiCondition_Benchmark.csv` |
| **Exp 4** | **16-Row Architecture, Horizon-Policy ($h_{{\\max}} \\in \\{{0.20, 0.35, 0.50, 1.0, \\infty\\}}$), Adaptive Substepping ($M_k$), & Dimension ($2d=64$) Ablation Matrix** (`Table 4`) | `scripts/run_v10_part2_table3_table4_and_claim3.py` <br>`scripts/run_ablations_and_true_ct.py` | `results/ress_revision/Ablation_MatchedAnchor_hmax_Dim.csv` <br>`results/ress_revision/10Seed_FD001_Ablations.csv` |
| **Exp 5** | **True ODE Convergence Order ($M \\in \\{{1, 2, 4, 8, 16\\}}$ vs. 2000-Step Float64 Reference) & Equal-NFE Rollout Benchmark (Strang vs. RK4, Heun RK2, Explicit Euler)** (`Table 5`, `Fig. 3(c)`) | `scripts/run_v11_math_overhaul_verification.py` <br>`scripts/verify_symplecticity.py` | `results/ress_revision/Symplectic_Convergence_and_NFE_Audit.csv` <br>`results/ress_revision/Controlled_Integrator_Energy_vs_RMSE_Regression.csv` <br>`results/ress_revision/Symplectic_LongHorizon_Energy_Benchmark.csv` <br>`results/ress_revision/symplecticity_verification.json` |
| **Exp 6** | **Backbone-Agnostic Bias-Aware Mondrian Drift-Diffusion Conformal Prediction Across 8 Backbones & True-RUL Strata** (`Table 6`, `Fig. 4(b)`) | `scripts/run_v10_part2_table3_table4_and_claim3.py` <br>`scripts/refine_conformal_and_cbm.py` | `results/ress_revision/Conformal_DriftDiffusion_and_Mondrian_ACI.csv` <br>`results/ress_revision/Conditional_Coverage_By_RUL_Bin.csv` <br>`results/ress_revision/Conditional_Coverage_Per_Engine.csv` |
| **Exp 7** | **Importance-Weighted Conformal Prediction Under Unseen Operational Outage Shifts (`ID 25% -> OOD 10x–20x`, `Markov ACARS`, `70% MCAR`)** (`Table 7`) | `scripts/run_v10_part2_table3_table4_and_claim3.py` | `results/ress_revision/Conformal_Unseen_Outage_Shift.csv` |
| **Exp 8** | **Validation-Tuned 500-Trajectory Condition-Based Maintenance (CBM) Study Across 10 Architectures, 4 Policies, & Cost-Ratio Sweeps** (`Table 8`, `Fig. 4(c)`) | `scripts/run_v10_part2_table3_table4_and_claim3.py` <br>`scripts/build_full_32_cbm.py` | `results/ress_revision/CBM_Sensitivity_Analysis.csv` <br>`results/ress_revision/RESS_Maintenance_Decision_CaseStudy.csv` |
| **Exp 9** | **10-Seed Mathematical Verification Audit ($C^2$ Cubic Hermite Reference on 1-Cycle Dense Oracle, Centered-Stiffness Hessian $\\underline{{\\kappa}}$, Energy-Norm Duhamel Bound, & Jury Stability)** (`Appendix A`, `Table 9` in v11) | `scripts/run_v11_math_overhaul_verification.py` <br>`scripts/run_v10_all_real_experiments.py` | `results/ress_revision/Theorem3_v11_CubicHermite_ENorm_Audit.csv` <br>`results/ress_revision/SpecNorm_Contractive_10Seed_Audit.csv` <br>`results/ress_revision/Theorem3_Empirical_Constants.csv` |
| **Exp 10** | **Small-Sample Fleet Regimes ($N_{{\\mathrm{{train}}}} \\in \\{{10, 20, 40, 80\\}}$), Stiffness/NFE Wall-Clock Profiling, & Sanity Controls** | `scripts/run_ood_and_small_data.py` <br>`scripts/run_ress_must_haves_suite.py` | `results/ress_revision/10Seed_SmallData_Fleet_Regimes.csv` <br>`results/ress_revision/Stiffness_NFE_WallClock_Benchmark.csv` <br>`results/ress_revision/Vectorized_ApplesToApples_Efficiency.csv` <br>`results/ress_revision/Sanity_Controls_Audit.csv` |

---

## 2. Complete Numerical Results for Every Experiment

### Experiment 1: 30-Seed 20-Model Benchmark & Holm–Bonferroni Summary (`30Seed_HolmBonferroni_Summary.csv`)
{df_to_md("30Seed_HolmBonferroni_Summary.csv")}

---

### Experiment 2A: Predicted $\\widehat{{\\mathrm{{RUL}}}}(\\Delta t)$ vs. Blackout Duration Curve (`RUL_vs_Blackout_Duration_Curve.csv`)
{df_to_md("RUL_vs_Blackout_Duration_Curve.csv")}

### Experiment 2B: Adversarial OOD Breakdown by True RUL Degradation Slope (`Adversarial_OOD_Slope_Breakdown.csv`)
{df_to_md("Adversarial_OOD_Slope_Breakdown.csv")}

---

### Experiment 3: 5-Fold Unit-Level Cross-Validation & Complete 20-Model Multi-Dataset Generalization (`FD001_5Fold_UnitCV_and_Complete_Table3.csv`)
{df_to_md("FD001_5Fold_UnitCV_and_Complete_Table3.csv")}

---

### Experiment 4: 30-Seed Architecture, Horizon-Policy, & Causal Attribution Ablation Matrix (`Ablation_MatchedAnchor_hmax_Dim.csv`)
{df_to_md("Ablation_MatchedAnchor_hmax_Dim.csv")}

---

### Experiment 5: True ODE Convergence Order & Equal-NFE Integrator Audit (`Symplectic_Convergence_and_NFE_Audit.csv`)
{df_to_md("Symplectic_Convergence_and_NFE_Audit.csv")}

---

### Experiment 6: True-RUL Conditional Conformal Coverage Across 8 Backbones (`Conformal_DriftDiffusion_and_Mondrian_ACI.csv`)
{df_to_md("Conformal_DriftDiffusion_and_Mondrian_ACI.csv")}

---

### Experiment 7: Importance-Weighted Conformal Prediction Under Unseen Outage Shift (`Conformal_Unseen_Outage_Shift.csv`)
{df_to_md("Conformal_Unseen_Outage_Shift.csv")}

---

### Experiment 8: 500-Trajectory Validation-Tuned Condition-Based Maintenance (CBM) Study (`CBM_Sensitivity_Analysis.csv`, Cost Ratio 10:1 Subset)
"""

df_cbm = pd.read_csv(os.path.join(res_dir, "CBM_Sensitivity_Analysis.csv"))
catalog_md += df_cbm[df_cbm["Cost_Ratio"] == "10:1"].to_markdown(index=False)

catalog_md += f"""

---

### Experiment 9: 10-Seed Mathematical Audit — $C^2$ Cubic Hermite Reference, Centered-Stiffness Hessian $\\underline{{\\kappa}}$, & Energy-Norm Duhamel Bound (`Theorem3_v11_CubicHermite_ENorm_Audit.csv`)
{df_to_md("Theorem3_v11_CubicHermite_ENorm_Audit.csv")}

---

### Experiment 10: Vectorized Efficiency & Small-Data Fleet Regimes (`Vectorized_ApplesToApples_Efficiency.csv` & `10Seed_SmallData_Fleet_Regimes.csv`)
#### 10A. Vectorized Efficiency (`Vectorized_ApplesToApples_Efficiency.csv`)
{df_to_md("Vectorized_ApplesToApples_Efficiency.csv")}

#### 10B. Small-Data Fleet Regimes (`10Seed_SmallData_Fleet_Regimes.csv`)
{df_to_md("10Seed_SmallData_Fleet_Regimes.csv")}
"""

catalog_path = os.path.join(out_dir, "EXPERIMENTS_SOURCE_AND_RESULTS_CATALOG.md")
with open(catalog_path, "w") as f:
    f.write(catalog_md)
print(f"Wrote {catalog_path} ({os.path.getsize(catalog_path)/1024:.1f} KB)")

# 3. Create CT_PhysProb_JEPA_Complete_Source_Code_and_Results.zip containing all source code, scripts, notebooks, results, and manuscript
zip_path = os.path.join(out_dir, "CT_PhysProb_JEPA_Complete_Source_Code_and_Results.zip")
include_dirs = ["ctpj", "configs", "tests", "scripts", "results", "manuscript"]
include_root_files = [
    "run.py",
    "RESULTS_LEDGER.csv",
    "CHANGELOG.md",
    "AUDIT_REPORT.md",
    "REVIEWER_RESPONSE.md",
    "Train_Eval_CT_PhysProb_JEPA_MultiSeed_Wilcoxon_Q1.ipynb",
    "Train_Eval_CT_PhysProb_JEPA_TriBenchmark_Q1.ipynb",
]

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    # Add catalog and master python file at root of zip
    zf.write(catalog_path, "EXPERIMENTS_SOURCE_AND_RESULTS_CATALOG.md")
    zf.write(master_py_path, "master_all_experiments_source_code.py")
    for rf in include_root_files:
        fp = os.path.join(base_dir, rf)
        if os.path.exists(fp):
            zf.write(fp, rf)
    for d in include_dirs:
        dp = os.path.join(base_dir, d)
        if not os.path.exists(dp):
            continue
        for root, dirs, files in os.walk(dp):
            # skip __pycache__
            dirs[:] = [x for x in dirs if x != "__pycache__"]
            for file in sorted(files):
                if file.endswith(".pyc") or file.endswith(".aux") or file.endswith(".log") or file.endswith(".out") or file.endswith(".blg"):
                    continue
                full_f = os.path.join(root, file)
                rel_f = os.path.relpath(full_f, base_dir)
                zf.write(full_f, rel_f)

print(f"Created {zip_path} ({os.path.getsize(zip_path)/(1024*1024):.2f} MB)")
