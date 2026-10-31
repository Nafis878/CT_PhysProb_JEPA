import os
import zipfile
import pandas as pd

base_dir = "/working_dir/c_345ec2334c102f49/extracted_archive/Continous"
out_dir = "/working_dir/c_345ec2334c102f49/artifacts/file_generation/ttl=63d/output"
res_dir = os.path.join(base_dir, "results/ress_revision")

df_hb = pd.read_csv(os.path.join(res_dir, "30Seed_HolmBonferroni_Summary.csv"))
df_hb_ood = df_hb[df_hb["Regime"] == "OOD_10x_20x"].copy()

df_blk = pd.read_csv(os.path.join(res_dir, "RUL_vs_Blackout_Duration_Curve.csv"))

df_slope = pd.read_csv(os.path.join(res_dir, "Adversarial_OOD_Slope_Breakdown.csv"))
df_slope_agg = df_slope[df_slope["Regime"] == "OOD_10x_20x"].groupby(["Model", "Slope_Regime"], as_index=False).agg(
    Mean_RMSE=("RMSE_Slope", "mean"),
    SD_RMSE=("RMSE_Slope", "std"),
    Mean_Bias=("Mean_Bias", "mean")
).round(2)

df_t3 = pd.read_csv(os.path.join(res_dir, "FD001_5Fold_UnitCV_and_Complete_Table3.csv"))
df_ab = pd.read_csv(os.path.join(res_dir, "Ablation_MatchedAnchor_hmax_Dim.csv"))
df_int = pd.read_csv(os.path.join(res_dir, "Symplectic_Convergence_and_NFE_Audit.csv"))
df_conf = pd.read_csv(os.path.join(res_dir, "Conformal_DriftDiffusion_and_Mondrian_ACI.csv"))
df_shift = pd.read_csv(os.path.join(res_dir, "Conformal_Unseen_Outage_Shift.csv"))

df_cbm = pd.read_csv(os.path.join(res_dir, "CBM_Sensitivity_Analysis.csv"))
df_cbm_10 = df_cbm[(df_cbm["Cost_Ratio"] == "10:1") & (df_cbm["C_waste_USD_per_cycle"] == 450)].copy()

df_thm = pd.read_csv(os.path.join(res_dir, "Theorem3_v11_CubicHermite_ENorm_Audit.csv"))

df_eff = pd.read_csv(os.path.join(res_dir, "Vectorized_ApplesToApples_Efficiency.csv"))
df_eff_agg = df_eff.groupby("Model", as_index=False).agg(
    Train_s_ep=("Train_sec_per_epoch", "mean"),
    Infer_us_win=("Infer_us_per_window", "mean"),
    Infer_ms_eng=("Infer_ms_per_engine", "mean")
).round(3)

df_sm = pd.read_csv(os.path.join(res_dir, "10Seed_SmallData_Fleet_Regimes.csv"))
df_sm_agg = df_sm[df_sm["Regime"] == "OOD_10x_20x"].groupby(["N_Train_Engines", "Model"], as_index=False).agg(
    Mean_RMSE=("RMSE_All", "mean"),
    SD_RMSE=("RMSE_All", "std")
).round(2)

catalog_md = f"""# Complete Source Code & Experimental Results Catalog (CT-PhysProb-JEPA)

This catalog maps **every single experiment** conducted for the paper to its **exact Python source script**, **raw and aggregated CSV/JSON output files**, and **complete summary results tables**.

---

## 1. Master Index of Source Code Scripts & Result Files

| Exp # | Experiment Description & Manuscript Reference | Source Code Script(s) in `scripts/` | Output CSV / JSON Files in `results/ress_revision/` |
| :---: | :--- | :--- | :--- |
| **1** | **30-Seed 20-Model Benchmark Across ID, OOD ($5\\times\\text{{--}}25\\times$), Markov ACARS, & Cruise-Throttle + Causal Control (`Plain_2ndOrder_NODE_Kin`)** (`Table 1`, `Fig. 2(a)`, `Fig. 3(a)`) | `scripts/run_v10_all_real_experiments.py` <br>`scripts/run_ress_must_haves_suite.py` | `30Seed_20Models_OOD_and_RealLogs.csv` (3,780 rows) <br>`30Seed_HolmBonferroni_Summary.csv` (126 rows) <br>`HolmBonferroni_and_UnitBootstrap_Summary.csv` <br>`real_telemetry_gap_distributions.json` |
| **2** | **Clamp Saturation Curve $\\widehat{{\\mathrm{{RUL}}}}(\\Delta t)$ Across $\\Delta t \\in [1\\tau_0, 25\\tau_0]$ & True-RUL Slope Breakdown (`Slope = -1`, `Knee`, `Plateau`)** (`Prop. 1`, `Table 2`, `Fig. 3(b)`) | `scripts/run_v11_math_overhaul_verification.py` <br>`scripts/run_v10_part2_table3_table4_and_claim3.py` | `RUL_vs_Blackout_Duration_Curve.csv` (25 rows) <br>`Adversarial_OOD_Slope_Breakdown.csv` (1,260 rows) <br>`PerEngine_Blackout_Trajectory.csv` (265 rows) |
| **3** | **5-Dataset Generalization (FD001 5-Fold Unit CV, N-CMAPSS DS02 9-Unit LOUO, FD004, FD002, PRONOSTIA 17-Bearing LOBO)** (`Table 3`, `Fig. 2(b–d)`) | `scripts/run_v10_part2_table3_table4_and_claim3.py` <br>`scripts/run_ds02_and_pronostia_lobo.py` | `FD001_5Fold_UnitCV_and_Complete_Table3.csv` <br>`Complete_Table3_MultiDataset_20Models.csv` <br>`NCMAPSS_9Unit_LOUO_CV.csv` (405 rows) <br>`PRONOSTIA_17Bearings_LOBO.csv` (765 rows) <br>`LOUO_LOBO_PerUnit_Paired_Stats.csv` |
| **4** | **16-Row Architecture, Horizon-Policy ($h_{{\\max}} \\in \\{{0.20, 0.35, 0.50, 1.0, \\infty\\}}$), Adaptive Substepping ($M_k$), & Dimension ($2d=64$) Ablation Matrix** (`Table 4`) | `scripts/run_v10_part2_table3_table4_and_claim3.py` <br>`scripts/run_ablations_and_true_ct.py` | `Ablation_MatchedAnchor_hmax_Dim.csv` <br>`10Seed_FD001_Ablations.csv` (240 rows) |
| **5** | **True ODE Convergence Order ($M \\in \\{{1, 2, 4, 8, 16\\}}$ vs. 2000-Step Float64 Reference) & Equal-NFE Rollout Audit (Strang vs. RK4, Heun RK2, Explicit Euler)** (`Table 5`, `Fig. 3(c)`) | `scripts/run_v11_math_overhaul_verification.py` <br>`scripts/verify_symplecticity.py` | `Symplectic_Convergence_and_NFE_Audit.csv` <br>`Controlled_Integrator_Energy_vs_RMSE_Regression.csv` <br>`Symplectic_LongHorizon_Energy_Benchmark.csv` <br>`symplecticity_verification.json` |
| **6** | **Backbone-Agnostic Bias-Aware Mondrian Drift-Diffusion Conformal Prediction Across 8 Backbones & True-RUL Strata** (`Table 6`, `Fig. 4(b)`) | `scripts/run_v10_part2_table3_table4_and_claim3.py` <br>`scripts/refine_conformal_and_cbm.py` | `Conformal_DriftDiffusion_and_Mondrian_ACI.csv` <br>`Conditional_Coverage_By_RUL_Bin.csv` <br>`Conditional_Coverage_Per_Engine.csv` |
| **7** | **Importance-Weighted Conformal Prediction Under Unseen Operational Outage Shifts (`ID 25% -> OOD 10x–20x`, `Markov ACARS`, `70% MCAR`)** (`Table 7`) | `scripts/run_v10_part2_table3_table4_and_claim3.py` | `Conformal_Unseen_Outage_Shift.csv` |
| **8** | **Validation-Tuned 500-Trajectory Condition-Based Maintenance (CBM) Study Across 10 Architectures, 4 Policies, & Cost-Ratio Sweeps** (`Table 8`, `Fig. 4(c)`) | `scripts/run_v10_part2_table3_table4_and_claim3.py` <br>`scripts/build_full_32_cbm.py` | `CBM_Sensitivity_Analysis.csv` (480 rows) <br>`RESS_Maintenance_Decision_CaseStudy.csv` (320 rows) |
| **9** | **10-Seed Mathematical Audit ($C^2$ Cubic Hermite Reference on 1-Cycle Dense Oracle, Centered-Stiffness Hessian $\\underline{{\\kappa}}$, Energy-Norm Duhamel Bound, & Jury Stability)** (`Appendix A`) | `scripts/run_v11_math_overhaul_verification.py` <br>`scripts/run_v10_all_real_experiments.py` | `Theorem3_v11_CubicHermite_ENorm_Audit.csv` <br>`SpecNorm_Contractive_10Seed_Audit.csv` <br>`Theorem3_Empirical_Constants.csv` <br>`trained_weights_jacobian_audit.json` |
| **10** | **Small-Sample Fleet Regimes ($N_{{\\mathrm{{train}}}} \\in \\{{10, 20, 40, 80\\}}$), Stiffness/NFE Wall-Clock Profiling, & Sanity Controls** | `scripts/run_ood_and_small_data.py` <br>`scripts/run_ress_must_haves_suite.py` | `10Seed_SmallData_Fleet_Regimes.csv` (720 rows) <br>`Stiffness_NFE_WallClock_Benchmark.csv` <br>`Vectorized_ApplesToApples_Efficiency.csv` <br>`Sanity_Controls_Audit.csv` |

---

## 2. Complete Summary Results for Every Experiment

### Experiment 1: 30-Seed 20-Model Benchmark on Primary Endpoint `OOD 10x–20x` (`30Seed_HolmBonferroni_Summary.csv`)
{df_hb_ood.to_markdown(index=False)}

---

### Experiment 2A: Predicted $\\widehat{{\\mathrm{{RUL}}}}(\\Delta t)$ vs. Blackout Duration $\\Delta t \\in [1\\tau_0, 25\\tau_0]$ (`RUL_vs_Blackout_Duration_Curve.csv`)
{df_blk.to_markdown(index=False)}

### Experiment 2B: Aggregated OOD $10\\times\\text{{--}}20\\times$ Error Stratified by True RUL Slope (`Adversarial_OOD_Slope_Breakdown.csv`)
{df_slope_agg.to_markdown(index=False)}

---

### Experiment 3: 5-Fold Unit-Level Cross-Validation & Multi-Dataset Generalization (`FD001_5Fold_UnitCV_and_Complete_Table3.csv`)
{df_t3.to_markdown(index=False)}

---

### Experiment 4: 30-Seed Architecture, Horizon-Policy, & Causal Attribution Ablation Matrix (`Ablation_MatchedAnchor_hmax_Dim.csv`)
{df_ab.to_markdown(index=False)}

---

### Experiment 5: True ODE Convergence Order ($M \\in \\{{1, 2, 4, 8, 16\\}}$) & Equal-NFE Integrator Audit (`Symplectic_Convergence_and_NFE_Audit.csv`)
{df_int.to_markdown(index=False)}

---

### Experiment 6: True-RUL Conditional Conformal Coverage Across 8 Backbones (`Conformal_DriftDiffusion_and_Mondrian_ACI.csv`)
{df_conf.to_markdown(index=False)}

---

### Experiment 7: Importance-Weighted Conformal Prediction Under Unseen Outage Shift (`Conformal_Unseen_Outage_Shift.csv`)
{df_shift.to_markdown(index=False)}

---

### Experiment 8: 500-Trajectory Validation-Tuned Condition-Based Maintenance (CBM) Study (`CBM_Sensitivity_Analysis.csv`, Cost Ratio 10:1, $C_{{\\mathrm{{waste}}}}=\\$450$)
{df_cbm_10.to_markdown(index=False)}

---

### Experiment 9: 10-Seed Mathematical Verification Audit (`Theorem3_v11_CubicHermite_ENorm_Audit.csv`)
{df_thm.to_markdown(index=False)}

---

### Experiment 10A: Vectorized Computational Efficiency (`Vectorized_ApplesToApples_Efficiency.csv`)
{df_eff_agg.to_markdown(index=False)}

### Experiment 10B: Small-Data Fleet Regimes on `OOD 10x–20x` (`10Seed_SmallData_Fleet_Regimes.csv`)
{df_sm_agg.to_markdown(index=False)}
"""

catalog_path = os.path.join(out_dir, "EXPERIMENTS_SOURCE_AND_RESULTS_CATALOG.md")
with open(catalog_path, "w") as f:
    f.write(catalog_md)
print(f"Updated {catalog_path} ({os.path.getsize(catalog_path)/1024:.1f} KB)")

# Update the ZIP archive with the clean catalog
zip_path = os.path.join(out_dir, "CT_PhysProb_JEPA_Complete_Source_Code_and_Results.zip")
master_py_path = os.path.join(out_dir, "master_all_experiments_source_code.py")
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
            dirs[:] = [x for x in dirs if x != "__pycache__"]
            for file in sorted(files):
                if file.endswith((".pyc", ".aux", ".log", ".out", ".blg")):
                    continue
                full_f = os.path.join(root, file)
                rel_f = os.path.relpath(full_f, base_dir)
                zf.write(full_f, rel_f)

print(f"Updated {zip_path} ({os.path.getsize(zip_path)/(1024*1024):.2f} MB)")
