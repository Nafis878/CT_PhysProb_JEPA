# PREREGISTRATION PROTOCOL (v2 REVISION — DATED AMENDMENT v2.5)
**Document Version:** 2.5 (Dated Amendment before Confirmatory Runs)  
**Date of Execution:** 2026-11-01  
**Target Journal:** *Reliability Engineering & System Safety* (RESS)  
**Investigator:** Research Engineer and Statistician  
**Frozen Model Configuration Path:** `configs/frozen_final_configuration_v2.json`  
**Configuration SHA-256 Hash:** `5f8b33b5253bb903011ba391c4e348af4c17cf72dbf31c971a2c07117cad656d`  
**Baseline Tuned Configuration Path:** `configs/baseline_tuned_configurations.json`  
**Manifest Path:** `results/raw/CONFIRMATORY_MANIFEST.json`  
**Remote HTTPS Host:** `https://github.com/Nafis878/CT_PhysProb_JEPA.git`  
**Document History:**  
- `v2.0` (2026-11-01): Initial Step 2 post-development freeze protocol.  
- `v2.1` (2026-11-01): SHA-256 `827728fcd941cd19bee0a81cf40e92b27d84b2f8784c20ea3500fb747e3db138` (Phase 1 diagnostic corrections and recipe framing).  
- `v2.2` (2026-11-01): SHA-256 `a5a49d1cb902564d869ae7f51194f39afc4fc0460669c985199d5e4eabaf47c8` (Two-sided hypothesis structure, TOST justifications, equal-budget baseline sweep).  
- `v2.3` (2026-11-01): SHA-256 `5317da1e5579bf25571c2a7b93e120097c25d480c574ee1f1fec7be5df355ee6` (Engine-level cluster bootstrap, practical significance thresholds, separate H3/H4b family structure).  
- `v2.4` (2026-11-01): SHA-256 `34224e9aa4a797a66059426bd06e7e6980b5108099651b17a45d97da7a081a6b` (Independent FD003 calibration sampling, precise H1/H2 definitions, descriptive companions for bias, edge extension re-freezing, seed variability disclosure).  
- `v2.5` (2026-11-01): Current locked version (External HTTPS GitHub registration, rewritten empirical pre-data disclosures matching validation table, H4 single-verdict rule on RMSE_All only, common random numbers and window sensitivity protocol, complete grid sweep table, and restricted H3 power note).  
**Protocol Status:** LOCKED & FROZEN prior to unblinded confirmatory runs.  

---

## 1. Context, Protocol Provenance, & Contribution Framing

This pre-registration document formally amends and locks the confirmatory evaluation protocol following Step 2 model development and Phase 1 diagnostics.

### 1.1 Contribution Naming & Framing
The primary object of study and core contribution of this work is the **RECIPE** for blackout-robust continuous-time remaining useful life prediction:
$$\text{Recipe} = \{\text{Masked GRU-D Temporal Encoder} + \text{Closed-Form Knee-Hazard Countdown Head} + \text{Mixed-Gap Training Augmentation} + \text{Stochastic Weight Averaging (SWA)}\}$$
The **Recipe Family** evaluated comprises:
$$\mathcal{M}_{\text{Recipe}} = \{\text{CT\_PhysProb\_JEPA\_NoPhysics}, \text{GRU\_D\_Enhanced\_SWA}, \text{CT\_PhysProb\_JEPA\_v2\_SWA}\}$$
The Hamiltonian continuous-time latent is evaluated strictly as an additive continuous-time physics hypothesis on top of this recipe, **NOT** as the primary driver of performance.  
*Development Budget Disclosure:* The proposed model received three development rounds on FD001 validation units prior to freezing, while baseline models received the equal-budget validation hyperparameter sweep.

### 1.2 Formal ODE Limitation Statement & Diagnostic Metrics
Following exhaustive diagnostic analysis on validation units across optimization dynamics, relative parameter updates, and variance partitioning:
> *"The Hamiltonian latent as parameterized receives no consistent learning signal and contributes no measurable benefit on any pre-registered endpoint; cause not isolated."*

Empirical block-level parameter displacement from initialization to epoch 8 ($\frac{\|\theta_{\text{final}} - \theta_{\text{init}}\|}{\|\theta_{\text{init}}\|}$) on validation engines:
- `encoder`: $63.26\%$ ($\|\Delta W\| = 9.301, \|W_{\text{init}}\| = 14.704$)
- `head`: $17.98\%$ ($\|\Delta W\| = 1.133, \|W_{\text{init}}\| = 6.301$)
- `W_pot` (Hamiltonian Potential): $13.24\%$ ($\|\Delta W\| = 0.294, \|W_{\text{init}}\| = 2.217$)
- `b_pot`: $\|\Delta W\| = 0.0586$ ($\|W_{\text{init}}\| = 0.0$)
- `force_net`: $40.09\%$ ($\|\Delta W\| = 1.502, \|W_{\text{init}}\| = 3.746$)
- `dyn_probe`: $15.89\%$ ($\|\Delta W\| = 0.545, \|W_{\text{init}}\| = 3.431$)
- `decay_w` (Damping friction): $8.51\%$ ($\|\Delta W\| = 0.021, \|W_{\text{init}}\| = 0.250$)

Under AdamW ($\eta = 3.5 \times 10^{-3}, \lambda = 10^{-4}$), the weight decay step is $\eta \lambda \theta \approx 3.5 \times 10^{-7} \theta$ per step. While the latent ODE accounts for $87.7\%$ of prediction variance under dense in-distribution sequences, its variance share collapses to $<4\%$ during long-gap blackout outages, where the closed-form countdown hazard head dominates $>96\%$ of total prediction variance.

### 1.3 Anchor Reconciliation Diagnosis
- **Core Empirical Finding**:
  > *"97.9% of the dense-vs-onset anchor gap is attributable to sparse observation history; mechanism (EWMA staleness vs general information loss) not isolated."*
- **Empirical Quantification on Validation Engines**:
  - Dense in-distribution validation anchor RMSE: **$11.53$ cycles**.
  - Sparse outage onset anchor RMSE (actual strided history): **$15.26$ cycles** (total shift: $+3.73$ cycles).
  - Recomputed features for the **exact same 280 onset windows** under dense historical observations: **$11.60$ cycles** ($\Delta = -3.65$ cycles, recovering $97.9\%$ of the baseline accuracy).
  - Contemporaneous raw sensor error at onset cycle $t_k$ is identically zero ($0.0$).
  - Validation windows are confirmed to be $38.2\%$ plateau ($RUL=125$) and $61.8\%$ active degradation (earlier claim of $>65\%$ plateau retracted).
- **Exploratory Status of Phase 1 Stress Probes**: All preliminary stress tests evaluated during Phase 1 on 5 or 10 seeds are classified as **EXPLORATORY ONLY** and will not be reported as confirmatory scientific findings.

### 1.4 Full Equal-Budget Baseline Sweep & Edge Extension Table
Each baseline was tuned across learning rates and model dimensions over **5 validation seeds** (`[42, 100, 101, 102, 103]`) on FD001 validation units (345 total training runs logged in `results/raw/extended_baseline_sweep_raw.csv` and `results/raw/baseline_edge_extensions_raw.csv`).

| Model Name | Param Name | Grid Size ($S$) | Grid LR ($\eta$) | Val RMSE Mean $\pm$ SD | Val Bias Mean | Status After Extensions |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`GRU_D_Enhanced_SWA`** | `hidden_dim` | 24 | $2.5 \times 10^{-3}$ | $18.691 \pm 0.315$ | $+4.12$ | Grid Point |
| | | 24 | $5.0 \times 10^{-3}$ | $18.458 \pm 0.240$ | $+4.71$ | Grid Point |
| | | 24 | $7.5 \times 10^{-3}$ | $18.491 \pm 0.282$ | $+5.34$ | Grid Point |
| | | 32 | $2.5 \times 10^{-3}$ | $18.204 \pm 0.112$ | $+3.89$ | Grid Point |
| | | 32 | $5.0 \times 10^{-3}$ | $18.181 \pm 0.119$ | $+4.55$ | Grid Point |
| | | 32 | $7.5 \times 10^{-3}$ | $18.314 \pm 0.121$ | $+5.21$ | Grid Point |
| | | 48 | $2.5 \times 10^{-3}$ | $18.236 \pm 0.290$ | $+4.68$ | Grid Point |
| | | **48** | **$5.0 \times 10^{-3}$** | **$18.143 \pm 0.265$** | **$+5.18$** | **Best (Concave; size 64 degrades)** |
| | | 48 | $7.5 \times 10^{-3}$ | $18.292 \pm 0.292$ | $+5.51$ | Grid Point |
| | | 64 | $5.0 \times 10^{-3}$ | $18.234 \pm 0.280$ | $+5.32$ | Extension check |
| **`GRU_D_Canonical_SWA`** | `hidden_dim` | 48 | $1.5 \times 10^{-3}$ | $18.529 \pm 0.257$ | $+4.72$ | Grid Point |
| | | 48 | $3.5 \times 10^{-3}$ | $18.271 \pm 0.288$ | $+5.69$ | Grid Point |
| | | 48 | $5.0 \times 10^{-3}$ | $18.242 \pm 0.348$ | $+5.86$ | Grid Point |
| | | 64 | $1.5 \times 10^{-3}$ | $18.448 \pm 0.235$ | $+4.98$ | Grid Point |
| | | 64 | $3.5 \times 10^{-3}$ | $18.090 \pm 0.102$ | $+5.28$ | Grid Point |
| | | 64 | $5.0 \times 10^{-3}$ | $18.090 \pm 0.092$ | $+5.35$ | Grid Point |
| | | 80 | $1.5 \times 10^{-3}$ | $18.234 \pm 0.112$ | $+5.41$ | Grid Point |
| | | **80** | **$3.5 \times 10^{-3}$** | **$17.909 \pm 0.160$** | **$+5.84$** | **Best (Concave; size 96 degrades)** |
| | | 80 | $5.0 \times 10^{-3}$ | $17.940 \pm 0.115$ | $+6.25$ | Grid Point |
| | | 96 | $3.5 \times 10^{-3}$ | $18.076 \pm 0.202$ | $+5.05$ | Extension check |
| **`Plain_NODE_Enhanced_SWA`** | `latent_dim` | 16 | $1.5 \times 10^{-3}$ | $18.478 \pm 0.501$ | $+5.50$ | Grid Point |
| | | 16 | $3.5 \times 10^{-3}$ | $18.200 \pm 0.376$ | $+5.16$ | Grid Point |
| | | 16 | $5.0 \times 10^{-3}$ | $18.215 \pm 0.341$ | $+5.44$ | Grid Point |
| | | 24 | $1.5 \times 10^{-3}$ | $18.471 \pm 0.333$ | $+5.47$ | Grid Point |
| | | **24** | **$3.5 \times 10^{-3}$** | **$18.105 \pm 0.293$** | **$+5.82$** | **Interior Optimal** |
| | | 24 | $5.0 \times 10^{-3}$ | $18.200 \pm 0.309$ | $+5.95$ | Grid Point |
| | | 32 | $1.5 \times 10^{-3}$ | $18.605 \pm 0.340$ | $+5.84$ | Grid Point |
| | | 32 | $3.5 \times 10^{-3}$ | $18.345 \pm 0.225$ | $+6.38$ | Grid Point |
| | | 32 | $5.0 \times 10^{-3}$ | $18.334 \pm 0.243$ | $+6.06$ | Grid Point |
| **`PatchTST_RUL_v2`** | `d_model` | 48 | $1.0 \times 10^{-3}$ | $18.719 \pm 0.343$ | $+5.57$ | Grid Point |
| | | 48 | $1.5 \times 10^{-3}$ | $18.755 \pm 0.298$ | $+5.76$ | Grid Point |
| | | 48 | $3.5 \times 10^{-3}$ | $19.044 \pm 0.242$ | $+5.18$ | Grid Point |
| | | **64** | **$1.0 \times 10^{-3}$** | **$18.455 \pm 0.164$** | **$+5.25$** | **Best (Concave; lr 5e-4 degrades)** |
| | | 64 | $1.5 \times 10^{-3}$ | $18.503 \pm 0.161$ | $+5.28$ | Grid Point |
| | | 64 | $3.5 \times 10^{-3}$ | $18.736 \pm 0.155$ | $+5.59$ | Grid Point |
| | | 80 | $1.0 \times 10^{-3}$ | $18.751 \pm 0.141$ | $+6.10$ | Grid Point |
| | | 80 | $1.5 \times 10^{-3}$ | $18.888 \pm 0.234$ | $+6.36$ | Grid Point |
| | | 80 | $3.5 \times 10^{-3}$ | $18.923 \pm 0.345$ | $+5.89$ | Grid Point |
| | | 64 | $5.0 \times 10^{-4}$ | $18.734 \pm 0.146$ | $+5.54$ | Extension check |
| **`GRU_D_Original`** | `hidden_dim` | 32 | $1.5 \times 10^{-3}$ | $20.041 \pm 0.725$ | $+8.96$ | Grid Point |
| | | 32 | $3.5 \times 10^{-3}$ | $19.897 \pm 0.580$ | $+7.86$ | Grid Point |
| | | 32 | $5.0 \times 10^{-3}$ | $19.997 \pm 0.550$ | $+8.10$ | Grid Point |
| | | 48 | $1.5 \times 10^{-3}$ | $19.462 \pm 1.172$ | $+9.36$ | Grid Point |
| | | 48 | $3.5 \times 10^{-3}$ | $19.805 \pm 0.975$ | $+9.03$ | Grid Point |
| | | 48 | $5.0 \times 10^{-3}$ | $19.956 \pm 1.123$ | $+9.10$ | Grid Point |
| | | 64 | $1.5 \times 10^{-3}$ | $19.286 \pm 0.782$ | $+9.18$ | Grid Point |
| | | 64 | $3.5 \times 10^{-3}$ | $19.264 \pm 0.421$ | $+8.01$ | Initial best |
| | | 64 | $5.0 \times 10^{-3}$ | $19.387 \pm 0.396$ | $+7.94$ | Grid Point |
| | | **80** | **$3.5 \times 10^{-3}$** | **$19.046 \pm 0.455$** | **$+8.58$** | **Re-frozen via Extension (Edge)** |
| **`PatchTST_Original`** | `d_model` | 32 | $1.5 \times 10^{-3}$ | $22.327 \pm 1.835$ | $+4.71$ | Grid Point |
| | | 32 | $3.5 \times 10^{-3}$ | $22.330 \pm 1.760$ | $+6.28$ | Grid Point |
| | | 32 | $5.0 \times 10^{-3}$ | $22.663 \pm 1.443$ | $+5.34$ | Grid Point |
| | | 48 | $1.5 \times 10^{-3}$ | $22.228 \pm 2.070$ | $+12.51$ | Grid Point |
| | | **48** | **$3.5 \times 10^{-3}$** | **$21.397 \pm 0.397$** | **$+9.11$** | **Interior Optimal** |
| | | 48 | $5.0 \times 10^{-3}$ | $21.463 \pm 1.805$ | $+6.00$ | Grid Point |
| | | 64 | $1.5 \times 10^{-3}$ | $21.680 \pm 1.209$ | $+4.22$ | Grid Point |
| | | 64 | $3.5 \times 10^{-3}$ | $21.801 \pm 1.449$ | $+5.41$ | Grid Point |
| | | 64 | $5.0 \times 10^{-3}$ | $21.581 \pm 1.375$ | $+5.94$ | Grid Point |
| **`Plain_2ndOrder_NODE_Kin_Original`** | `latent_dim` | 12 | $1.5 \times 10^{-3}$ | $25.199 \pm 3.640$ | $-3.03$ | Grid Point |
| | | 12 | $3.5 \times 10^{-3}$ | $25.503 \pm 3.647$ | $-4.22$ | Grid Point |
| | | 12 | $5.0 \times 10^{-3}$ | $25.774 \pm 3.857$ | $-2.92$ | Grid Point |
| | | 16 | $1.5 \times 10^{-3}$ | $25.355 \pm 2.540$ | $-1.05$ | Grid Point |
| | | 16 | $3.5 \times 10^{-3}$ | $25.964 \pm 3.353$ | $-3.53$ | Grid Point |
| | | 16 | $5.0 \times 10^{-3}$ | $24.761 \pm 2.637$ | $-1.70$ | Grid Point |
| | | 24 | $1.5 \times 10^{-3}$ | $26.689 \pm 3.674$ | $-6.63$ | Grid Point |
| | | 24 | $3.5 \times 10^{-3}$ | $25.670 \pm 1.621$ | $-4.48$ | Grid Point |
| | | 24 | $5.0 \times 10^{-3}$ | $24.458 \pm 1.313$ | $-1.80$ | Initial best |
| | | 32 | $5.0 \times 10^{-3}$ | $23.650 \pm 2.062$ | $-1.56$ | Extension check |
| | | **24** | **$7.5 \times 10^{-3}$** | **$23.261 \pm 0.676$** | **$+3.36$** | **Re-frozen via Extension (Edge)** |

*Under-Tuning Limitation:* Baselines `GRU_D_Original` (`dim=80`) and `Plain_2ndOrder_NODE_Kin_Original` (`lr=7.5e-3`) improved upon extension and remain at grid edges. They are frozen as-is to preserve experimental boundaries, with the explicit disclosure that they may remain under-tuned relative to larger capacities.

### 1.5 Validation Seed Variability Diagnosis & Common Random Numbers
- **Investigation of Seed Variance Shifts**:
  - In Historical Step 2 development (10 seeds), the seed simultaneously varied parameter initialization, minibatch permutation, and **validation blackout onset window sampling** (`seed = s + 4000`), yielding an SD of **$0.8367$ cycles** (Mean = $18.947$).
  - When validation onset windows are fixed (`seed = 4042`, common random numbers across models), isolating training stochasticity (weight initialization `torch.manual_seed(seed)` and SGD ordering `torch.randperm(N)`):
    - Across the extended sweep 5 seeds (`[42, 100, 101, 102, 103]`): SD = **$0.3753$ cycles** (Mean = $18.208$).
    - Across the original 10 seeds under fixed validation windows: SD = **$0.2384$ cycles** (Mean = $18.108$).
- **Common Random Numbers Protocol**:
  Blackout onset windows for each evaluated outage regime are generated **once from a fixed seed** and are identical across all models and training seeds. The training seed controls strictly parameter initialization and minibatch ordering.
- **Sensitivity Window Protocol**:
  Because the engine-level cluster bootstrap accounts for engine sampling but not onset-window sampling, a descriptive sensitivity analysis will be reported using a second, independently generated window set (evaluated descriptive-only, not part of any formal hypothesis test).

### 1.6 Pre-Data Disclosures & Empirical Expectations
Based on complete validation data logged to `results/raw/`:
1. **Recipe Effect (H4)**:
   Expected to be supported on Primary Endpoint `RMSE_All` ($p_{\text{Holm}} < 0.05$, bootstrap CI excluding 0, and $|\bar{\Delta}| \ge 0.5\text{ cycles}$). On active signed bias, however, original baselines do **NOT** uniformly under-predict: `PatchTST_Original` ($-1.701$) under-predicts, while `Plain_2ndOrder_NODE_Kin` ($+3.360$) and `GRU_D_Original` ($+8.580$) over-predict. Furthermore, `GRU_D_Original` bias shifted from $+4.12$ to $+8.58$ under tuning from size 64 to 80, proving signed bias is highly tuning-sensitive. Hence, H4 bias outcomes are expected to be mixed, and H4 is decided on `RMSE_All` only.
2. **Additive Continuous-Time Physics Benefit (H5)**:
   - *Primary Endpoint RMSE:* In validation, the proposed model achieved lower RMSE than `NoPhysics` on all 5 of 5 seeds (`diffs: [-0.178, -0.248, -0.067, -0.082, -0.255]`, paired mean $\bar{\Delta} = -0.1661\text{ cycles}$, paired SD $s_{\Delta} = 0.0891\text{ cycles}$, paired $t = -4.1677, p = 0.0141$). Because this difference is statistically detectable ($p < 0.05$) but strictly below the locked practical threshold ($|\bar{\Delta}| = 0.166 < 0.50\text{ cycles}$), the pre-registered expected verdict is:
     > **`STATISTICALLY DETECTABLE BUT NOT PRACTICALLY RELEVANT`**
   - *Active Degradation Bias:* `NoPhysics` achieved lower validation bias ($+4.380$ cycles) than the proposed model ($+5.642$ cycles), a gap of $1.262\text{ cycles}$ exceeding the $1.0$-cycle threshold in favor of the non-physics baseline. Consequently, H5 on signed bias is expected to yield:
     > **`PHYSICS HARMS`**
3. **Integrator Symplecticity Equivalence (H3)**:
   Expected to confirm practical equivalence within $\delta = \pm 0.25$ cycles based on near-identical validation performance ($s_{\Delta} \le 0.02$).
4. **No Superiority Claim**:
   There is no pre-registered BEST / COMPETITIVE superiority verdict versus all models, and the paper will make no "best model" claim.

---

## 2. Extended Symmetry & Re-Frozen Baseline Configurations

All 11 evaluated architectures are frozen and specified in [`configs/baseline_tuned_configurations.json`](file:///C:/Users/UseR/Downloads/Continous/configs/baseline_tuned_configurations.json):

| # | Model Name | Temporal Encoder | Prediction Head | Latent Integrator / Dynamics | SWA (Last 3 Ep.) | Explicit Mask $m \in \{0, 1\}^{14}$ | Training Recipe | Epochs / Batch Size | Optimizer ($lr$, $wd$) | Tuned Parameters (Equal Budget) | Val RMSE (5 Seeds) | Input Dim ($D$) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `CT_PhysProb_JEPA_v2_SWA` | GRU-D Temporal (dim 48) | KneeHazardHead (closed-form) | Hamiltonian Strang-symplectic (damped) | Yes (ep 5-7) | Yes (concatenated) | Mixed gaps (short + stride 3-15) | 8 ep / 512 | AdamW ($3.5 \times 10^{-3}$, $10^{-4}$) | Frozen (3 dev rounds) | $18.208 \pm 0.375$ | 56 |
| 2 | `NoPhysics` | GRU-D Temporal (dim 48) | KneeHazardHead (closed-form) | None ($\Delta y_{\text{ODE}} = 0$) | Yes (ep 5-7) | Yes (concatenated) | Mixed gaps (short + stride 3-15) | 8 ep / 512 | AdamW ($3.5 \times 10^{-3}$, $10^{-4}$) | Frozen matched control | $18.374 \pm 0.417$ | 56 |
| 3 | `Euler_Damped` (Descriptive) | GRU-D Temporal (dim 48) | KneeHazardHead (closed-form) | Forward Euler (damped, $\gamma > 0$) | Yes (ep 5-7) | Yes (concatenated) | Mixed gaps (short + stride 3-15) | 8 ep / 512 | AdamW ($3.5 \times 10^{-3}$, $10^{-4}$) | Frozen matched control | — | 56 |
| 4 | `Undamped` (Descriptive) | GRU-D Temporal (dim 48) | KneeHazardHead (closed-form) | Hamiltonian Strang-symplectic ($\gamma = 0$) | Yes (ep 5-7) | Yes (concatenated) | Mixed gaps (short + stride 3-15) | 8 ep / 512 | AdamW ($3.5 \times 10^{-3}$, $10^{-4}$) | Frozen matched control | — | 56 |
| 5 | `GRU_D_Enhanced_SWA` | Canonical GRU-D Cell | KneeHazardHead (closed-form) | None (recurrent decay) | Yes (ep 5-7) | Yes (concatenated) | Mixed gaps (short + stride 3-15) | 8 ep / 512 | AdamW ($5.0 \times 10^{-3}$, $10^{-4}$) | Tuned: `dim=48`, `lr=5e-3` | $18.143 \pm 0.265$ | 56 |
| 6 | `GRU_D_Canonical_SWA` | Canonical GRU-D Cell | Linear-Tanh-Linear Probe | None (mask-driven decay) | Yes (ep 5-7) | Yes (drives decay toward last obs) | Mixed gaps (short + stride 3-15) | 8 ep / 512 | AdamW ($3.5 \times 10^{-3}$, $10^{-4}$) | Tuned: `dim=80`, `lr=3.5e-3` | $17.909 \pm 0.160$ | 56 |
| 7 | `Plain_NODE_Enhanced_SWA` (Descriptive) | GRU-D Temporal (dim 48) | KneeHazardHead (closed-form) | 2nd-order Neural ODE (Euler 4-step) | Yes (ep 5-7) | Yes (concatenated) | Mixed gaps (short + stride 3-15) | 8 ep / 512 | AdamW ($3.5 \times 10^{-3}$, $10^{-4}$) | Tuned: `latent=24`, `lr=3.5e-3` | $18.105 \pm 0.293$ | 56 |
| 8 | `PatchTST_RUL_v2` (Descriptive) | Patched Transformer (5 tokens) | KneeHazardHead (closed-form) | None | Yes (ep 5-7) | Yes (dedicated mask token) | Mixed gaps (short + stride 3-15) | 8 ep / 512 | AdamW ($1.0 \times 10^{-3}$, $10^{-4}$) | Tuned: `d_model=64`, `lr=1e-3` | $18.455 \pm 0.164$ | 56 |
| 9 | `GRU_D` (original) | Canonical GRU-D Cell | Linear-Tanh-Linear Probe | None | No* | No* (zero-fill) | Short dropouts only ($p=0.25$)* | 8 ep / 512 | AdamW ($3.5 \times 10^{-3}$, $10^{-4}$) | Re-frozen: `dim=80`, `lr=3.5e-3` | $19.046 \pm 0.455$ | 42 |
| 10 | `PatchTST_RUL` (original) | Patched Transformer (3 tokens) | Linear-Tanh-Linear Probe | None | No* | No* | Short dropouts only ($p=0.25$)* | 8 ep / 512 | AdamW ($3.5 \times 10^{-3}$, $10^{-4}$) | Tuned: `d_model=48`, `lr=3.5e-3` | $21.397 \pm 0.397$ | 42 |
| 11 | `Plain_2ndOrder_NODE_Kin` (original) | Linear-Tanh-Linear MLP | Linear-Tanh-Linear Probe | Kinematic Euler ($M=1$) | No* | No* | Short dropouts only ($p=0.25$)* | 8 ep / 512 | AdamW ($7.5 \times 10^{-3}$, $10^{-4}$) | Re-frozen: `latent=24`, `lr=7.5e-3` | $23.261 \pm 0.676$ | 42 |

*\*Intentional Asymmetries: Original benchmarks 9–11 lack mixed-gap augmentation, KneeHazardHead, binary mask vectors, and SWA by historical definition to quantify cumulative recipe benefits.*

### 2.1 Descriptive Model Classification
Models `PatchTST_RUL_v2`, `Plain_NODE_Enhanced_SWA`, `Euler_Damped`, `Undamped`, `EstimatedAnchor_Countdown`, and `TimeCorrected_LOCF_Oracle` are classified as **DESCRIPTIVE**. Their results are reported for scientific completeness without multiplicity penalties.

---

## 3. Inference Units, Dual Significance Criteria, & Practical Thresholds

### 3.1 Inference Units
1. **Seed-Level Inference**: Paired Student's $t$-test and Wilcoxon signed-rank test across $n = 30$ fixed random seeds.
2. **Engine-Level Paired Cluster Bootstrap**:
   - Resamples the 100 official test engines with replacement ($B = 10,000$ resamples).
   - Predictions are averaged across the 30 seeds per engine before computing metric differences.
   - Computes non-parametric percentile 95% confidence intervals ($[q_{0.025}, q_{0.975}]$).

### 3.2 Practical Significance Thresholds
- **RMSE Threshold**: $|\Delta_{\text{RMSE}}| \ge 0.50\text{ cycles}$.
- **Signed Bias Threshold**: $|\Delta_{\text{Bias}}| \ge 1.00\text{ cycle}$.

### 3.3 Dual Significance Decision Rule
A formal **`SUPPORTED`** or **`HARMS`** verdict requires satisfying **ALL THREE** criteria simultaneously:
1. Two-sided joint Holm-Bonferroni adjusted $p$-value $p_{\text{Holm}} < 0.05$ across seeds, **AND**
2. Engine-level paired cluster bootstrap 95% CI strictly excludes 0, **AND**
3. Paired sample mean difference $|\Delta| \ge \text{practical threshold}$ ($0.5$ cycles for RMSE, $1.0$ cycle for bias).

*Sub-Threshold Designation:* Any comparison where $p_{\text{Holm}} < 0.05$ and the bootstrap CI excludes 0, but $|\Delta| < \text{threshold}$, is formally labeled:
**`STATISTICALLY DETECTABLE BUT NOT PRACTICALLY RELEVANT`**.

### 3.4 Interpretation of Signed Bias & Descriptive Companions
Because signed bias can be moved closer to zero by under-prediction, active signed bias must be read alongside RMSE, the NASA score, and late-prediction error $\text{mean}(\max(\hat{y} - y^*, 0))$.

---

## 4. Evaluation Protocols, Data Partitions, & Endpoints

### 4.1 Primary & Secondary Endpoints
- **Primary Endpoint**: Total RMSE across all blackout windows (`RMSE_All`) under Out-of-Distribution Long Outages ($10\text{x}\text{--}20\text{x}$ stride) on the 100 official NASA C-MAPSS FD003 test engines.
- **Secondary Endpoints**:
  1. `RMSE_Last`: Prediction RMSE at final operational cycle before failure.
  2. `Active-Degradation Signed Bias`: Mean error $(\hat{y} - y^*)$ over windows with active degradation ($y^* < 125$ cycles).
  3. `NASA Asymmetric Scoring Function`: Asymmetric penalty function.
  4. `Anchor RMSE`: Prediction error at blackout onset $t_k$.
  5. `Gap-Induced RMSE`: Total RMSE minus Anchor RMSE.
  6. `Conformal Coverage & Interval Width`: Empirical coverage and width at nominal $90\%$ coverage ($1 - \alpha = 0.90$).
  7. `Missed Critical Failures`: Percentage of engines where actual RUL is $\le 10$ cycles but predicted lower bound exceeds 15 cycles.
  8. `Expected Condition-Based Maintenance (CBM) Cost`: Cost per unit under $C_u / C_p = 5.0$, with intervention threshold fixed on validation units.

### 4.2 Evaluated Outage & Stress Regimes
1. `In-Distribution`: Nominal continuous observation ($dt=1$, random dropout $p=0.25$).
2. `OOD 10x-20x`: Primary blackout horizon (stride $10\text{--}20$ cycles).
3. `OOD 15x-25x`: Extended blackout horizon (stride $15\text{--}25$ cycles).
4. `OOD 20x-30x`: Extreme blackout horizon (stride $20\text{--}30$ cycles).
5. `Sensor Dropout 10%`: Random channel masking of $10\%$ of sensors during OOD 10x-20x outage.
6. `Sensor Dropout 20%`: Random channel masking of $20\%$ of sensors during OOD 10x-20x outage.
7. `Sensor Dropout 30%`: Extreme channel masking of $30\%$ of sensors during OOD 10x-20x outage.

### 4.3 Conformal Prediction & CBM Protocol
- Non-conformity score $s_i = |y_i - \hat{y}_i|$.
- Calibration evaluated on the 20 held-out training engines to compute $\hat{q}_{\alpha}$.
- Intervention threshold $\tau_{\text{CBM}}$ and cost ratio ($C_u / C_p = 5.0$) fixed strictly on validation units.

### 4.4 Confirmatory Data Partition & Separation Protocol
- **Dataset**: NASA C-MAPSS FD003.
- **20 Conformal Calibration Engines**: `[3, 12, 18, 23, 36, 46, 49, 51, 58, 62, 63, 64, 69, 73, 75, 76, 77, 92, 97, 99]` (independently sampled with `seed=20261101`). Never trained on.
- **80 Training Engines for Official Test Models**: `[1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 47, 48, 50, 52, 53, 54, 55, 56, 57, 59, 60, 61, 65, 66, 67, 68, 70, 71, 72, 74, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 93, 94, 95, 96, 98, 100]`.
- **5-Fold Cross-Validation Rotation**: 80 engines partitioned into 5 folds (16 engines each).
- **Official Test Set (100 engines in `test_FD003.txt` + `RUL_FD003.txt`)**: Evaluated once per seed at frozen SWA checkpoint (epochs 5–7).

---

## 5. Confirmatory Hypotheses & Separate Test Families

All hypotheses evaluated across **30 fixed random seeds**:
```python
SEEDS_30 = [
    42, 100, 101, 102, 103, 104, 105, 106, 107, 108, 
    109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 
    119, 120, 121, 122, 123, 124, 125, 126, 127, 128
]
```

### 5.1 Primary Hypothesis Family ($M = 40$ Tests, Joint Holm-Bonferroni at $\alpha = 0.05$)

#### Hypothesis H1: Sensor Channel Dropout Resilience (Two-Sided, 12 Tests)
- **Regime**: Out-of-Distribution $10\text{x}\text{--}20\text{x}$ outage with sensor dropouts $\rho \in \{10\%, 20\%, 30\%\}$.
- **Endpoint**: Primary Endpoint `RMSE_All` (12 tests total).

#### Hypothesis H2: Extreme Outage Optimistic Bias Suppression (Two-Sided, 8 Tests)
- **Regimes**: Extended outage ($15\text{x}\text{--}25\text{x}$) and extreme outage ($20\text{x}\text{--}30\text{x}$).
- **Endpoint**: Active-Degradation Signed Bias (8 tests total). Descriptive companions: NASA score and late-prediction error.

#### Hypothesis H4: Recipe Effect vs. Original Baselines (Two-Sided, 18 Tests)
- **Comparisons**: Recipe Family (`NoPhysics`, `GRU_D_Enhanced_SWA`, `CT_PhysProb_JEPA_v2_SWA`) vs Original Baselines (`GRU_D`, `PatchTST_RUL`, `Plain_2ndOrder_NODE_Kin`) across `RMSE_All` (9 tests) and Active Signed Bias (9 tests).
- **Single-Verdict Rule**: Declare **`RECIPE EFFECT SUPPORTED`** on the **primary-endpoint `RMSE_All` only** (all 9 pairs must achieve $p_{\text{Holm}} < 0.05$, bootstrap CI excluding 0, and $|\bar{\Delta}| \ge 0.5\text{ cycles}$). Bias and descriptive companions are reported per pair without a single overarching label.

#### Hypothesis H5: Additive Latent Physics Benefit (Two-Sided, 2 Tests)
- **Objective**: Proposed model (`CT_PhysProb_JEPA_v2_SWA`) vs matched non-physics backbone (`CT_PhysProb_JEPA_NoPhysics`) on `RMSE_All` (1 test) and Active Signed Bias (1 test).
- **Verdicts**:
  - Primary `RMSE_All`: expected verdict is **`STATISTICALLY DETECTABLE BUT NOT PRACTICALLY RELEVANT`** (if $p_{\text{Holm}} < 0.05$ and CI excludes 0, but $|\bar{\Delta}| < 0.5\text{ cycles}$).
  - Active Signed Bias: expected verdict is **`PHYSICS HARMS`** (if $p_{\text{Holm}} < 0.05$, CI excludes 0, and $\bar{\Delta}_{\text{Bias}} \ge 1.0\text{ cycle}$ in favor of `NoPhysics`).

---

### 5.2 Separate Evaluation Families (Outside the 40-Test Holm Family)

#### Hypothesis H3: Integrator Symplecticity Equivalence (TOST Protocol)
- **Framework**: Two One-Sided Tests (TOST) on Strang vs Euler integrators on `RMSE_All` ($\delta = \pm 0.25$) and Active Signed Bias ($\delta = \pm 0.50$).
- **Statistical Power Restriction**: Power $>99.9\%$ claim is **restricted strictly to the Strang-vs-Euler comparison** observed in validation ($s_{\Delta} \le 0.02$). Other model comparisons have paired SD $\approx 0.1\text{--}0.4$ cycles and are tested against the 0.5-cycle practical threshold.

#### Hypothesis H4b: Recipe Component Ablation (`GRU_D_Enhanced` Backbone, 10 Tests)
- Evaluates All-On vs All-Off and 4 Leave-One-Out conditions across 30 seeds; separate Holm family ($\alpha = 0.05$).

#### Hypothesis H6: Low-Data Fleet Scaling (Omitted)
- Omitted to protect confirmatory power.

---

## 6. Limitations & Validity Boundaries

1. **Operating Conditions**: NASA C-MAPSS FD003 operates under a single nominal flight condition.
2. **Outage Realism**: Telemetric blackout intervals and channel dropouts are synthetically generated.
3. **Development Provenance**: Model architecture developed on FD001 validation units prior to freezing.
4. **Boundary Tuning Limitation**: Baselines `GRU_D_Original` (`dim=80`) and `Plain_2ndOrder_NODE_Kin_Original` (`lr=7.5e-3`) improved upon extension and remain at grid edges. They are frozen as-is to preserve experimental boundaries, with the explicit disclosure that they may remain under-tuned relative to larger capacities.
5. **Second Confirmatory Dataset**: If computational budget permits, **NASA C-MAPSS FD004** will serve as a second confirmatory stress dataset; otherwise evaluation is restricted to FD003 as registered.

---

## 7. Manifest & Cryptographic Registration

Full unit lists, split mappings, re-frozen baseline hyperparameters, and cryptographic hashes are permanently sealed in [`results/raw/CONFIRMATORY_MANIFEST.json`](file:///C:/Users/UseR/Downloads/Continous/results/raw/CONFIRMATORY_MANIFEST.json).
