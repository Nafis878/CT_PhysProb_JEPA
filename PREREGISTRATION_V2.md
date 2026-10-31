# PREREGISTRATION PROTOCOL (v2 REVISION — DATED AMENDMENT v2.3)
**Document Version:** 2.3 (Dated Amendment before Confirmatory Runs)  
**Date of Execution:** 2026-11-01  
**Target Journal:** *Reliability Engineering & System Safety* (RESS)  
**Investigator:** Research Engineer and Statistician  
**Frozen Model Configuration Path:** `configs/frozen_final_configuration_v2.json`  
**Configuration SHA-256 Hash:** `5f8b33b5253bb903011ba391c4e348af4c17cf72dbf31c971a2c07117cad656d`  
**Baseline Tuned Configuration Path:** `configs/baseline_tuned_configurations.json`  
**Manifest Path:** `results/raw/CONFIRMATORY_MANIFEST.json`  
**Document History:**  
- `v2.0` (2026-11-01): Initial Step 2 post-development freeze protocol.  
- `v2.1` (2026-11-01): SHA-256 `827728fcd941cd19bee0a81cf40e92b27d84b2f8784c20ea3500fb747e3db138` (Phase 1 diagnostic corrections and recipe framing).  
- `v2.2` (2026-11-01): SHA-256 `a5a49d1cb902564d869ae7f51194f39afc4fc0460669c985199d5e4eabaf47c8` (Two-sided hypothesis structure, TOST justifications, equal-budget baseline sweep).  
- `v2.3` (2026-11-01): Current locked version (engine-level paired cluster bootstrap, practical significance thresholds, extended 5-seed baseline sweep, separate H3/H4b family structure, and external registration commit provenance).  
**Protocol Status:** LOCKED & FROZEN prior to unblinded confirmatory runs.  

---

## 1. Context, Protocol Provenance, & Contribution Framing

This pre-registration document formally amends the confirmatory evaluation protocol following Step 2 model development and Phase 1 diagnostics.

### 1.1 Contribution Naming & Framing
The primary object of study and core contribution of this work is the **RECIPE** for blackout-robust continuous-time remaining useful life prediction:
$$\text{Recipe} = \{\text{Masked GRU-D Temporal Encoder} + \text{Closed-Form Knee-Hazard Countdown Head} + \text{Mixed-Gap Training Augmentation} + \text{Stochastic Weight Averaging (SWA)}\}$$
The **Recipe Family** evaluated comprises:
$$\mathcal{M}_{\text{Recipe}} = \{\text{CT\_PhysProb\_JEPA\_NoPhysics}, \text{GRU\_D\_Enhanced\_SWA}, \text{CT\_PhysProb\_JEPA\_v2\_SWA}\}$$
The Hamiltonian continuous-time latent is evaluated strictly as an additive continuous-time physics hypothesis on top of this recipe, **NOT** as the primary driver of performance.  
*Development Budget Clarification:* The proposed model received three development rounds on FD001 validation units prior to freezing, while baseline models received the equal-budget hyperparameter sweep.

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
- **Mechanism Separation Test on Exact Same 280 Onset Windows**:
  - Recomputing features for the SAME 280 onset windows under dense history ($dt=1$ up to $t_k$) vs. actual strided history yields an anchor RMSE of **$11.60$ cycles** (dense history) vs. **$15.26$ cycles** (strided history), a difference of **$-3.65$ cycles** that brings performance to within $0.07$ cycles of the dense validation baseline of $11.53$ cycles.
  - Because contemporaneous raw sensor observations at onset cycle $t_k$ are identical ($0.0$ difference between strided and dense samples), replacing EWMA features with oracle dense EWMA while keeping strided raw inputs isolates that the $+3.65$-cycle anchor inflation is $100\%$ mediated through the EWMA memory channels (`mem_fast` and `mem_slow`).
  - Dense validation windows are $38.2\%$ plateau / $61.8\%$ active degradation (not $>65\%$). Stratification explains only $-1.1\%$ of the gap. Mean absolute feature differences are $0.268$ (fast EWMA) and $0.160$ (slow EWMA); the earlier exploratory "29.6% lag" figure is retracted.
- **Exploratory Status of Phase 1 Stress Probes**: All preliminary stress tests evaluated during Phase 1 on 5 or 10 seeds are classified as **EXPLORATORY ONLY** and will not be reported as confirmatory scientific findings.

---

## 2. Extended Symmetry & Re-Frozen Baseline Configurations

Before protocol locking, an extended equal-budget validation hyperparameter sweep (9 trials per model over 3 learning rates $\times$ 3 sizes, evaluated across 5 validation seeds: `[42, 100, 101, 102, 103]`) was conducted on FD001 validation units and logged to `results/raw/extended_baseline_sweep_raw.csv`. The resulting best configurations are frozen in `configs/baseline_tuned_configurations.json` and tabulated below alongside the frozen proposed model.

| # | Model Name | Temporal Encoder | Prediction Head | Latent Integrator / Dynamics | SWA (Last 3 Ep.) | Explicit Mask $m \in \{0, 1\}^{14}$ | Training Recipe | Epochs / Batch Size | Optimizer ($lr$, $wd$) | Tuned Parameters (Equal Budget) | Val RMSE (5 Seeds) | Input Dim ($D$) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `CT_PhysProb_JEPA_v2_SWA` | GRU-D Temporal (dim 48) | KneeHazardHead (closed-form) | Hamiltonian Strang-symplectic (damped) | Yes (ep 5-7) | Yes (concatenated) | Mixed gaps (short + stride 3-15) | 8 ep / 512 | AdamW ($3.5 \times 10^{-3}$, $10^{-4}$) | Frozen (3 dev rounds) | $18.208 \pm 0.375$ | 56 |
| 2 | `NoPhysics` | GRU-D Temporal (dim 48) | KneeHazardHead (closed-form) | None ($\Delta y_{\text{ODE}} = 0$) | Yes (ep 5-7) | Yes (concatenated) | Mixed gaps (short + stride 3-15) | 8 ep / 512 | AdamW ($3.5 \times 10^{-3}$, $10^{-4}$) | Frozen matched control | — | 56 |
| 3 | `Euler_Damped` (Descriptive) | GRU-D Temporal (dim 48) | KneeHazardHead (closed-form) | Forward Euler (damped, $\gamma > 0$) | Yes (ep 5-7) | Yes (concatenated) | Mixed gaps (short + stride 3-15) | 8 ep / 512 | AdamW ($3.5 \times 10^{-3}$, $10^{-4}$) | Frozen matched control | — | 56 |
| 4 | `Undamped` (Descriptive) | GRU-D Temporal (dim 48) | KneeHazardHead (closed-form) | Hamiltonian Strang-symplectic ($\gamma = 0$) | Yes (ep 5-7) | Yes (concatenated) | Mixed gaps (short + stride 3-15) | 8 ep / 512 | AdamW ($3.5 \times 10^{-3}$, $10^{-4}$) | Frozen matched control | — | 56 |
| 5 | `GRU_D_Enhanced_SWA` | Canonical GRU-D Cell | KneeHazardHead (closed-form) | None (recurrent decay) | Yes (ep 5-7) | Yes (concatenated) | Mixed gaps (short + stride 3-15) | 8 ep / 512 | AdamW ($5.0 \times 10^{-3}$, $10^{-4}$) | Tuned: `dim=48`, `lr=5e-3` | $18.143 \pm 0.265$ | 56 |
| 6 | `GRU_D_Canonical_SWA` | Canonical GRU-D Cell | Linear-Tanh-Linear Probe | None (mask-driven decay) | Yes (ep 5-7) | Yes (drives decay toward last obs) | Mixed gaps (short + stride 3-15) | 8 ep / 512 | AdamW ($3.5 \times 10^{-3}$, $10^{-4}$) | Tuned: `dim=80`, `lr=3.5e-3` | $17.909 \pm 0.160$ | 56 |
| 7 | `Plain_NODE_Enhanced_SWA` (Descriptive) | GRU-D Temporal (dim 48) | KneeHazardHead (closed-form) | 2nd-order Neural ODE (Euler 4-step) | Yes (ep 5-7) | Yes (concatenated) | Mixed gaps (short + stride 3-15) | 8 ep / 512 | AdamW ($3.5 \times 10^{-3}$, $10^{-4}$) | Tuned: `latent=24`, `lr=3.5e-3` | $18.105 \pm 0.293$ | 56 |
| 8 | `PatchTST_RUL_v2` (Descriptive) | Patched Transformer (5 tokens) | KneeHazardHead (closed-form) | None | Yes (ep 5-7) | Yes (dedicated mask token) | Mixed gaps (short + stride 3-15) | 8 ep / 512 | AdamW ($1.0 \times 10^{-3}$, $10^{-4}$) | Tuned: `d_model=64`, `lr=1e-3` | $18.455 \pm 0.164$ | 56 |
| 9 | `GRU_D` (original) | Canonical GRU-D Cell | Linear-Tanh-Linear Probe | None | No* | No* (zero-fill) | Short dropouts only ($p=0.25$)* | 8 ep / 512 | AdamW ($3.5 \times 10^{-3}$, $10^{-4}$) | Tuned: `dim=64`, `lr=3.5e-3` | $19.264 \pm 0.421$ | 42 |
| 10 | `PatchTST_RUL` (original) | Patched Transformer (3 tokens) | Linear-Tanh-Linear Probe | None | No* | No* | Short dropouts only ($p=0.25$)* | 8 ep / 512 | AdamW ($3.5 \times 10^{-3}$, $10^{-4}$) | Tuned: `d_model=48`, `lr=3.5e-3` | $21.397 \pm 0.397$ | 42 |
| 11 | `Plain_2ndOrder_NODE_Kin` (original) | Linear-Tanh-Linear MLP | Linear-Tanh-Linear Probe | Kinematic Euler ($M=1$) | No* | No* | Short dropouts only ($p=0.25$)* | 8 ep / 512 | AdamW ($5.0 \times 10^{-3}$, $10^{-4}$) | Tuned: `latent=24`, `lr=5e-3` | $24.458 \pm 1.313$ | 42 |

### 2.1 Descriptive Model Classification & No "Best Model" Claim
- Models `PatchTST_RUL_v2`, `Plain_NODE_Enhanced_SWA`, `Euler_Damped`, `Undamped`, and `EstimatedAnchor_Countdown` are classified as **DESCRIPTIVE**. Their results will be reported for scientific completeness but are not subject to formal confirmatory hypothesis tests and are excluded from the Primary Holm family.
- **Explicit Scientific Boundary**:
  > *"There is no pre-registered BEST / COMPETITIVE superiority verdict versus all models, and the paper will make no 'best model' claim."*

---

## 3. Inference Units, Dual Significance Criteria, & Practical Thresholds

### 3.1 Inference Units
For every primary comparison, statistical evaluation operates on two distinct complementary units:
1. **Seed-Level Inference**: Paired Student's $t$-test and Wilcoxon signed-rank test across the $n = 30$ fixed random seeds.
2. **Engine-Level Paired Cluster Bootstrap**:
   - Resamples the 100 official test engines with replacement ($B = 10,000$ resamples).
   - For each resample, predictions are averaged across the 30 seeds per engine before computing the paired metric difference.
   - Computes non-parametric percentile 95% confidence intervals ($[q_{0.025}, q_{0.975}]$).

### 3.2 Pre-Registered Practical Significance Thresholds
To prevent trivial numerical differences from masquerading as engineering advances, the following fixed minimum effect thresholds are locked:
- **RMSE Threshold**: $|\Delta_{\text{RMSE}}| \ge 0.50\text{ cycles}$.
- **Signed Bias Threshold**: $|\Delta_{\text{Bias}}| \ge 1.00\text{ cycle}$.

### 3.3 Dual Significance Decision Rule
A formal **`SUPPORTED`** or **`HARMS`** verdict requires satisfying **ALL THREE** criteria simultaneously:
1. Two-sided joint Holm-Bonferroni adjusted $p$-value $p_{\text{Holm}} < 0.05$ across seeds, **AND**
2. Engine-level paired cluster bootstrap 95% CI strictly excludes 0, **AND**
3. Paired sample mean difference $|\Delta| \ge \text{practical threshold}$ ($0.5$ cycles for RMSE, $1.0$ cycle for bias).

*Special Labeling:* Any comparison where $p_{\text{Holm}} < 0.05$ and the bootstrap CI excludes 0, but $|\Delta| < \text{threshold}$, is formally labeled:
**`STATISTICALLY DETECTABLE BUT NOT PRACTICALLY RELEVANT`**.

---

## 4. Evaluation Protocols, Data Partitions, & Endpoints

### 4.1 Primary & Secondary Endpoints
- **Primary Endpoint**: Total RMSE across all blackout windows (`RMSE_All`) under Out-of-Distribution Long Outages ($10\text{x}\text{--}20\text{x}$ stride) evaluated on the 100 official NASA C-MAPSS FD003 test engines.
- **Secondary Endpoints**:
  1. `RMSE_Last`: Prediction RMSE strictly at the final observed operational cycle before failure.
  2. `Active-Degradation Signed Bias`: Mean error $(\hat{y} - y^*)$ over windows with active degradation ($y^* < 125$ cycles).
  3. `NASA Asymmetric Scoring Function`: Asymmetric penalty $S = \sum (\exp(d_i/10) - 1)$ for overestimates ($d_i > 0$) and $(\exp(-d_i/13) - 1)$ for underestimates ($d_i < 0$).
  4. `Anchor RMSE`: Prediction error at blackout onset $t_k$.
  5. `Gap-Induced RMSE`: Total RMSE minus Anchor RMSE.
  6. `Conformal Coverage & Interval Width`: Empirical coverage and mean prediction interval width at nominal $90\%$ coverage ($1 - \alpha = 0.90$).
  7. `Missed Critical Failures`: Percentage of engines where actual RUL is $\le 10$ cycles but predicted lower bound exceeds 15 cycles.
  8. `Expected Condition-Based Maintenance (CBM) Cost`: Cost per unit under cost ratio $C_{\text{unplanned}} / C_{\text{planned}} = 5.0$, with intervention threshold fixed on validation units.

### 4.2 Evaluated Outage & Stress Regimes
1. `In-Distribution`: Nominal continuous observation ($dt=1$, random dropout $p=0.25$).
2. `OOD 10x-20x`: Primary blackout horizon (stride $10\text{--}20$ cycles).
3. `OOD 15x-25x`: Extended blackout horizon (stride $15\text{--}25$ cycles).
4. `OOD 20x-30x`: Extreme blackout horizon (stride $20\text{--}30$ cycles).
5. `Sensor Dropout 10%`: Random channel masking of $10\%$ of sensors.
6. `Sensor Dropout 20%`: Random channel masking of $20\%$ of sensors.
7. `Sensor Dropout 30%`: Extreme channel masking of $30\%$ of sensors.

### 4.3 Confirmatory Data Partition & Separation Protocol
- **Dataset**: NASA C-MAPSS FD003 (two fault modes: HPC degradation + Fan degradation).
- **Training Pool Mapping (100 engines in `train_FD003.txt`)**:
  - **Official-Test Training (All 80 engines)**: Models evaluated on the official test set are trained on all 80 engines in `split_info['train']`.
  - **5-Fold Unit-Level Cross-Validation**: Evaluates the 80 engines (16 per fold) strictly as a separate replication outside the Holm family.
  - **Conformal Calibration Pool (20 engines)**: Engines `[1, 4, 7, 15, 20, 33, 35, 45, 48, 52, 59, 60, 61, 74, 84, 85, 87, 93, 94, 98]` are NEVER used in training; they are used exclusively for computing non-conformity scores ($q_{\hat{\alpha}}$) and validation monitoring.
- **Checkpoint Rule**: The evaluated checkpoint is **ALWAYS** the fixed SWA average of epochs 5–7 (strictly zero checkpoint selection).
- **Official Test Evaluation (100 engines in `test_FD003.txt` + `RUL_FD003.txt`)**:
  - Held out strictly until the final frozen evaluation.
  - Evaluated exactly once per seed at the frozen SWA checkpoint. Zero test-set tuning.
- **Zero-Overlap Guarantee**: Zero engines from FD001 used during exploratory or development stages are present in FD003.

---

## 5. Confirmatory Hypotheses & Separate Test Families

All hypotheses will be evaluated across **30 fixed random seeds**:
```python
SEEDS_30 = [
    42, 100, 101, 102, 103, 104, 105, 106, 107, 108, 
    109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 
    119, 120, 121, 122, 123, 124, 125, 126, 127, 128
]
```

### 5.1 Primary Hypothesis Family ($M = 40$ Tests, Joint Holm-Bonferroni at $\alpha = 0.05$)

#### Hypothesis H1: Sensor Channel Dropout Resilience (Two-Sided, 12 Tests)
- **Endpoint**: Primary Endpoint `RMSE_All` under channel dropout fractions $\{10\%, 20\%, 30\%\}$.
- **Comparisons (12 tests total)**:
  1. `CT_PhysProb_JEPA_v2_SWA` vs `GRU_D_Enhanced_SWA` (3 dropouts).
  2. `CT_PhysProb_JEPA_v2_SWA` vs `NoPhysics` (3 dropouts).
  3. `NoPhysics` vs `GRU_D_Canonical_SWA` (3 dropouts).
  4. `GRU_D_Enhanced_SWA` vs `GRU_D_Canonical_SWA` (3 dropouts).

#### Hypothesis H2: Extreme Outage Optimistic Bias Suppression (Two-Sided, 8 Tests)
- **Endpoint**: Active-degradation signed bias under extreme blackout horizons $\{15\text{x}\text{--}25\text{x}, 20\text{x}\text{--}30\text{x}\}$.
- **Comparisons (8 tests total)**:
  1. `CT_PhysProb_JEPA_v2_SWA` vs `GRU_D_Enhanced_SWA` (2 horizons).
  2. `CT_PhysProb_JEPA_v2_SWA` vs `NoPhysics` (2 horizons).
  3. `NoPhysics` vs `GRU_D_Canonical_SWA` (2 horizons).
  4. `GRU_D_Enhanced_SWA` vs `GRU_D_Canonical_SWA` (2 horizons).

#### Hypothesis H4: Recipe Effect vs. Original-Recipe Baselines (Two-Sided, 18 Tests)
- **Objective**: Quantify cumulative performance shift attributable strictly to the training recipe.
- **Comparisons (18 tests total)**:
  - Recipe Family (`NoPhysics`, `GRU_D_Enhanced_SWA`, `CT_PhysProb_JEPA_v2_SWA`) vs. Original Baselines (`GRU_D`, `PatchTST_RUL`, `Plain_2ndOrder_NODE_Kin`):
    - 9 pairwise comparisons on Primary Endpoint `RMSE_All` at OOD 10x-20x.
    - 9 pairwise comparisons on Active-Degradation Signed Bias at OOD 10x-20x.
- **Verdict Rule**:
  - Declare **`RECIPE EFFECT SUPPORTED`** only if every recipe-family model beats every original baseline with $p_{\text{Holm}} < 0.05$, bootstrap CI excluding 0, and $|\Delta| \ge \text{threshold}$.
  - Otherwise: report per-pair outcomes without a single overarching label.

#### Hypothesis H5: Additive Latent Physics Benefit (Two-Sided, 2 Tests)
- **Objective**: Test whether incorporating 2nd-order Hamiltonian ODE dynamics yields a significant benefit over the matched non-physics recipe (`CT_PhysProb_JEPA_v2_SWA` vs. `CT_PhysProb_JEPA_NoPhysics`).
- **Endpoints**: Primary Endpoint `RMSE_All` at OOD 10x-20x (1 test) and Active-Degradation Signed Bias at OOD 10x-20x (1 test).
- **Verdicts per Endpoint**:
  - Declare **`PHYSICS BENEFIT SUPPORTED`**: if $p_{\text{Holm}} < 0.05$, engine bootstrap CI excludes 0, and $|\Delta| \ge \text{threshold}$ in favor of proposed model.
  - Declare **`PHYSICS HARMS`**: if $p_{\text{Holm}} < 0.05$, engine bootstrap CI excludes 0, and $|\Delta| \ge \text{threshold}$ in favor of non-physics baseline.
  - Declare **`STATISTICALLY DETECTABLE BUT NOT PRACTICALLY RELEVANT`**: if $p_{\text{Holm}} < 0.05$ and CI excludes 0, but $|\Delta| < \text{threshold}$.
  - Declare **`PHYSICS BENEFIT NOT SUPPORTED`**: if $p_{\text{Holm}} \ge 0.05$ or bootstrap CI covers 0.
- Combined summary across both endpoints is reported only as descriptive text.

---

### 5.2 Separate Evaluation Families (Outside the 40-Test Holm Family)

#### Hypothesis H3: Integrator Symplecticity Equivalence (TOST Protocol)
- **Framework**: Two One-Sided Tests (TOST) assessing whether Strang-symplectic leapfrog integration is practically equivalent to standard forward Euler integration under identical architecture and SWA.
- **Family Structure**: Separate family ($\alpha = 0.05$) with Holm adjustment across its 2 endpoints (`RMSE_All` and active signed bias).
- **Equivalence Margins & Justifications**:
  - $\delta_{\text{RMSE}} = 0.25\text{ cycles}$: Corresponds to $\sim 1.3\%$ of the nominal $\sim 19$-cycle RMSE scale and is calibrated against the standard deviation of paired differences ($s_{\Delta} \le 0.02$ cycles in validation).
  - $\delta_{\text{Bias}} = 0.50\text{ cycles}$: Corresponds to $\sim 5.5\%$ of the $\sim 9$-cycle active degradation bias.
- **Statistical Power Note**: For $n = 30$ seeds with observed paired difference standard deviation $s_{\Delta} \le 0.02$ cycles, the statistical power to confirm equivalence within $\delta = 0.25$ exceeds $99.9\%$ ($\beta < 0.001$).
- **Scope Note**:
  > *"Equivalence reflects the marginal ODE contribution in this parameterization, not a general statement about symplectic integration."*

#### Hypothesis H4b: Recipe Component Ablation (`GRU_D_Enhanced` Backbone)
- **Protocol**: Evaluated across all 30 seeds on identical splits using `GRU_D_Enhanced` backbone:
  1. **All-On (Full Recipe)**: Mask + Mixed-Gap Augmentation + KneeHazardHead + SWA.
  2. **All-Off (Baseline Recipe)**: No mask (zero-fill) + Short-gap only ($p=0.25$) + Linear Probe Head + No SWA.
  3. **Leave-One-Out Ablations**:
     - Without Explicit Mask ($m_t = 0$)
     - Without Mixed-Gap Augmentation (trained on short gaps only)
     - Without KneeHazardHead (replaced by linear probe)
     - Without SWA (last-epoch weights only)
- **Constraint**: Separate family ($\alpha = 0.05$). No claims about individual components beyond this ablation.

#### Hypothesis H6: Low-Data Fleet Scaling (Omitted)
- **Status**: **OMITTED** from confirmatory testing to protect statistical power on primary endpoints.

---

## 6. Limitations & Validity Boundaries

1. **Operating Conditions**: NASA C-MAPSS FD003 operates under a single nominal flight condition (sea level, condition 1).
2. **Outage Realism**: Blackout intervals and channel dropouts are synthetically generated to model telemetric communication loss.
3. **Development Provenance**: Model architecture and recipe were developed on FD001 validation units prior to freezing.
4. **Second Confirmatory Dataset**: If computational budget permits following completion of FD003 confirmatory runs, **NASA C-MAPSS FD004** (multi-condition with 6 operational regimes and dual failure modes) will serve as a second confirmatory stress dataset; otherwise evaluation is restricted to FD003 as registered.

---

## 7. Manifest & Cryptographic Registration

Full unit lists, split mappings, re-frozen baseline hyperparameters, and cryptographic hashes are permanently sealed in [`results/raw/CONFIRMATORY_MANIFEST.json`](file:///C:/Users/UseR/Downloads/Continous/results/raw/CONFIRMATORY_MANIFEST.json).
