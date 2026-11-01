#!/usr/bin/env python3
"""
Compute unified diff between PREREGISTRATION_V2.md (v2.2) and PREREGISTRATION_V2.md (v2.4).
"""

import difflib

# Full v2.2 text
v22_text = """# PREREGISTRATION PROTOCOL (v2 REVISION — DATED AMENDMENT v2.2)
**Document Version:** 2.2 (Dated Amendment before Confirmatory Runs)  
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
- `v2.2` (2026-11-01): Current locked version (formal two-sided hypothesis structure, H4b ablation, TOST margin justifications, equal-budget baseline tuning sweep, and full primary/secondary metric protocol).  
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

### 1.3 Anchor Reconciliation Diagnosis (UNEXPLAINED)
Comparing dense in-distribution validation anchor RMSE ($11.53$ cycles) to sparse outage onset anchor RMSE ($15.26$ cycles):
- **Stratification Shift**: Explains only $-1.1\%$ of the gap (dense validation windows are $38.2\%$ plateau / $61.8\%$ active degradation, not $>65\%$; sparse onset windows are $39.3\%$ plateau / $60.7\%$ active). Reweighting dense windows yields $11.49$ cycles.
- **Direct EWMA-Lag Test on Exact Same 280 Onset Windows**: Recomputing features for the SAME 280 onset windows under dense history vs. actual strided history yields an anchor RMSE of **$11.60$ cycles** (dense history) vs. **$15.26$ cycles** (strided history), a difference of $-3.65$ cycles that brings performance within $0.07$ cycles of the dense validation baseline. The earlier exploratory "29.6% lag" figure is retracted; mean absolute feature differences are $0.268$ (fast EWMA) and $0.160$ (slow EWMA).
- **Formal Status**: Marked **UNEXPLAINED** in official reporting, acknowledging that while feature history latency correlates with the shift, the underlying causal interaction with temporal representation remains an open question.
- **Exploratory Status of Phase 1 Stress Probes**: All preliminary stress tests evaluated during Phase 1 on 5 or 10 seeds are classified as **EXPLORATORY ONLY** and will not be reported as confirmatory scientific findings.

---

## 2. Extended Symmetry & Fair Baseline Comparison Table

Before protocol locking, an equal-budget validation hyperparameter sweep (6 trials per baseline over learning rate $\in \{1.5 \times 10^{-3}, 3.5 \times 10^{-3}, 5.0 \times 10^{-3}\}$ and model size $\in \{32, 64\}$ or $\{12, 24\}$) was conducted on FD001 validation units and logged to `results/raw/baseline_sweep_raw.csv`. The best configurations are frozen in `configs/baseline_tuned_configurations.json` and tabulated below. The proposed model is **NOT** retuned.

| # | Model Name | Temporal Encoder | Prediction Head | Latent Integrator / Dynamics | SWA (Last 3 Ep.) | Explicit Mask $m \in \{0, 1\}^{14}$ | Training Recipe | Epochs / Batch Size | Optimizer ($lr$, $wd$) | Tuned Parameters (Equal Budget) | Input Dim ($D$) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `CT_PhysProb_JEPA_v2_SWA` | GRU-D Temporal (dim 48) | KneeHazardHead (closed-form) | Hamiltonian Strang-symplectic (damped) | Yes (ep 5-7) | Yes (concatenated) | Mixed gaps (short + stride 3-15) | 8 ep / 512 | AdamW ($3.5 \times 10^{-3}$, $10^{-4}$) | Frozen (not retuned) | 56 |
| 2 | `NoPhysics` | GRU-D Temporal (dim 48) | KneeHazardHead (closed-form) | None ($\Delta y_{\text{ODE}} = 0$) | Yes (ep 5-7) | Yes (concatenated) | Mixed gaps (short + stride 3-15) | 8 ep / 512 | AdamW ($3.5 \times 10^{-3}$, $10^{-4}$) | Frozen matched control | 56 |
| 3 | `Euler_Damped` | GRU-D Temporal (dim 48) | KneeHazardHead (closed-form) | Forward Euler (damped, $\gamma > 0$) | Yes (ep 5-7) | Yes (concatenated) | Mixed gaps (short + stride 3-15) | 8 ep / 512 | AdamW ($3.5 \times 10^{-3}$, $10^{-4}$) | Frozen matched control | 56 |
| 4 | `Undamped` | GRU-D Temporal (dim 48) | KneeHazardHead (closed-form) | Hamiltonian Strang-symplectic ($\gamma = 0$) | Yes (ep 5-7) | Yes (concatenated) | Mixed gaps (short + stride 3-15) | 8 ep / 512 | AdamW ($3.5 \times 10^{-3}$, $10^{-4}$) | Frozen matched control | 56 |
| 5 | `GRU_D_Enhanced_SWA` | Canonical GRU-D Cell | KneeHazardHead (closed-form) | None (recurrent decay) | Yes (ep 5-7) | Yes (concatenated) | Mixed gaps (short + stride 3-15) | 8 ep / 512 | AdamW ($5.0 \times 10^{-3}$, $10^{-4}$) | Tuned: `dim=32`, `lr=5e-3` | 56 |
| 6 | `GRU_D_Canonical_SWA` | Canonical GRU-D Cell | Linear-Tanh-Linear Probe | None (mask-driven decay) | Yes (ep 5-7) | Yes (drives decay toward last obs) | Mixed gaps (short + stride 3-15) | 8 ep / 512 | AdamW ($3.5 \times 10^{-3}$, $10^{-4}$) | Tuned: `dim=64`, `lr=3.5e-3` | 56 |
| 7 | `Plain_NODE_Enhanced_SWA`| GRU-D Temporal (dim 48) | KneeHazardHead (closed-form) | 2nd-order Neural ODE (Euler 4-step) | Yes (ep 5-7) | Yes (concatenated) | Mixed gaps (short + stride 3-15) | 8 ep / 512 | AdamW ($3.5 \times 10^{-3}$, $10^{-4}$) | Tuned: `latent=24`, `lr=3.5e-3` | 56 |
| 8 | `PatchTST_RUL_v2` | Patched Transformer (5 tokens) | KneeHazardHead (closed-form) | None | Yes (ep 5-7) | Yes (dedicated mask token) | Mixed gaps (short + stride 3-15) | 8 ep / 512 | AdamW ($1.5 \times 10^{-3}$, $10^{-4}$) | Tuned: `d_model=64`, `lr=1.5e-3` | 56 |
| 9 | `GRU_D` (original) | Canonical GRU-D Cell | Linear-Tanh-Linear Probe | None | No* | No* (zero-fill) | Short dropouts only ($p=0.25$)* | 8 ep / 512 | AdamW ($3.5 \times 10^{-3}$, $10^{-4}$) | Original benchmark | 42 |
| 10 | `PatchTST_RUL` (original) | Patched Transformer (3 tokens) | Linear-Tanh-Linear Probe | None | No* | No* | Short dropouts only ($p=0.25$)* | 8 ep / 512 | AdamW ($3.5 \times 10^{-3}$, $10^{-4}$) | Original benchmark | 42 |
| 11 | `Plain_2ndOrder_NODE_Kin` | Linear-Tanh-Linear MLP | Linear-Tanh-Linear Probe | Kinematic Euler ($M=1$) | No* | No* | Short dropouts only ($p=0.25$)* | 8 ep / 512 | AdamW ($3.5 \times 10^{-3}$, $10^{-4}$) | Original benchmark | 42 |

*\*Intentional Asymmetries: Original-recipe benchmarks 9–11 lack mixed-gap augmentation, KneeHazardHead, binary mask vectors, and SWA by historical definition to quantify the cumulative benefit of the proposed recipe (Hypothesis H4).*

---

## 3. Evaluation Protocols, Data Partitions, & Endpoints

### 3.1 Primary & Secondary Endpoints
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

### 3.2 Evaluated Outage & Stress Regimes
1. `In-Distribution`: Nominal continuous observation ($dt=1$, random dropout $p=0.25$).
2. `OOD 10x-20x`: Primary blackout horizon (stride $10\text{--}20$ cycles).
3. `OOD 15x-25x`: Extended blackout horizon (stride $15\text{--}25$ cycles).
4. `OOD 20x-30x`: Extreme blackout horizon (stride $20\text{--}30$ cycles).
5. `Sensor Dropout 10%`: Random channel masking of $10\%$ of sensors.
6. `Sensor Dropout 20%`: Random channel masking of $20\%$ of sensors.
7. `Sensor Dropout 30%`: Extreme channel masking of $30\%$ of sensors.

### 3.3 Confirmatory Data Partition & Separation Protocol
- **Dataset**: NASA C-MAPSS FD003 (two fault modes: HPC degradation + Fan degradation).
- **Training & Calibration Pool (100 engines in `train_FD003.txt`)**:
  - **Model Training (80 engines)**: Partitioned into 5 folds (16 engines per fold) for 5-fold unit-level cross-validation.
  - **Conformal Calibration & Checkpoint Selection (20 engines)**: Engines `[1, 4, 7, 15, 20, 33, 35, 45, 48, 52, 59, 60, 61, 74, 84, 85, 87, 93, 94, 98]` held out from training to compute non-conformity scores ($q_{\hat{\alpha}}$) and select final model checkpoints.
- **Official Test Set (100 engines in `test_FD003.txt` + `RUL_FD003.txt`)**:
  - Held out strictly until the final frozen evaluation.
  - Evaluated once per seed at the frozen checkpoint. Zero test-set checkpoint selection.
- **Zero-Overlap Guarantee**: Zero engines from FD001 used during exploratory or development stages are present in FD003.

---

## 4. Confirmatory Hypotheses (All Two-Sided Tests)

All hypotheses will be evaluated across **30 fixed random seeds**:
```python
SEEDS_30 = [
    42, 100, 101, 102, 103, 104, 105, 106, 107, 108, 
    109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 
    119, 120, 121, 122, 123, 124, 125, 126, 127, 128
]
```

### Hypothesis H1: Sensor Channel Dropout Resilience (Two-Sided)
- **Null Hypothesis ($H_{1,0}$):** Across sensor channel dropout fractions $\{10\%, 20\%, 30\%\}$, there is no difference in Total RMSE between models:
  $$\Delta_{\text{RMSE}} = \text{RMSE}_A - \text{RMSE}_B = 0$$
- **Alternative Hypothesis ($H_{1,A}$):** $\Delta_{\text{RMSE}} \ne 0$ at two-sided family-wise error rate $\alpha = 0.05$ after joint Holm-Bonferroni correction.
- **Reported Comparisons (12 tests total)**:
  1. `CT_PhysProb_JEPA_v2_SWA` vs `GRU_D_Enhanced_SWA` (3 dropouts).
  2. `CT_PhysProb_JEPA_v2_SWA` vs `NoPhysics` (3 dropouts).
  3. `NoPhysics` vs `GRU_D_Canonical_SWA` (3 dropouts).
  4. `GRU_D_Enhanced_SWA` vs `GRU_D_Canonical_SWA` (3 dropouts).

### Hypothesis H2: Extreme Outage Optimistic Bias Suppression (Two-Sided)
- **Null Hypothesis ($H_{2,0}$):** Under extreme outage horizons $\{15\text{x}\text{--}25\text{x}, 20\text{x}\text{--}30\text{x}\}$, there is no difference in active-degradation signed bias between models:
  $$\Delta_{\text{Bias}} = \text{Bias}_A - \text{Bias}_B = 0$$
- **Alternative Hypothesis ($H_{2,A}$):** $\Delta_{\text{Bias}} \ne 0$ at two-sided family-wise error rate $\alpha = 0.05$ after joint Holm-Bonferroni correction.
- **Reported Comparisons (8 tests total)**:
  1. `CT_PhysProb_JEPA_v2_SWA` vs `GRU_D_Enhanced_SWA` (2 horizons).
  2. `CT_PhysProb_JEPA_v2_SWA` vs `NoPhysics` (2 horizons).
  3. `NoPhysics` vs `GRU_D_Canonical_SWA` (2 horizons).
  4. `GRU_D_Enhanced_SWA` vs `GRU_D_Canonical_SWA` (2 horizons).

### Hypothesis H3: Integrator Symplecticity Equivalence (TOST Protocol)
- **Framework**: Two One-Sided Tests (TOST) assessing whether Strang-symplectic leapfrog integration is practically equivalent to standard forward Euler integration under identical architecture and SWA.
- **Equivalence Bounds**:
  - $\delta_{\text{RMSE}} = 0.25$ cycles ($\sim 1.3\%$ of the nominal RUL scale, well within seed standard error of $\sim 0.8\text{--}0.9$ cycles).
  - $\delta_{\text{Bias}} = 0.50$ cycles ($<5\%$ of typical active degradation bias).
- **Decision Rule**: Equivalence is confirmed if the 90% confidence interval for the paired difference lies entirely within $[-\delta, +\delta]$.
- **Scope Note**: *"Equivalence reflects the marginal ODE contribution in this parameterization, not a general statement about symplectic integration."*

### Hypothesis H4: Recipe Effect vs. Original-Recipe Baselines (Two-Sided)
- **Objective**: Quantify the cumulative performance shift attributable strictly to the training recipe (mixed-gap augmentation + masked GRU-D encoder + KneeHazardHead + SWA).
- **Comparisons (18 tests total)**:
  - Recipe Family (`NoPhysics`, `GRU_D_Enhanced_SWA`, `CT_PhysProb_JEPA_v2_SWA`) vs. Original-Recipe Baselines (`GRU_D`, `PatchTST_RUL`, `Plain_2ndOrder_NODE_Kin`):
    - 9 pairwise comparisons on Primary Endpoint Total RMSE.
    - 9 pairwise comparisons on Active-Degradation Signed Bias.
- **Verdict Rule**:
  - Declare **`RECIPE EFFECT SUPPORTED`** only if, on the primary endpoint and on active-degradation bias, every recipe-family model beats every original-recipe baseline with $p_{\text{Holm}} < 0.05$.
  - Otherwise: report the per-pair outcomes without a single overarching label.

### Hypothesis H4b: Recipe Component Ablation (`GRU_D_Enhanced` Backbone)
- **Protocol**: Evaluated across all 30 seeds on identical splits using `GRU_D_Enhanced` backbone:
  1. **All-On (Full Recipe)**: Mask + Mixed-Gap Augmentation + KneeHazardHead + SWA.
  2. **All-Off (Baseline Recipe)**: No mask (zero-fill) + Short-gap only ($p=0.25$) + Linear Probe Head + No SWA.
  3. **Leave-One-Out Ablations**:
     - Without Explicit Mask ($m_t = 0$)
     - Without Mixed-Gap Augmentation (trained on short gaps only)
     - Without KneeHazardHead (replaced by linear probe)
     - Without SWA (last-epoch weights only)
- **Constraint**: No claims about individual components beyond this ablation.

### Hypothesis H5: Additive Latent Physics Benefit (Two-Sided)
- **Objective**: Directly test whether incorporating the 2nd-order Hamiltonian latent ODE dynamics yields a statistically significant benefit over the matched non-physics recipe (`CT_PhysProb_JEPA_v2_SWA` vs. `CT_PhysProb_JEPA_NoPhysics`).
- **Endpoints**: Primary Endpoint Total RMSE (1 test) and Active-Degradation Signed Bias (1 test). Two-sided.
- **Verdicts**:
  - Declare **`PHYSICS BENEFIT SUPPORTED`**: if $p_{\text{Holm}} < 0.05$ with $\Delta_{\text{RMSE}} < 0$ and $\Delta_{\text{Bias}} \le 0$.
  - Declare **`PHYSICS BENEFIT NOT SUPPORTED`**: if $p_{\text{Holm}} \ge 0.05$.
  - Declare **`PHYSICS HARMS`**: if $p_{\text{Holm}} < 0.05$ with $\Delta_{\text{RMSE}} > 0$ (the non-physics baseline significantly outperforms the proposed model).

### Hypothesis H6: Low-Data Fleet Scaling (Omitted)
- **Status**: **OMITTED** from confirmatory testing to prevent excessive multiplicity penalties on primary reliability hypotheses H1–H5.

---

## 5. Multiplicity Control & Statistical Decision Framework

1. **Joint Family-Wise Error Rate Control**:
   - Holm-Bonferroni step-down correction is applied across the **Primary Confirmatory Family** containing **$M = 40$ tests** jointly:
     - H1 (Sensor Dropout): 12 tests
     - H2 (Extreme Outage Bias): 8 tests
     - H4 (Recipe Effect on RMSE & Bias): 18 tests
     - H5 (Physics Add-On): 2 tests
     - Total: $M = 40$ tests at family-wise $\alpha = 0.05$.
2. **Reported Statistical Metrics per Comparison**:
   - Paired sample mean difference ($\Delta = \mu_A - \mu_B$).
   - Parametric 95% Student's $t$ confidence interval.
   - Two-sided paired Student's $t$-test $p$-value ($p_t$).
   - Non-parametric two-sided Wilcoxon signed-rank test $p$-value ($p_w$).
   - Joint Holm-Bonferroni adjusted $p$-value ($p_{\text{Holm}}$).

---

## 6. Limitations & Validity Boundaries

1. **Operating Conditions**: NASA C-MAPSS FD003 operates under a single nominal flight condition (sea level, condition 1).
2. **Outage Realism**: Blackout intervals and channel dropouts are synthetically generated to model telemetric communication loss.
3. **Development Provenance**: Model architecture and recipe were developed on FD001 validation units prior to freezing.
4. **Second Confirmatory Dataset**: If computational budget permits following completion of FD003 confirmatory runs, **NASA C-MAPSS FD004** (multi-condition with 6 operational regimes and dual failure modes) will serve as a second confirmatory stress dataset; otherwise evaluation is restricted to FD003 as registered.

---

## 7. Manifest & Cryptographic Provenance

Full unit lists, split mappings, baseline tuned hyperparameters, and cryptographic hashes are permanently sealed in [`results/raw/CONFIRMATORY_MANIFEST.json`](file:///C:/Users/UseR/Downloads/Continous/results/raw/CONFIRMATORY_MANIFEST.json).
"""

with open('PREREGISTRATION_V2.md', 'r', encoding='utf-8') as f:
    v24_text = f.read()

diff = list(difflib.unified_diff(
    v22_text.splitlines(keepends=True),
    v24_text.splitlines(keepends=True),
    fromfile='PREREGISTRATION_V2.md (v2.2)',
    tofile='PREREGISTRATION_V2.md (v2.4)',
    n=2
))

print(''.join(diff))
