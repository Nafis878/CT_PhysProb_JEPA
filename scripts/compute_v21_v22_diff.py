#!/usr/bin/env python3
import difflib
from pathlib import Path

v21_text = """# PREREGISTRATION PROTOCOL (v2 REVISION — DATED AMENDMENT)
**Document Version:** 2.1 (Dated Amendment before Confirmatory Runs)  
**Date of Execution:** 2026-11-01  
**Target Journal:** *Reliability Engineering & System Safety* (RESS)  
**Investigator:** Research Engineer and Statistician  
**Frozen Model Configuration Path:** `configs/frozen_final_configuration_v2.json`  
**Configuration SHA-256 Hash:** `5f8b33b5253bb903011ba391c4e348af4c17cf72dbf31c971a2c07117cad656d`  
**Manifest Path:** `results/raw/CONFIRMATORY_MANIFEST.json`  
**Protocol Status:** LOCKED & FROZEN prior to unblinded confirmatory runs.  

---

## 1. Context, Protocol Provenance, & Contribution Framing

This pre-registration document amends the primary pre-registration (`PREREGISTRATION.md`) following the completion and freezing of Step 2 model development (`configs/frozen_final_configuration_v2.json`) and Phase 1 diagnostics.

### 1.1 Contribution Naming & Framing
The primary object of study and core contribution of this work is the **RECIPE** for blackout-robust continuous-time remaining useful life prediction:
$$\\text{Recipe} = \\{\\text{Masked GRU-D Temporal Encoder} + \\text{Closed-Form Knee-Hazard Countdown Head} + \\text{Mixed-Gap Training Augmentation} + \\text{Stochastic Weight Averaging (SWA)}\\}$$
The **Recipe Family** evaluated comprises:
$$\\mathcal{M}_{\\text{Recipe}} = \\{\\text{CT\\_PhysProb\\_JEPA\\_NoPhysics}, \\text{GRU\\_D\\_Enhanced\\_SWA}, \\text{CT\\_PhysProb\\_JEPA\\_v2\\_SWA}\\}$$
The Hamiltonian continuous-time latent is evaluated strictly as an additive continuous-time physics hypothesis on top of this recipe, **NOT** as the primary driver of performance.

### 1.2 Formal ODE Limitation Statement
Following exhaustive diagnostic analysis across optimization dynamics, relative parameter update trajectories, and gradient cosine alignment:
> *"The Hamiltonian latent as parameterized receives no consistent learning signal and contributes no measurable benefit."*

Parameter norms of the potential $W_{\\text{pot}}$ remain pinned within $1.5\\%$ of initialization ($2.22 \\to 2.18$), gradient directional consistency across consecutive minibatches is low ($+0.2704$, with decay parameter at only $+0.1300$), and relative epoch update steps shrink to $0.013$ under AdamW weight decay ($10^{-4}$). The closed-form countdown hazard head accounts for $>96\\%$ of total prediction variance during long blackout outages.

### 1.3 Exploratory Status of Phase 1 Stress Probes
All preliminary stress tests (sensor dropout, extreme outages) conducted during Phase 1 on 5 or 10 seeds are classified as **EXPLORATORY ONLY**. They were evaluated post hoc without joint multiplicity control and shall not be reported as confirmatory scientific findings. Confirmatory claims rest solely on the protocol defined herein.

---

## 2. Extended Symmetry & Fair Baseline Comparison Table

To ensure strict scientific integrity and avoid baseline handicapping, every model configuration evaluated in the confirmatory stage is tabulated below:

| # | Model Name | Temporal Encoder | Prediction Head | Latent Integrator / Dynamics | SWA (Last 3 Ep.) | Explicit Mask $m \\in \\{0, 1\\}^{14}$ | Training Recipe | Epochs / Batch Size | Optimizer ($lr$, $wd$) | Tuning Budget | Input Dim ($D$) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `CT_PhysProb_JEPA_v2_SWA` | GRU-D Temporal (dim 48) | KneeHazardHead (closed-form) | Hamiltonian Strang-symplectic (damped) | Yes (ep 5-7) | Yes (concatenated) | Mixed gaps (short + stride 3-15) | 8 ep / 512 | AdamW ($3.5 \\times 10^{-3}$, $10^{-4}$) | 8 epochs (fixed) | 56 |
| 2 | `NoPhysics` | GRU-D Temporal (dim 48) | KneeHazardHead (closed-form) | None ($\\Delta y_{\\text{ODE}} = 0$) | Yes (ep 5-7) | Yes (concatenated) | Mixed gaps (short + stride 3-15) | 8 ep / 512 | AdamW ($3.5 \\times 10^{-3}$, $10^{-4}$) | 8 epochs (fixed) | 56 |
| 3 | `Euler_Damped` | GRU-D Temporal (dim 48) | KneeHazardHead (closed-form) | Forward Euler (damped, $\\gamma > 0$) | Yes (ep 5-7) | Yes (concatenated) | Mixed gaps (short + stride 3-15) | 8 ep / 512 | AdamW ($3.5 \\times 10^{-3}$, $10^{-4}$) | 8 epochs (fixed) | 56 |
| 4 | `Undamped` | GRU-D Temporal (dim 48) | KneeHazardHead (closed-form) | Hamiltonian Strang-symplectic ($\\gamma = 0$) | Yes (ep 5-7) | Yes (concatenated) | Mixed gaps (short + stride 3-15) | 8 ep / 512 | AdamW ($3.5 \\times 10^{-3}$, $10^{-4}$) | 8 epochs (fixed) | 56 |
| 5 | `GRU_D_Enhanced_SWA` | Canonical GRU-D Cell | KneeHazardHead (closed-form) | None (recurrent decay) | Yes (ep 5-7) | Yes (concatenated) | Mixed gaps (short + stride 3-15) | 8 ep / 512 | AdamW ($3.5 \\times 10^{-3}$, $10^{-4}$) | 8 epochs (fixed) | 56 |
| 6 | `GRU_D_Canonical_SWA` | Canonical GRU-D Cell | Linear-Tanh-Linear Probe | None (mask-driven decay) | Yes (ep 5-7) | Yes (drives decay toward last obs) | Mixed gaps (short + stride 3-15) | 8 ep / 512 | AdamW ($3.5 \\times 10^{-3}$, $10^{-4}$) | 8 epochs (fixed) | 56 |
| 7 | `Plain_NODE_Enhanced_SWA`| GRU-D Temporal (dim 48) | KneeHazardHead (closed-form) | 2nd-order Neural ODE (Euler 4-step) | Yes (ep 5-7) | Yes (concatenated) | Mixed gaps (short + stride 3-15) | 8 ep / 512 | AdamW ($3.5 \\times 10^{-3}$, $10^{-4}$) | 8 epochs (fixed) | 56 |
| 8 | `PatchTST_RUL_v2` | Patched Transformer (4 tokens) | KneeHazardHead (closed-form) | None | Yes (ep 5-7) | Yes (dedicated mask token) | Mixed gaps (short + stride 3-15) | 8 ep / 512 | AdamW ($3.5 \\times 10^{-3}$, $10^{-4}$) | 8 epochs (fixed) | 56 |
| 9 | `GRU_D` (original) | Canonical GRU-D Cell | Linear-Tanh-Linear Probe | None | No* | No* (zero-fill) | Short dropouts only ($p=0.25$)* | 8 ep / 512 | AdamW ($3.5 \\times 10^{-3}$, $10^{-4}$) | 8 epochs (fixed) | 42 |
| 10 | `PatchTST_RUL` (original) | Patched Transformer (3 tokens) | Linear-Tanh-Linear Probe | None | No* | No* | Short dropouts only ($p=0.25$)* | 8 ep / 512 | AdamW ($3.5 \\times 10^{-3}$, $10^{-4}$) | 8 epochs (fixed) | 42 |
| 11 | `Plain_2ndOrder_NODE_Kin` | Linear-Tanh-Linear MLP | Linear-Tanh-Linear Probe | Kinematic Euler ($M=1$) | No* | No* | Short dropouts only ($p=0.25$)* | 8 ep / 512 | AdamW ($3.5 \\times 10^{-3}$, $10^{-4}$) | 8 epochs (fixed) | 42 |

*\\*Note on Intentional Asymmetries:* Models 9–11 represent original-recipe published benchmarks. Their lack of mixed-gap data augmentation, KneeHazardHead, binary mask vectors, and SWA is intentional and serves specifically to quantify the cumulative benefit of the proposed recipe (Hypothesis H4). Models 5–8 represent enhanced competitors receiving 100% equal treatment.

---

## 3. Fixed Pre-Registered Hypotheses (Confirmatory Protocol)

All hypotheses will be evaluated across **30 fixed random seeds**:
```python
SEEDS_30 = [
    42, 100, 101, 102, 103, 104, 105, 106, 107, 108, 
    109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 
    119, 120, 121, 122, 123, 124, 125, 126, 127, 128
]
```

### Hypothesis H1: Sensor Channel Dropout Resilience
- **Null Hypothesis ($H_{1,0}$):** Across fixed sensor channel dropout fractions $\\{10\\%, 20\\%, 30\\%\\}$, the paired difference in Total RMSE between `CT_PhysProb_JEPA_v2_SWA` and fair competitors (`GRU_D_Canonical_SWA`, `GRU_D_Enhanced_SWA`) is greater than or equal to zero:
  $$\\Delta_{\\text{RMSE}} = \\text{RMSE}_{\\text{Proposed}} - \\text{RMSE}_{\\text{Competitor}} \\ge 0$$
- **Alternative Hypothesis ($H_{1,A}$):** $\\Delta_{\\text{RMSE}} < 0$ at family-wise error rate $\\alpha = 0.05$ after joint Holm-Bonferroni correction.

### Hypothesis H2: Extreme Outage Optimistic Bias Suppression
- **Null Hypothesis ($H_{2,0}$):** Under extreme outage horizons $\\{15\\text{x}\\text{--}25\\text{x}, 20\\text{x}\\text{--}30\\text{x}\\}$, the paired difference in active-degradation signed bias between `CT_PhysProb_JEPA_v2_SWA` and competitors (`GRU_D_Canonical_SWA`, `GRU_D_Enhanced_SWA`) is greater than or equal to zero:
  $$\\Delta_{\\text{Bias}} = \\text{Bias}_{\\text{Proposed}} - \\text{Bias}_{\\text{Competitor}} \\ge 0$$
- **Alternative Hypothesis ($H_{2,A}$):** $\\Delta_{\\text{Bias}} < 0$ (lower optimistic failure bias) at family-wise error rate $\\alpha = 0.05$ after joint Holm-Bonferroni correction.

### Hypothesis H3: Integrator Symplecticity Equivalence (TOST Protocol)
- **Equivalence Framework:** Two One-Sided Tests (TOST) assessing whether Strang-symplectic leapfrog integration is practically equivalent to standard non-symplectic forward Euler integration under identical conditions.
- **Equivalence Bounds:**
  - $\\delta_{\\text{RMSE}} = 0.25$ cycles.
  - $\\delta_{\\text{Bias}} = 0.50$ cycles.
- **Hypotheses:**
  - $H_{3,0}^{(1)}: \\Delta \\le -\\delta \\quad \\text{or} \\quad H_{3,0}^{(2)}: \\Delta \\ge +\\delta$
  - $H_{3,A}: -\\delta < \\Delta < +\\delta$ (Equivalence confirmed if the 90% confidence interval for the paired difference lies entirely within $[-\\delta, +\\delta]$).

### Hypothesis H4: Recipe Effect vs. Original-Recipe Baselines
- **Objective:** Quantify the performance shift attributable strictly to the training recipe (mixed-gap augmentation + masked GRU-D encoder + KneeHazardHead + SWA).
- **Evaluation:** Compare the Recipe Family (`NoPhysics`, `GRU_D_Enhanced_SWA`, `CT_PhysProb_JEPA_v2_SWA`) against original-recipe baselines (`GRU_D`, `PatchTST_RUL`, `Plain_2ndOrder_NODE_Kin`) across all outage regimes on Total RMSE, active-degradation signed bias, and NASA asymmetric scoring.
- **Verdict Rule:**
  - If $p_{\\text{Holm}} < 0.05$ across regimes in favor of the recipe family: declare **`RECIPE EFFECT SUPPORTED`**.
  - Otherwise: declare **`RECIPE EFFECT NOT SUPPORTED`**.

### Hypothesis H5: Additive Latent Physics Benefit (Ablation Endpoint)
- **Objective:** Directly test whether incorporating the 2nd-order Hamiltonian latent ODE dynamics yields a statistically significant benefit over the matched non-physics recipe (`CT_PhysProb_JEPA_v2_SWA` vs. `CT_PhysProb_JEPA_NoPhysics`).
- **Endpoint:** Total RMSE and active-degradation bias across all stress and nominal regimes.
- **Verdict Rule:**
  - If $p_{\\text{Holm}} < 0.05$ and $\\Delta_{\\text{RMSE}} < 0$: declare **`PHYSICS BENEFIT SUPPORTED`**.
  - If $p_{\\text{Holm}} \\ge 0.05$ or $\\Delta_{\\text{RMSE}} \\ge 0$: declare **`PHYSICS BENEFIT NOT SUPPORTED`**.

### Hypothesis H6: Low-Data Engine Scaling (Omitted from Confirmatory Protocol)
- **Status:** **OMITTED**. To maintain maximal statistical power for the primary reliability endpoints (H1–H5) and prevent severe multiplicity penalties under family-wise Holm correction, the low-data fleet variation is formally omitted from confirmatory testing.

---

## 4. Multiplicity Control & Statistical Decision Framework

1. **Joint Family-Wise Error Rate (FWER) Control:**  
   Holm-Bonferroni step-down correction is applied across ALL confirmatory tests across Hypotheses H1, H2, H4, and H5 jointly, maintaining overall family-wise $\\alpha = 0.05$.
2. **Reported Statistical Metrics:**  
   Every pairwise comparison reports:
   - Paired sample mean difference ($\\Delta$).
   - Parametric 95% Student's $t$ confidence interval.
   - Two-sided paired Student's $t$-test $p$-value ($p_t$).
   - Non-parametric two-sided Wilcoxon signed-rank test $p$-value ($p_w$).
   - Joint Holm-Bonferroni adjusted $p$-value ($p_{\\text{Holm}}$).
3. **Equivalence Tests (H3):**  
   Evaluated using TOST 90% confidence intervals against pre-registered margins ($\\pm 0.25$ cycles RMSE, $\\pm 0.50$ cycles bias).

---

## 5. Confirmatory Data Partition & Zero-Overlap Guarantee

- **Confirmatory Dataset:** Official **NASA C-MAPSS FD003** (dual failure mode: High-Pressure Compressor degradation + Fan degradation).
- **Evaluated Cohorts:**
  1. **5-Fold Unit-Level Cross-Validation:** Conducted across the FD003 training pool.
  2. **Official Held-Out Test Evaluation:** Evaluated on all 100 official FD003 test engines with ground truth failure cycles from `RUL_FD003.txt`.
- **Zero-Overlap Guarantee:**  
  All 100 training engines and 100 test engines of FD001 used during exploratory phases, Step 1 decomposition, and Step 2 validation are strictly quarantined. Zero FD001 data will be used in any confirmatory metric.
- **Manifest Provenance:**  
  Full engine lists and cryptographic hashes are permanently sealed in `results/raw/CONFIRMATORY_MANIFEST.json`.
"""

with open('PREREGISTRATION_V2.md', 'r', encoding='utf-8') as f:
    v22_text = f.read()

diff = list(difflib.unified_diff(
    v21_text.splitlines(keepends=True),
    v22_text.splitlines(keepends=True),
    fromfile='PREREGISTRATION_V2.md (v2.1)',
    tofile='PREREGISTRATION_V2.md (v2.2)',
    n=2
))

print(''.join(diff))
