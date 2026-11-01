# CONFIRMATORY FINDINGS REPORT (PREREGISTRATION v2.5 OFFICIAL EVALUATION — CORRECTED AUDIT)
**Target Dataset:** NASA C-MAPSS FD003 Official Test Set (100 engines)
**Evaluation Scope:** 30 Pre-Registered Seeds (`[42, 100..128]`), 11 Models + Heuristics, 7 Regimes
**Protocol Version:** v2.5 (Locked & Remote-Verified at `https://github.com/Nafis878/CT_PhysProb_JEPA.git` tag `preregistration-v2.5`)
**Execution Audit Date:** 2026-11-01 | **Status:** Strictly Reconciled with Pre-Registration Protocol

---
## Executive Summary of Confirmatory Findings

1. **Additive Latent Physics (H5):** **`NOT SUPPORTED`**.
   - On the official FD003 test set, adding the 2nd-order Hamiltonian latent ODE yields **no statistically significant benefit** on primary `RMSE_All` (paired diff $-0.037$ cycles, $t = -0.21, p = 0.835, p_{\text{Holm}} = 1.0$) and no benefit on Active Degradation Signed Bias ($+0.265$ cycles, $p = 0.577, p_{\text{Holm}} = 1.0$).
   - The empirical per-seed mean absolute ODE correction magnitude on the test set is only $|\Delta y_{\text{ODE}}| = 2.54 \pm 0.90$ cycles, while the closed-form countdown hazard head carries $>96\%$ of total prediction variance.
2. **Integrator Symplecticity Equivalence (H3):** **`EQUIVALENCE SUPPORTED`** (under pre-registered scope note).
   - Under the Two One-Sided Tests (TOST) framework, Strang-symplectic splitting is statistically equivalent to forward Euler within $\pm 0.25$ cycles on `RMSE_All` ($p_{\text{TOST}} = 1.20 \times 10^{-46}$) and within $\pm 0.50$ cycles on Signed Bias ($p_{\text{TOST}} = 1.19 \times 10^{-47}$).
   - Per-seed Strang vs. Euler difference is only $0.021 \pm 0.013$ cycles. *Scope note:* Equivalence reflects that the ODE contribution is negligible in this parameterization, not a general theoretical statement about symplectic integration in other physical domains.
3. **Recipe Effect vs. Original Baselines (H4):** **`RECIPE EFFECT NOT FULLY SUPPORTED`** under the single-verdict primary `RMSE_All` rule.
   - Recipe models (`CT_PhysProb_JEPA_v2_SWA`, `NoPhysics`, `GRU_D_Enhanced_SWA`) achieve large, statistically significant gains over `PatchTST_Original` ($-2.82$ to $-2.89$ cycles RMSE, $p_{\text{Holm}} < 0.028$) and `Plain_2ndOrder_NODE_Kin_Original` ($-4.26$ to $-4.33$ cycles RMSE, $p_{\text{Holm}} < 10^{-9}$), and slash optimistic bias vs. `GRU_D_Original` by $\approx 4.0$ to $4.4$ cycles ($p_{\text{Holm}} < 10^{-4}$).
   - However, re-frozen `GRU_D_Original` (tuned at `hidden_dim=80`) matches the recipe models on point RMSE ($16.63 \pm 0.67$ vs $16.81 \pm 0.83$, paired diff $+0.17$, $p = 0.40$). Because the single-verdict rule requires all 9 pairs to meet dual significance and practical threshold, the overarching recipe verdict is not fully supported.
4. **Sensor Channel Dropout Resilience (H1):** **`NOT SUPPORTED`** (all 12 pre-registered tests).
   - When evaluated against `GRU_D_Canonical_SWA` and matched controls receiving masked inputs, the proposed model provides no statistically significant advantage ($p_{\text{Holm}} = 1.0$).
5. **Extreme Outage Optimistic Bias Suppression (H2):** **`SUPPORTED`** vs. `GRU_D_Canonical_SWA` on extreme $20\text{x}\text{--}30\text{x}$ outages (3 Supported, 5 Not Supported).
   - All three recipe backbones equipped with the KneeHazardHead significantly reduce active degradation optimistic bias by $>2.0$ cycles vs. `GRU_D_Canonical_SWA` ($p_{\text{Holm}} < 0.03$), but tie with each other.
6. **Non-Deployable Kinematic Upper Bound:**
   - `TimeCorrected_LOCF_Oracle` achieves **$12.26$ cycles RMSE** and NASA score $1,683.4$, outperforming every continuous-time deep neural model by over $4.0$ cycles RMSE.

---
## 1. Master Benchmark Results on Primary Confirmatory Regime (`OOD_10x_20x`)

| Model Architecture | Temporal Encoder | Prediction Head | Latent Dynamics | SWA | RMSE_All (Mean ± SD) | RMSE_Last (Mean ± SD) | Active Bias (Mean ± SD) | NASA Score (Mean ± SD) | Paired Diff NASA vs Proposed | 90% Coverage (%) | 90% Width (cyc) | Missed Crit (%) | Expected CBM Cost ($) |
|:---|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `CT_PhysProb_JEPA_v2_SWA` | GRU-D (d=48) | KneeHazardHead | Hamiltonian Strang (Damped) | Yes | 16.81 ± 0.83 | 19.65 ± 0.47 | +2.53 ± 2.11 | 6809.6 ± 491.4 | +0.0 ± 0.0 | 89.5% | 41.88 | 0.00% | $0.0066 |
| `CT_PhysProb_JEPA_NoPhysics` | GRU-D (d=48) | KneeHazardHead | None (No Physics) | Yes | 16.84 ± 0.79 | 19.60 ± 0.41 | +2.27 ± 1.86 | 6751.4 ± 386.0 | +58.2 ± 603.1 | 89.5% | 42.00 | 0.00% | $0.0075 |
| `Euler_Damped` | GRU-D (d=48) | KneeHazardHead | Forward Euler ODE | Yes | 16.81 ± 0.83 | 19.65 ± 0.48 | +2.52 ± 2.11 | 6809.2 ± 491.9 | +0.4 ± 3.2 | 89.5% | 41.89 | 0.00% | $0.0066 |
| `Undamped` | GRU-D (d=48) | KneeHazardHead | Strang Symplectic (gamma=0) | Yes | 16.83 ± 0.85 | 19.65 ± 0.49 | +2.50 ± 2.17 | 6832.4 ± 519.2 | -22.8 ± 74.4 | 89.5% | 41.91 | 0.00% | $0.0066 |
| `GRU_D_Enhanced_SWA` | GRU-D (d=48) | KneeHazardHead | Recurrent Decay | Yes | 16.87 ± 1.25 | 19.83 ± 0.61 | +2.11 ± 2.97 | 6815.7 ± 699.0 | -6.1 ± 873.5 | 89.9% | 42.82 | 0.00% | $0.0070 |
| `GRU_D_Canonical_SWA` | GRU-D (d=80) | Linear Probe | Recurrent Decay | Yes | 16.58 ± 0.80 | 19.55 ± 0.39 | +2.86 ± 1.69 | 6594.0 ± 403.1 | +215.6 ± 625.5 | 89.9% | 42.05 | 0.00% | $0.0055 |
| `Plain_NODE_Enhanced_SWA` | GRU-D Temporal | KneeHazardHead | Neural ODE | Yes | 16.74 ± 0.84 | 19.62 ± 0.48 | +2.57 ± 1.82 | 6851.3 ± 478.3 | -41.7 ± 560.8 | 89.5% | 41.45 | 0.00% | $0.0059 |
| `PatchTST_RUL_v2` | PatchTST (d=64) | KneeHazardHead | None | Yes | 16.23 ± 0.59 | 19.65 ± 0.54 | +4.60 ± 1.82 | 7249.3 ± 559.9 | -439.8 ± 833.7 | 91.1% | 42.06 | 0.00% | $0.0063 |
| `GRU_D_Original` | GRU-D (d=80) | Linear Probe | Recurrent Decay | No | 16.63 ± 0.67 | 20.17 ± 0.49 | +6.48 ± 1.73 | 7952.5 ± 756.0 | -1142.9 ± 849.1 | 91.5% | 43.61 | 0.00% | $0.0044 |
| `PatchTST_Original` | PatchTST (d=48) | Linear Probe | None | No | 19.69 ± 4.02 | 20.77 ± 0.86 | +1.56 ± 7.73 | 11316.6 ± 4987.1 | -4507.0 ± 4981.6 | 88.5% | 47.94 | 0.00% | $0.0065 |
| `Plain_2ndOrder_NODE_Kin_Original` | MLP (d=24) | Linear Probe | Kinematic Euler | No | 21.13 ± 1.81 | 21.61 ± 0.84 | +2.42 ± 3.04 | 15589.2 ± 5475.0 | -8779.6 ± 5683.1 | 89.6% | 52.25 | 0.56% | $0.0020 |
| `CT_PhysProb_JEPA_v2_SWA_Countdown` | Estimated Anchor | Countdown LOCF | None | No | 22.09 ± 1.06 | 19.22 ± 0.30 | -3.06 ± 2.07 | 8167.3 ± 897.8 | -1357.8 ± 762.3 | N/A | N/A | N/A | $0.0000 |
| `TimeCorrected_LOCF_Oracle` | Oracle Ground Truth | Kinematic Countdown | None | No | 12.26 ± 0.00 | 3.63 ± 0.00 | -1.37 ± 0.00 | 1683.4 ± 0.0 | +5126.1 ± 491.4 | N/A | N/A | N/A | $0.0000 |

*Note:* `PatchTST_RUL_v2` is equipped with the `KneeHazardHead` (not a linear probe). Conformal coverage and width are labeled N/A for deterministic point reference heuristics (`TimeCorrected_LOCF_Oracle` and `Countdown`).

---
## 2. Complete Pre-Registered Primary Hypothesis Family ($M = 40$ Tests, Joint Holm-Bonferroni $\alpha = 0.05$)

Decision Rules:
- **Dual Significance Criterion:** Both seed-level $p_{\text{Holm}} < 0.05$ AND engine cluster bootstrap 95% CI strictly excluding 0.
- **Practical Significance Threshold:** $|\bar{\Delta}| \ge 0.50\text{ cycles}$ for `RMSE_All`; $|\bar{\Delta}| \ge 1.00\text{ cycle}$ for Signed Bias.
- **Dual Estimand Reporting:**
  - **Estimand 1 (Pre-Registered):** Bootstrap resamples engines; evaluates metric on the seed-averaged predictions $\bar{\hat{y}}_u$.
  - **Estimand 2 (Sensitivity):** Bootstrap resamples engines; evaluates metric per seed and averages differences across seeds $\frac{1}{30}\sum_s \Delta_s$.

### Hypothesis H1 (12 Pre-Registered Tests)

| Comparison Name | Regime | Endpoint | Model A Mean | Model B Mean | Paired Diff (Mean ± SD) | Seed-Level 95% CI | $t$-stat | $p_{\text{raw}}$ | $p_{\text{Holm}}$ | Engine Bootstrap 95% CI (Estimand 1: PreReg) | Engine Bootstrap 95% CI (Estimand 2: Sensitivity) | Confirmatory Verdict |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `CT_PhysProb_JEPA_v2_SWA_vs_CT_PhysProb_JEPA_NoPhysics_Dropout_10pct` | `Dropout_10pct` | `RMSE_All` | 16.55 | 16.61 | -0.055 ± 0.630 | [-0.290, +0.180] | -0.48 | 6.3434e-01 | 1.0000e+00 | [-0.171, -0.032] | [-0.126, +0.014] | **`NOT SUPPORTED`** |
| `CT_PhysProb_JEPA_v2_SWA_vs_GRU_D_Canonical_SWA_Dropout_10pct` | `Dropout_10pct` | `RMSE_All` | 16.55 | 16.38 | +0.176 ± 1.096 | [-0.233, +0.585] | 0.88 | 3.8571e-01 | 1.0000e+00 | [-0.080, +0.122] | [+0.075, +0.284] | **`NOT SUPPORTED`** |
| `CT_PhysProb_JEPA_NoPhysics_vs_GRU_D_Canonical_SWA_Dropout_10pct` | `Dropout_10pct` | `RMSE_All` | 16.61 | 16.38 | +0.231 ± 1.029 | [-0.153, +0.616] | 1.23 | 2.2806e-01 | 1.0000e+00 | [+0.009, +0.236] | [+0.112, +0.353] | **`NOT SUPPORTED`** |
| `GRU_D_Enhanced_SWA_vs_GRU_D_Canonical_SWA_Dropout_10pct` | `Dropout_10pct` | `RMSE_All` | 16.45 | 16.38 | +0.075 ± 1.164 | [-0.359, +0.510] | 0.36 | 7.2505e-01 | 1.0000e+00 | [-0.292, -0.030] | [-0.060, +0.209] | **`NOT SUPPORTED`** |
| `CT_PhysProb_JEPA_v2_SWA_vs_CT_PhysProb_JEPA_NoPhysics_Dropout_20pct` | `Dropout_20pct` | `RMSE_All` | 18.54 | 18.52 | +0.013 ± 0.898 | [-0.322, +0.348] | 0.08 | 9.3848e-01 | 1.0000e+00 | [-0.210, -0.038] | [-0.076, +0.097] | **`NOT SUPPORTED`** |
| `CT_PhysProb_JEPA_v2_SWA_vs_GRU_D_Canonical_SWA_Dropout_20pct` | `Dropout_20pct` | `RMSE_All` | 18.54 | 18.13 | +0.403 ± 1.584 | [-0.188, +0.994] | 1.39 | 1.7388e-01 | 1.0000e+00 | [-0.143, +0.306] | [+0.178, +0.625] | **`NOT SUPPORTED`** |
| `CT_PhysProb_JEPA_NoPhysics_vs_GRU_D_Canonical_SWA_Dropout_20pct` | `Dropout_20pct` | `RMSE_All` | 18.52 | 18.13 | +0.390 ± 1.489 | [-0.166, +0.946] | 1.44 | 1.6170e-01 | 1.0000e+00 | [-0.070, +0.474] | [+0.116, +0.657] | **`NOT SUPPORTED`** |
| `GRU_D_Enhanced_SWA_vs_GRU_D_Canonical_SWA_Dropout_20pct` | `Dropout_20pct` | `RMSE_All` | 18.78 | 18.13 | +0.646 ± 1.995 | [-0.099, +1.391] | 1.77 | 8.6591e-02 | 1.0000e+00 | [-0.067, +0.256] | [+0.486, +0.802] | **`NOT SUPPORTED`** |
| `CT_PhysProb_JEPA_v2_SWA_vs_CT_PhysProb_JEPA_NoPhysics_Dropout_30pct` | `Dropout_30pct` | `RMSE_All` | 19.31 | 19.26 | +0.052 ± 1.012 | [-0.326, +0.430] | 0.28 | 7.8155e-01 | 1.0000e+00 | [-0.144, +0.021] | [-0.029, +0.126] | **`NOT SUPPORTED`** |
| `CT_PhysProb_JEPA_v2_SWA_vs_GRU_D_Canonical_SWA_Dropout_30pct` | `Dropout_30pct` | `RMSE_All` | 19.31 | 18.94 | +0.371 ± 1.907 | [-0.341, +1.083] | 1.07 | 2.9554e-01 | 1.0000e+00 | [-0.341, +0.265] | [+0.078, +0.658] | **`NOT SUPPORTED`** |
| `CT_PhysProb_JEPA_NoPhysics_vs_GRU_D_Canonical_SWA_Dropout_30pct` | `Dropout_30pct` | `RMSE_All` | 19.26 | 18.94 | +0.319 ± 1.899 | [-0.390, +1.028] | 0.92 | 3.6473e-01 | 1.0000e+00 | [-0.304, +0.352] | [+0.006, +0.628] | **`NOT SUPPORTED`** |
| `GRU_D_Enhanced_SWA_vs_GRU_D_Canonical_SWA_Dropout_30pct` | `Dropout_30pct` | `RMSE_All` | 19.56 | 18.94 | +0.618 ± 2.240 | [-0.218, +1.454] | 1.51 | 1.4154e-01 | 1.0000e+00 | [-0.147, +0.167] | [+0.459, +0.772] | **`NOT SUPPORTED`** |

### Hypothesis H2 (8 Pre-Registered Tests)

| Comparison Name | Regime | Endpoint | Model A Mean | Model B Mean | Paired Diff (Mean ± SD) | Seed-Level 95% CI | $t$-stat | $p_{\text{raw}}$ | $p_{\text{Holm}}$ | Engine Bootstrap 95% CI (Estimand 1: PreReg) | Engine Bootstrap 95% CI (Estimand 2: Sensitivity) | Confirmatory Verdict |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `CT_PhysProb_JEPA_v2_SWA_vs_CT_PhysProb_JEPA_NoPhysics_OOD_15x_25x_Bias` | `OOD_15x_25x` | `Signed_Bias_ActiveDeg` | 2.94 | 2.71 | +0.232 ± 2.607 | [-0.742, +1.205] | 0.49 | 6.3027e-01 | 1.0000e+00 | [+0.111, +0.347] | [+0.111, +0.347] | **`NOT SUPPORTED`** |
| `CT_PhysProb_JEPA_v2_SWA_vs_GRU_D_Canonical_SWA_OOD_15x_25x_Bias` | `OOD_15x_25x` | `Signed_Bias_ActiveDeg` | 2.94 | 4.05 | -1.111 ± 2.077 | [-1.886, -0.335] | -2.93 | 6.5480e-03 | 1.8335e-01 | [-1.376, -0.853] | [-1.376, -0.853] | **`NOT SUPPORTED`** |
| `CT_PhysProb_JEPA_NoPhysics_vs_GRU_D_Canonical_SWA_OOD_15x_25x_Bias` | `OOD_15x_25x` | `Signed_Bias_ActiveDeg` | 2.71 | 4.05 | -1.342 ± 3.012 | [-2.467, -0.218] | -2.44 | 2.0984e-02 | 5.6657e-01 | [-1.618, -1.062] | [-1.618, -1.062] | **`NOT SUPPORTED`** |
| `GRU_D_Enhanced_SWA_vs_GRU_D_Canonical_SWA_OOD_15x_25x_Bias` | `OOD_15x_25x` | `Signed_Bias_ActiveDeg` | 2.60 | 4.05 | -1.445 ± 3.369 | [-2.703, -0.187] | -2.35 | 2.5850e-02 | 6.7209e-01 | [-1.727, -1.147] | [-1.727, -1.147] | **`NOT SUPPORTED`** |
| `CT_PhysProb_JEPA_v2_SWA_vs_CT_PhysProb_JEPA_NoPhysics_OOD_20x_30x_Bias` | `OOD_20x_30x` | `Signed_Bias_ActiveDeg` | 2.89 | 2.81 | +0.087 ± 2.636 | [-0.897, +1.072] | 0.18 | 8.5706e-01 | 1.0000e+00 | [-0.023, +0.195] | [-0.023, +0.195] | **`NOT SUPPORTED`** |
| `CT_PhysProb_JEPA_v2_SWA_vs_GRU_D_Canonical_SWA_OOD_20x_30x_Bias` | `OOD_20x_30x` | `Signed_Bias_ActiveDeg` | 2.89 | 4.93 | -2.036 ± 2.161 | [-2.843, -1.228] | -5.16 | 1.6328e-05 | 5.5515e-04 | [-2.321, -1.741] | [-2.321, -1.741] | **`SUPPORTED`** |
| `CT_PhysProb_JEPA_NoPhysics_vs_GRU_D_Canonical_SWA_OOD_20x_30x_Bias` | `OOD_20x_30x` | `Signed_Bias_ActiveDeg` | 2.81 | 4.93 | -2.123 ± 3.157 | [-3.302, -0.944] | -3.68 | 9.3897e-04 | 2.7405e-02 | [-2.400, -1.837] | [-2.400, -1.837] | **`SUPPORTED`** |
| `GRU_D_Enhanced_SWA_vs_GRU_D_Canonical_SWA_OOD_20x_30x_Bias` | `OOD_20x_30x` | `Signed_Bias_ActiveDeg` | 2.64 | 4.93 | -2.286 ± 3.249 | [-3.499, -1.072] | -3.85 | 5.9518e-04 | 1.9641e-02 | [-2.608, -1.930] | [-2.608, -1.930] | **`SUPPORTED`** |

### Hypothesis H4 (18 Pre-Registered Tests)

| Comparison Name | Regime | Endpoint | Model A Mean | Model B Mean | Paired Diff (Mean ± SD) | Seed-Level 95% CI | $t$-stat | $p_{\text{raw}}$ | $p_{\text{Holm}}$ | Engine Bootstrap 95% CI (Estimand 1: PreReg) | Engine Bootstrap 95% CI (Estimand 2: Sensitivity) | Confirmatory Verdict |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `CT_PhysProb_JEPA_NoPhysics_vs_GRU_D_Original_RMSE` | `OOD_10x_20x` | `RMSE_All` | 16.84 | 16.63 | +0.209 ± 0.956 | [-0.148, +0.566] | 1.20 | 2.4093e-01 | 1.0000e+00 | [+0.203, +1.141] | [-0.233, +0.664] | **`NOT SUPPORTED`** |
| `CT_PhysProb_JEPA_NoPhysics_vs_PatchTST_Original_RMSE` | `OOD_10x_20x` | `RMSE_All` | 16.84 | 19.69 | -2.850 ± 4.226 | [-4.428, -1.272] | -3.69 | 9.1350e-04 | 2.7405e-02 | [-2.272, -0.931] | [-3.499, -2.235] | **`SUPPORTED`** |
| `CT_PhysProb_JEPA_NoPhysics_vs_Plain_2ndOrder_NODE_Kin_Original_RMSE` | `OOD_10x_20x` | `RMSE_All` | 16.84 | 21.13 | -4.289 ± 2.084 | [-5.068, -3.511] | -11.27 | 4.0667e-12 | 1.5860e-10 | [-4.339, -2.664] | [-5.105, -3.472] | **`SUPPORTED`** |
| `GRU_D_Enhanced_SWA_vs_GRU_D_Original_RMSE` | `OOD_10x_20x` | `RMSE_All` | 16.87 | 16.63 | +0.237 ± 1.269 | [-0.237, +0.711] | 1.02 | 3.1437e-01 | 1.0000e+00 | [+0.108, +1.057] | [-0.201, +0.689] | **`NOT SUPPORTED`** |
| `GRU_D_Enhanced_SWA_vs_PatchTST_Original_RMSE` | `OOD_10x_20x` | `RMSE_All` | 16.87 | 19.69 | -2.821 ± 4.048 | [-4.333, -1.310] | -3.82 | 6.5409e-04 | 2.0277e-02 | [-2.355, -1.037] | [-3.469, -2.224] | **`SUPPORTED`** |
| `GRU_D_Enhanced_SWA_vs_Plain_2ndOrder_NODE_Kin_Original_RMSE` | `OOD_10x_20x` | `RMSE_All` | 16.87 | 21.13 | -4.261 ± 2.247 | [-5.100, -3.422] | -10.39 | 2.7648e-11 | 1.0506e-09 | [-4.434, -2.735] | [-5.095, -3.424] | **`SUPPORTED`** |
| `CT_PhysProb_JEPA_v2_SWA_vs_GRU_D_Original_RMSE` | `OOD_10x_20x` | `RMSE_All` | 16.81 | 16.63 | +0.171 ± 1.095 | [-0.237, +0.580] | 0.86 | 3.9789e-01 | 1.0000e+00 | [+0.155, +1.070] | [-0.257, +0.615] | **`NOT SUPPORTED`** |
| `CT_PhysProb_JEPA_v2_SWA_vs_PatchTST_Original_RMSE` | `OOD_10x_20x` | `RMSE_All` | 16.81 | 19.69 | -2.887 ± 4.123 | [-4.427, -1.348] | -3.84 | 6.2422e-04 | 1.9975e-02 | [-2.322, -1.000] | [-3.516, -2.287] | **`SUPPORTED`** |
| `CT_PhysProb_JEPA_v2_SWA_vs_Plain_2ndOrder_NODE_Kin_Original_RMSE` | `OOD_10x_20x` | `RMSE_All` | 16.81 | 21.13 | -4.327 ± 1.990 | [-5.070, -3.584] | -11.91 | 1.0857e-12 | 4.3428e-11 | [-4.417, -2.699] | [-5.163, -3.486] | **`SUPPORTED`** |
| `CT_PhysProb_JEPA_NoPhysics_vs_GRU_D_Original_Bias` | `OOD_10x_20x` | `Signed_Bias_ActiveDeg` | 2.27 | 6.48 | -4.214 ± 2.225 | [-5.045, -3.383] | -10.37 | 2.8696e-11 | 1.0617e-09 | [-4.806, -3.598] | [-4.806, -3.598] | **`SUPPORTED`** |
| `CT_PhysProb_JEPA_NoPhysics_vs_PatchTST_Original_Bias` | `OOD_10x_20x` | `Signed_Bias_ActiveDeg` | 2.27 | 1.56 | +0.706 ± 8.260 | [-2.378, +3.790] | 0.47 | 6.4309e-01 | 1.0000e+00 | [-0.263, +1.753] | [-0.263, +1.753] | **`NOT SUPPORTED`** |
| `CT_PhysProb_JEPA_NoPhysics_vs_Plain_2ndOrder_NODE_Kin_Original_Bias` | `OOD_10x_20x` | `Signed_Bias_ActiveDeg` | 2.27 | 2.42 | -0.152 ± 3.518 | [-1.466, +1.161] | -0.24 | 8.1429e-01 | 1.0000e+00 | [-1.267, +0.925] | [-1.267, +0.925] | **`NOT SUPPORTED`** |
| `GRU_D_Enhanced_SWA_vs_GRU_D_Original_Bias` | `OOD_10x_20x` | `Signed_Bias_ActiveDeg` | 2.11 | 6.48 | -4.368 ± 3.880 | [-5.817, -2.920] | -6.17 | 1.0082e-06 | 3.5287e-05 | [-4.969, -3.758] | [-4.969, -3.758] | **`SUPPORTED`** |
| `GRU_D_Enhanced_SWA_vs_PatchTST_Original_Bias` | `OOD_10x_20x` | `Signed_Bias_ActiveDeg` | 2.11 | 1.56 | +0.552 ± 7.871 | [-2.388, +3.491] | 0.38 | 7.0395e-01 | 1.0000e+00 | [-0.414, +1.583] | [-0.414, +1.583] | **`NOT SUPPORTED`** |
| `GRU_D_Enhanced_SWA_vs_Plain_2ndOrder_NODE_Kin_Original_Bias` | `OOD_10x_20x` | `Signed_Bias_ActiveDeg` | 2.11 | 2.42 | -0.307 ± 3.865 | [-1.750, +1.136] | -0.43 | 6.6681e-01 | 1.0000e+00 | [-1.474, +0.813] | [-1.474, +0.813] | **`NOT SUPPORTED`** |
| `CT_PhysProb_JEPA_v2_SWA_vs_GRU_D_Original_Bias` | `OOD_10x_20x` | `Signed_Bias_ActiveDeg` | 2.53 | 6.48 | -3.949 ± 3.280 | [-5.174, -2.724] | -6.59 | 3.1632e-07 | 1.1388e-05 | [-4.552, -3.331] | [-4.552, -3.331] | **`SUPPORTED`** |
| `CT_PhysProb_JEPA_v2_SWA_vs_PatchTST_Original_Bias` | `OOD_10x_20x` | `Signed_Bias_ActiveDeg` | 2.53 | 1.56 | +0.971 ± 7.937 | [-1.993, +3.934] | 0.67 | 5.0823e-01 | 1.0000e+00 | [+0.069, +1.944] | [+0.069, +1.944] | **`NOT SUPPORTED`** |
| `CT_PhysProb_JEPA_v2_SWA_vs_Plain_2ndOrder_NODE_Kin_Original_Bias` | `OOD_10x_20x` | `Signed_Bias_ActiveDeg` | 2.53 | 2.42 | +0.112 ± 3.142 | [-1.061, +1.286] | 0.20 | 8.4618e-01 | 1.0000e+00 | [-1.052, +1.230] | [-1.052, +1.230] | **`NOT SUPPORTED`** |

### Hypothesis H5 (2 Pre-Registered Tests)

| Comparison Name | Regime | Endpoint | Model A Mean | Model B Mean | Paired Diff (Mean ± SD) | Seed-Level 95% CI | $t$-stat | $p_{\text{raw}}$ | $p_{\text{Holm}}$ | Engine Bootstrap 95% CI (Estimand 1: PreReg) | Engine Bootstrap 95% CI (Estimand 2: Sensitivity) | Confirmatory Verdict |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `CT_PhysProb_JEPA_v2_SWA_vs_CT_PhysProb_JEPA_NoPhysics_RMSE` | `OOD_10x_20x` | `RMSE_All` | 16.81 | 16.84 | -0.037 ± 0.973 | [-0.401, +0.326] | -0.21 | 8.3458e-01 | 1.0000e+00 | [-0.112, -0.003] | [-0.097, +0.022] | **`NOT SUPPORTED`** |
| `CT_PhysProb_JEPA_v2_SWA_vs_CT_PhysProb_JEPA_NoPhysics_Bias` | `OOD_10x_20x` | `Signed_Bias_ActiveDeg` | 2.53 | 2.27 | +0.265 ± 2.566 | [-0.694, +1.223] | 0.56 | 5.7664e-01 | 1.0000e+00 | [+0.147, +0.374] | [+0.147, +0.374] | **`NOT SUPPORTED`** |

*Sensitivity Verdict Stability:* Exactly 0 verdicts shift between Estimand 1 and Estimand 2 across all 40 primary confirmatory tests.

---
## 3. Exploratory Comparisons Table (Post-Hoc / Not in Pre-Registered $M=40$ Family)

Comparisons between the proposed model and other baselines (`PatchTST_RUL_v2`, `Plain_NODE_Enhanced_SWA`, `GRU_D_Enhanced_SWA`) across stress regimes that were reported in preliminary drafts but not part of the pre-registered 40-test family are tabulated below as **EXPLORATORY ONLY**:

| Exploratory Comparison Name | Regime | Endpoint | Proposed Mean | Competitor Mean | Paired Diff (Mean ± SD) | Seed-Level 95% CI | $t$-stat | $p_{\text{raw}}$ | $p_{\text{Holm}}$ (Within Exploratory) | Engine Bootstrap 95% CI (Estimand 1) | Engine Bootstrap 95% CI (Estimand 2) | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `Proposed_vs_GRU_D_Enhanced_SWA_Dropout_10pct` | `Dropout_10pct` | `RMSE_All` | 16.55 | 16.45 | +0.101 ± 1.035 | [-0.286, +0.487] | 0.53 | 5.9813e-01 | 1.0000e+00 | [+0.059, +0.300] | [-0.019, +0.223] | Exploratory |
| `Proposed_vs_Plain_NODE_Enhanced_SWA_Dropout_10pct` | `Dropout_10pct` | `RMSE_All` | 16.55 | 16.36 | +0.197 ± 0.776 | [-0.093, +0.487] | 1.39 | 1.7460e-01 | 1.0000e+00 | [+0.047, +0.225] | [+0.104, +0.289] | Exploratory |
| `Proposed_vs_PatchTST_RUL_v2_Dropout_10pct` | `Dropout_10pct` | `RMSE_All` | 16.55 | 16.07 | +0.487 ± 0.728 | [+0.216, +0.759] | 3.67 | 9.7347e-04 | 1.1682e-02 | [+0.064, +1.142] | [-0.026, +1.018] | Exploratory |
| `Proposed_vs_GRU_D_Enhanced_SWA_Dropout_20pct` | `Dropout_20pct` | `RMSE_All` | 18.54 | 18.78 | -0.243 ± 1.519 | [-0.810, +0.324] | -0.88 | 3.8836e-01 | 1.0000e+00 | [-0.178, +0.148] | [-0.405, -0.079] | Exploratory |
| `Proposed_vs_Plain_NODE_Enhanced_SWA_Dropout_20pct` | `Dropout_20pct` | `RMSE_All` | 18.54 | 18.40 | +0.137 ± 1.327 | [-0.359, +0.632] | 0.56 | 5.7664e-01 | 1.0000e+00 | [-0.059, +0.110] | [+0.042, +0.222] | Exploratory |
| `Proposed_vs_PatchTST_RUL_v2_Dropout_20pct` | `Dropout_20pct` | `RMSE_All` | 18.54 | 17.25 | +1.289 ± 1.755 | [+0.634, +1.945] | 4.02 | 3.7391e-04 | 4.8609e-03 | [+0.595, +2.404] | [+0.448, +2.139] | Exploratory |
| `Proposed_vs_GRU_D_Enhanced_SWA_Dropout_30pct` | `Dropout_30pct` | `RMSE_All` | 19.31 | 19.56 | -0.247 ± 1.498 | [-0.806, +0.312] | -0.90 | 3.7367e-01 | 1.0000e+00 | [-0.248, +0.154] | [-0.440, -0.047] | Exploratory |
| `Proposed_vs_Plain_NODE_Enhanced_SWA_Dropout_30pct` | `Dropout_30pct` | `RMSE_All` | 19.31 | 19.15 | +0.164 ± 1.442 | [-0.374, +0.703] | 0.62 | 5.3718e-01 | 1.0000e+00 | [-0.025, +0.177] | [+0.057, +0.264] | Exploratory |
| `Proposed_vs_PatchTST_RUL_v2_Dropout_30pct` | `Dropout_30pct` | `RMSE_All` | 19.31 | 18.22 | +1.097 ± 1.997 | [+0.351, +1.843] | 3.01 | 5.3807e-03 | 5.9188e-02 | [+0.018, +2.164] | [+0.121, +2.104] | Exploratory |
| `Proposed_vs_GRU_D_Enhanced_SWA_OOD_15x_25x_Bias` | `OOD_15x_25x` | `Signed_Bias_ActiveDeg` | 2.94 | 2.60 | +0.334 ± 3.526 | [-0.983, +1.650] | 0.52 | 6.0780e-01 | 1.0000e+00 | [+0.159, +0.505] | [+0.159, +0.505] | Exploratory |
| `Proposed_vs_Plain_NODE_Enhanced_SWA_OOD_15x_25x_Bias` | `OOD_15x_25x` | `Signed_Bias_ActiveDeg` | 2.94 | 2.76 | +0.175 ± 3.192 | [-1.017, +1.367] | 0.30 | 7.6601e-01 | 1.0000e+00 | [+0.082, +0.270] | [+0.082, +0.270] | Exploratory |
| `Proposed_vs_PatchTST_RUL_v2_OOD_15x_25x_Bias` | `OOD_15x_25x` | `Signed_Bias_ActiveDeg` | 2.94 | 5.18 | -2.246 ± 2.687 | [-3.249, -1.242] | -4.58 | 8.1882e-05 | 1.1463e-03 | [-2.949, -1.550] | [-2.949, -1.550] | Exploratory |
| `Proposed_vs_GRU_D_Enhanced_SWA_OOD_20x_30x_Bias` | `OOD_20x_30x` | `Signed_Bias_ActiveDeg` | 2.89 | 2.64 | +0.250 ± 3.490 | [-1.053, +1.553] | 0.39 | 6.9753e-01 | 1.0000e+00 | [+0.029, +0.474] | [+0.029, +0.474] | Exploratory |
| `Proposed_vs_Plain_NODE_Enhanced_SWA_OOD_20x_30x_Bias` | `OOD_20x_30x` | `Signed_Bias_ActiveDeg` | 2.89 | 2.69 | +0.203 ± 3.372 | [-1.056, +1.462] | 0.33 | 7.4383e-01 | 1.0000e+00 | [+0.097, +0.308] | [+0.097, +0.308] | Exploratory |
| `Proposed_vs_PatchTST_RUL_v2_OOD_20x_30x_Bias` | `OOD_20x_30x` | `Signed_Bias_ActiveDeg` | 2.89 | 5.63 | -2.735 ± 2.626 | [-3.716, -1.755] | -5.71 | 3.5806e-06 | 5.3708e-05 | [-3.557, -1.892] | [-3.557, -1.892] | Exploratory |

---
## 4. Hypothesis H3: Integrator Symplecticity Equivalence (TOST Protocol)

| Endpoint | Practical Equivalence Margin ($\pm \delta$) | Paired Difference (Mean ± SD) | 90% Equivalence CI | $p_{\text{TOST}}$ | Equivalence Established | Pre-Registered Verdict |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `RMSE_All` | $\pm 0.25$ cycles | -0.0040 ± 0.0074 | [-0.0063, -0.0017] | 1.1960e-46 | True | **`EQUIVALENCE SUPPORTED`** |
| `Signed_Bias_ActiveDeg` | $\pm 0.50$ cycles | +0.0084 ± 0.0132 | [+0.0043, +0.0125] | 1.1919e-47 | True | **`EQUIVALENCE SUPPORTED`** |

### Pre-Registered Scope Note & Mechanism Diagnosis
> *"Equivalence reflects that the ODE contribution is negligible in this parameterization, not a general statement about symplectic integration."*
- Empirical per-seed ODE correction magnitude on official FD003 test set: $|\Delta y_{\text{ODE}}| = 2.54 \pm 0.90\text{ cycles}$ (min $0.94$, max $4.72$).
- Per-seed absolute difference between Strang-symplectic and forward Euler integration: $|\Delta y_{\text{Strang}} - \Delta y_{\text{Euler}}| = 0.0210 \pm 0.0135\text{ cycles}$.
- The numerical equivalence is therefore an empirical consequence of the small magnitude and weak learning signal of the continuous-time ODE block relative to the dominant countdown hazard head.

---
## 5. Hypothesis H4b: Component Ablation Audit & Reconciliation on `GRU_D_Enhanced` Backbone

### Sign Convention Definition
All paired differences in H4b are strictly defined as:
$$\Delta = \text{Metric}(\text{All\_On}) - \text{Metric}(\text{Ablated condition})$$
- For `RMSE_All`: A negative difference ($\Delta < 0$) indicates that `All_On` achieved lower error, meaning the ablated component was **beneficial**.
- For `Signed_Bias_ActiveDeg`: A negative difference ($\Delta < 0$) indicates that `All_On` achieved lower optimistic bias.

### H4b Corrected Multi-Estimand Audit Table

| Component Ablation Condition | Endpoint | All-On Mean | Ablated Mean | Paired Diff (Mean ± SD) | Seed-Level 95% Student-$t$ CI | Engine Bootstrap 95% CI (Estimand 1: PreReg) | Engine Bootstrap 95% CI (Estimand 2: Sensitivity) | $t$-stat | $p_{\text{raw}}$ | $p_{\text{Holm}}$ | Verdict (Estimand 1) | Verdict (Estimand 2) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `H4b_All_Off` | `RMSE_All` | 16.87 | 18.24 | -1.368 ± 1.850 | [-2.059, -0.677] | [-1.088, -0.367] | [-1.717, -1.018] | -4.05 | 3.5121e-04 | 2.8097e-03 | **`COMPONENT BENEFICIAL`** | **`COMPONENT BENEFICIAL`** |
| `H4b_All_Off` | `Signed_Bias_ActiveDeg` | 2.11 | 2.94 | -0.832 ± 4.045 | [-2.342, +0.679] | [-1.476, -0.178] | [-1.476, -0.178] | -1.13 | 2.6946e-01 | 9.5089e-01 | **`NOT SUPPORTED`** | **`NOT SUPPORTED`** |
| `H4b_No_Mask` | `RMSE_All` | 16.87 | 17.38 | -0.512 ± 2.562 | [-1.469, +0.445] | [+0.300, +0.786] | [-0.768, -0.217] | -1.09 | 2.8273e-01 | 9.5089e-01 | **`NOT SUPPORTED`** | **`NOT SUPPORTED`** |
| `H4b_No_Mask` | `Signed_Bias_ActiveDeg` | 2.11 | 4.20 | -2.090 ± 7.592 | [-4.925, +0.745] | [-2.387, -1.784] | [-2.387, -1.784] | -1.51 | 1.4237e-01 | 8.0889e-01 | **`NOT SUPPORTED`** | **`NOT SUPPORTED`** |
| `H4b_No_MixedGap` | `RMSE_All` | 16.87 | 19.09 | -2.220 ± 1.966 | [-2.954, -1.486] | [-2.068, -1.201] | [-2.641, -1.792] | -6.19 | 9.5841e-07 | 9.5841e-06 | **`COMPONENT BENEFICIAL`** | **`COMPONENT BENEFICIAL`** |
| `H4b_No_MixedGap` | `Signed_Bias_ActiveDeg` | 2.11 | 1.10 | +1.013 ± 3.607 | [-0.334, +2.360] | [+0.394, +1.638] | [+0.394, +1.638] | 1.54 | 1.3481e-01 | 8.0889e-01 | **`NOT SUPPORTED`** | **`NOT SUPPORTED`** |
| `H4b_No_HazardHead` | `RMSE_All` | 16.87 | 16.52 | +0.352 ± 1.597 | [-0.245, +0.948] | [+0.149, +0.335] | [+0.262, +0.438] | 1.21 | 2.3772e-01 | 9.5089e-01 | **`NOT SUPPORTED`** | **`NOT SUPPORTED`** |
| `H4b_No_HazardHead` | `Signed_Bias_ActiveDeg` | 2.11 | 2.68 | -0.569 ± 3.800 | [-1.987, +0.850] | [-0.691, -0.440] | [-0.691, -0.440] | -0.82 | 4.1920e-01 | 9.5089e-01 | **`NOT SUPPORTED`** | **`NOT SUPPORTED`** |
| `H4b_No_SWA` | `RMSE_All` | 16.87 | 18.48 | -1.606 ± 1.640 | [-2.219, -0.994] | [-1.167, -0.400] | [-2.009, -1.209] | -5.36 | 9.2101e-06 | 8.2891e-05 | **`COMPONENT BENEFICIAL`** | **`COMPONENT BENEFICIAL`** |
| `H4b_No_SWA` | `Signed_Bias_ActiveDeg` | 2.11 | 0.39 | +1.721 ± 3.925 | [+0.256, +3.187] | [+1.264, +2.219] | [+1.264, +2.219] | 2.40 | 2.2936e-02 | 1.6055e-01 | **`NOT SUPPORTED`** | **`NOT SUPPORTED`** |

### Statistical Explanation: Why Seed-Level Means Lie Outside Estimand 1 CIs
In non-linear metrics such as RMSE, Jensen's inequality dictates that the RMSE of a seed-averaged prediction is strictly strictly less than or equal to the average of individual-seed RMSEs:
$$\text{RMSE}\left(\frac{1}{30}\sum_{s=1}^{30} \hat{y}^{(s)}\right) \le \frac{1}{30}\sum_{s=1}^{30} \text{RMSE}(\hat{y}^{(s)})$$
Under Estimand 1 (pre-registered), bootstrap resamples engines and evaluates the metric on the seed-averaged ensemble prediction. Ensembling removes zero-mean seed-to-seed variance, shrinking the baseline and ablated errors differentially and shifting the center of the bootstrap distribution away from the arithmetic mean of individual single-seed paired differences $\bar{\Delta}$. Under Estimand 2 (sensitivity), the metric is evaluated per seed and differences are averaged across seeds, completely reconciling the bootstrap center with the seed-level mean $\bar{\Delta}$.

### Structural Reconciliation: H4 vs. H4b
In Hypothesis H4, `GRU_D_Original` achieved an RMSE of $16.63 \pm 0.67$ (slightly lower than `GRU_D_Enhanced` at $16.87 \pm 1.25$, diff $+0.24$), whereas in H4b, `H4b_All_Off` achieved an RMSE of $18.24 \pm 1.48$ (worse than `All_On` by $-1.37$ cycles). This apparent divergence is fully explained by their exact architectural configurations:

| Hyperparameter / Component | `GRU_D_Original` (H4 Competitor) | `H4b_All_Off` (H4b Ablation Arm) | Architectural Impact |
|:---|:---:|:---:|:---|
| **Model Architecture Class** | `GRU_D_Original_Configurable` | `GRU_D_LinearProbe` | Canonical GRU-D cell vs Temporal Encoder + Probe |
| **Input Dimension ($D$)** | **42** (Sensors + Fast/Slow EWMA) | **56** (Sensors + Fast/Slow EWMA + 14 Mask Channels) | `GRU_D_Original` receives no mask overhead |
| **Hidden Dimension ($d$)** | **80** (Re-frozen from grid extension) | **48** (Fixed to match `GRU_D_Enhanced`) | **$67\%$ larger recurrent state** in `GRU_D_Original` |
| **Learning Rate ($\eta$)** | **$3.5 \times 10^{-3}$** | **$5.0 \times 10^{-3}$** | $\eta=5e-3$ is tuned for SWA and hazard head; without SWA, it causes sub-optimal convergence |
| **Training Augmentation** | Short dropouts only ($p=0.25$) | Short dropouts only ($p=0.25$) | Identical short-gap training data |
| **SWA** | No | No | Both un-averaged checkpoints |
| **Prediction Head** | Linear-Tanh-Linear Probe | Linear-Tanh-Linear Probe | Identical probe head architecture |

*Reconciliation Conclusion:* `H4b_All_Off` evaluates the effect of removing the 4 recipe components while holding the backbone capacity and learning rate fixed at $d=48, \eta=5e-3$, where stripping the components degrades performance by $1.37$ cycles. In contrast, `GRU_D_Original` was independently re-tuned across an expanded grid to $d=80, \eta=3.5e-3$, where its $67\%$ larger recurrent capacity compensates for the lack of mixed-gap augmentation on point RMSE.

### Correction of Bias Claims
- **Retraction:** The claim that 'the KneeHazardHead contributes a $-4.0$-cycle bias reduction' is **RETRACTED**. In H4b, the leave-one-out ablation `H4b_No_HazardHead` produced a non-significant active bias difference of only $-0.569 \pm 3.800\text{ cycles}$ ($p_{\text{raw}} = 0.420, p_{\text{Holm}} = 0.951$).
- The $-4.0$-cycle bias reduction observed in H4 occurred when comparing `GRU_D_Enhanced` to `GRU_D_Original`, which reflected the cumulative effect of masking, asymmetric safety loss, and head integration—not the hazard head in isolation.

---
## 6. Comprehensive Secondary Results Portfolio (Pre-Registered Endpoints)

### 6.1 Split-Conformal 90% Empirical Coverage & Interval Width across Regimes

| Model Architecture | Metric | InDist (1x-2x) | OOD (10x-20x) | Extended (15x-25x) | Extreme (20x-30x) | Dropout 10% | Dropout 20% | Dropout 30% |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `CT_PhysProb_JEPA_v2_SWA` | 90% Coverage | 96.7% | 89.5% | 87.7% | 84.8% | 90.1% | 86.9% | 85.6% |
| | Mean Width | 38.3c | 41.9c | 44.2c | 46.9c | 41.3c | 42.8c | 43.1c |
| `CT_PhysProb_JEPA_NoPhysics` | 90% Coverage | 96.9% | 89.5% | 88.0% | 85.1% | 90.2% | 87.1% | 85.8% |
| | Mean Width | 38.3c | 42.0c | 44.4c | 47.0c | 41.5c | 43.0c | 43.2c |
| `Euler_Damped` | 90% Coverage | 96.7% | 89.5% | 87.6% | 84.7% | 90.1% | 86.8% | 85.6% |
| | Mean Width | 38.3c | 41.9c | 44.2c | 46.9c | 41.3c | 42.8c | 43.1c |
| `Undamped` | 90% Coverage | 96.8% | 89.5% | 87.5% | 83.9% | 90.1% | 86.9% | 85.6% |
| | Mean Width | 38.3c | 41.9c | 44.3c | 47.3c | 41.3c | 42.8c | 43.1c |
| `GRU_D_Enhanced_SWA` | 90% Coverage | 96.8% | 89.9% | 88.6% | 85.8% | 91.1% | 87.3% | 86.0% |
| | Mean Width | 38.9c | 42.8c | 45.3c | 48.0c | 42.0c | 43.4c | 43.5c |
| `GRU_D_Canonical_SWA` | 90% Coverage | 97.0% | 89.9% | 88.6% | 86.7% | 90.8% | 87.9% | 86.8% |
| | Mean Width | 38.6c | 42.1c | 43.9c | 46.0c | 41.5c | 42.3c | 42.5c |
| `Plain_NODE_Enhanced_SWA` | 90% Coverage | 96.8% | 89.5% | 87.5% | 84.3% | 90.4% | 87.0% | 85.7% |
| | Mean Width | 37.9c | 41.4c | 43.8c | 46.5c | 40.7c | 42.4c | 42.6c |
| `PatchTST_RUL_v2` | 90% Coverage | 96.6% | 91.1% | 89.5% | 88.2% | 91.3% | 89.6% | 88.2% |
| | Mean Width | 38.5c | 42.1c | 43.4c | 45.0c | 40.9c | 40.6c | 39.5c |
| `GRU_D_Original` | 90% Coverage | 98.0% | 91.5% | 90.2% | 89.5% | 91.6% | 86.3% | 83.0% |
| | Mean Width | 42.3c | 43.6c | 43.1c | 43.2c | 41.4c | 38.6c | 36.7c |
| `PatchTST_Original` | 90% Coverage | 97.3% | 88.5% | 85.4% | 84.7% | 89.3% | 88.7% | 87.9% |
| | Mean Width | 42.1c | 47.9c | 49.0c | 49.4c | 47.4c | 47.0c | 46.6c |
| `Plain_2ndOrder_NODE_Kin_Original` | 90% Coverage | 98.9% | 89.6% | 88.2% | 87.8% | 90.9% | 90.7% | 90.6% |
| | Mean Width | 48.9c | 52.3c | 52.7c | 53.6c | 50.9c | 48.9c | 47.5c |
| `CT_PhysProb_JEPA_v2_SWA_Countdown` | Coverage / Width | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| `TimeCorrected_LOCF_Oracle` | Coverage / Width | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

### 6.2 RMSE_Last (Final Step Failure Prediction) across Regimes

| Model Architecture | InDist (1x-2x) | OOD (10x-20x) | Extended (15x-25x) | Extreme (20x-30x) | Dropout 10% | Dropout 20% | Dropout 30% |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `CT_PhysProb_JEPA_v2_SWA` | 13.70 ± 0.31 | 19.65 ± 0.47 | 16.80 ± 0.57 | 19.70 ± 0.55 | 21.60 ± 1.42 | 24.19 ± 2.32 | 25.69 ± 2.66 |
| `CT_PhysProb_JEPA_NoPhysics` | 13.68 ± 0.30 | 19.60 ± 0.41 | 16.77 ± 0.45 | 19.69 ± 0.53 | 21.66 ± 1.53 | 24.16 ± 2.09 | 25.73 ± 2.38 |
| `Euler_Damped` | 13.70 ± 0.31 | 19.65 ± 0.48 | 16.80 ± 0.58 | 19.69 ± 0.55 | 21.60 ± 1.42 | 24.19 ± 2.32 | 25.69 ± 2.66 |
| `Undamped` | 13.70 ± 0.31 | 19.65 ± 0.49 | 16.83 ± 0.58 | 19.69 ± 0.56 | 21.60 ± 1.44 | 24.20 ± 2.33 | 25.71 ± 2.64 |
| `GRU_D_Enhanced_SWA` | 13.76 ± 0.44 | 19.83 ± 0.61 | 16.70 ± 0.68 | 19.56 ± 0.82 | 22.05 ± 1.79 | 25.24 ± 2.83 | 27.30 ± 3.11 |
| `GRU_D_Canonical_SWA` | 13.71 ± 0.29 | 19.55 ± 0.39 | 16.37 ± 0.32 | 19.19 ± 0.48 | 21.67 ± 1.06 | 25.23 ± 1.92 | 27.70 ± 2.31 |
| `Plain_NODE_Enhanced_SWA` | 13.73 ± 0.36 | 19.62 ± 0.48 | 16.72 ± 0.42 | 19.66 ± 0.57 | 21.82 ± 1.22 | 24.20 ± 1.97 | 26.08 ± 2.15 |
| `PatchTST_RUL_v2` | 14.13 ± 0.44 | 19.65 ± 0.54 | 17.68 ± 0.63 | 19.99 ± 0.57 | 21.39 ± 1.47 | 24.21 ± 2.58 | 27.86 ± 2.96 |
| `GRU_D_Original` | 13.90 ± 0.30 | 20.17 ± 0.49 | 18.03 ± 0.89 | 21.61 ± 0.99 | 24.29 ± 1.28 | 32.65 ± 2.11 | 38.43 ± 2.47 |
| `PatchTST_Original` | 14.15 ± 0.61 | 20.77 ± 0.86 | 18.61 ± 1.55 | 22.07 ± 1.23 | 21.87 ± 1.35 | 25.05 ± 1.82 | 28.69 ± 2.24 |
| `Plain_2ndOrder_NODE_Kin_Original` | 14.29 ± 0.58 | 21.61 ± 0.84 | 21.99 ± 1.46 | 25.25 ± 1.15 | 23.30 ± 1.35 | 27.82 ± 1.82 | 31.39 ± 1.83 |
| `CT_PhysProb_JEPA_v2_SWA_Countdown` | nan ± nan | 19.22 ± 0.30 | nan ± nan | nan ± nan | nan ± nan | nan ± nan | nan ± nan |
| `TimeCorrected_LOCF_Oracle` | nan ± nan | 3.63 ± 0.00 | nan ± nan | nan ± nan | nan ± nan | nan ± nan | nan ± nan |

### 6.3 Error Decomposition: Anchor RMSE vs. Gap-Induced RMSE (`OOD_10x_20x`)

| Model Architecture | Total RMSE (Mean ± SD) | Anchor RMSE (Mean ± SD) | Gap-Induced RMSE (Mean ± SD) | Variance Share Anchor ($A^2 / T^2$) | Late Prediction Error (Mean ± SD) |
|:---|:---:|:---:|:---:|:---:|:---:|
| `CT_PhysProb_JEPA_v2_SWA` | 16.81 ± 0.83 | 14.16 ± 0.89 | +2.65 ± 0.37 | 71.0% | 8.83 ± 1.04 |
| `CT_PhysProb_JEPA_NoPhysics` | 16.84 ± 0.79 | 14.17 ± 0.79 | +2.67 ± 0.34 | 70.8% | 8.66 ± 0.92 |
| `Euler_Damped` | 16.81 ± 0.83 | 14.16 ± 0.89 | +2.65 ± 0.37 | 70.9% | 8.82 ± 1.04 |
| `Undamped` | 16.83 ± 0.85 | 14.16 ± 0.90 | +2.67 ± 0.37 | 70.8% | 8.82 ± 1.07 |
| `GRU_D_Enhanced_SWA` | 16.87 ± 1.25 | 14.31 ± 1.25 | +2.56 ± 0.33 | 72.0% | 8.64 ± 1.43 |
| `GRU_D_Canonical_SWA` | 16.58 ± 0.80 | 13.84 ± 0.66 | +2.74 ± 0.46 | 69.7% | 8.83 ± 0.89 |
| `Plain_NODE_Enhanced_SWA` | 16.74 ± 0.84 | 14.01 ± 0.80 | +2.73 ± 0.47 | 70.1% | 8.87 ± 0.92 |
| `PatchTST_RUL_v2` | 16.23 ± 0.59 | 13.23 ± 0.67 | +3.00 ± 0.19 | 66.4% | 10.01 ± 1.07 |
| `GRU_D_Original` | 16.63 ± 0.67 | 14.39 ± 0.56 | +2.24 ± 0.61 | 74.8% | 11.40 ± 1.06 |
| `PatchTST_Original` | 19.69 ± 4.02 | 14.10 ± 0.94 | +5.59 ± 4.02 | 51.3% | 9.49 ± 3.56 |
| `Plain_2ndOrder_NODE_Kin_Original` | 21.13 ± 1.81 | 15.77 ± 0.68 | +5.36 ± 1.84 | 55.7% | 10.69 ± 1.49 |
| `CT_PhysProb_JEPA_v2_SWA_Countdown` | 22.09 ± 1.06 | 14.16 ± 0.89 | +7.93 ± 0.27 | 41.1% | 6.06 ± 0.75 |
| `TimeCorrected_LOCF_Oracle` | 12.26 ± 0.00 | 0.00 ± 0.00 | +12.26 ± 0.00 | 0.0% | 0.00 ± 0.00 |

---
## 7. Corrected Deviations & Reconciliation Log

1. **Pre-Registration Comparison List Reconciliation:**
   - In preliminary reporting drafts, comparisons between the proposed model and non-preregistered models (`PatchTST_RUL_v2`, `Plain_NODE_Enhanced_SWA`, `GRU_D_Enhanced_SWA`) across stress regimes were inadvertently grouped under H1 and H2.
   - **Correction:** The complete pre-registered set of 40 tests (comparing the Recipe Family to `GRU_D_Canonical_SWA`, original baselines, and `NoPhysics`) has been restored and evaluated under joint Holm correction. Non-preregistered comparisons have been moved to Section 3 as **EXPLORATORY ONLY**.
2. **Labeling Corrections:**
   - `PatchTST_RUL_v2` is explicitly identified as having the `KneeHazardHead` (not a linear probe).
   - Conformal coverage and interval widths are correctly designated as `N/A` for deterministic point reference heuristics (`TimeCorrected_LOCF_Oracle` and `Countdown`).
3. **H3 Language Revision:**
   - Rewritten per pre-registered scope note: equivalence reflects negligible ODE contribution in this parameterization; general symplectic benefits in other physical domains are not claimed to be falsified.
4. **H4b Audit & Reconciliation:**
   - Sign convention explicitly fixed: $\Delta = \text{All\_On} - \text{Ablated}$.
   - Dual bootstrap estimands reported; mathematical basis for Estimand 1 mean shifts explained via Jensen's inequality.
   - Exact architectural differences between `H4b_All_Off` and `GRU_D_Original` documented.
   - Unsubstantiated $-4.0$-cycle bias claim for the hazard head retracted.
5. **Secondary Dataset Scope:**
   - NASA C-MAPSS FD004 pre-registered under Amendment v2.6 below; evaluation halted for user review prior to remote tag resolution and execution.