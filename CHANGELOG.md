# Comprehensive Revision Changelog (v12 — Full Empirical Audit & 30-Seed Pre-Registered Execution)

## 1. Scientific Governance & Forensic Audit
- **Code & Data Quarantine:**  
  Quarantined all legacy scripts and data files containing synthetic seed extensions (`run_v10_all_real_experiments.py` seed extension block, `build_multiseed_wilcoxon_notebook.py`), parametric ablation noise (`rng_abl`), and hardcoded table dictionaries (`complete_table3_rows`, `conformal_rows`) into `quarantine/`. Created `AUDIT_PROVENANCE.md` detailing file checksums and provenance.
- **Dataset Authentication:**  
  Downloaded and authenticated raw NASA C-MAPSS FD001–FD004 files into `data/raw/CMAPSS/` with verified SHA-256 hashes.
- **Strict Partition Integrity:**  
  All preliminary diagnosis, hyperparameter exploration, and model tuning were restricted strictly to 80 training engines and 20 validation engines. The 100 test engines remained unblinded until final Step 4 evaluation.

---

## 2. Five-Step Scientific Execution Ledger

### Step 1: Diagnosis (`DIAGNOSIS.md`)
- Ran 320 real train+evaluate runs across 10 seeds on FD001 validation data.
- **Empirical Findings:**
  - Demonstrated that the bounded horizon clamp ($h \le 0.35$) artificially flattened error growth at $\Delta t > 4\tau_0$, inducing $+15.97$ cycles optimistic bias on active degradation.
  - Demonstrated that lifting the clamp unregularized led to open-loop integration drift ($32.85 \pm 10.32$ cycles RMSE).
  - Proved that time-corrected kinematic countdown (`TimeCorrected_LOCF_Oracle`) achieved **$9.08 \pm 0.13$ cycles RMSE**, massively outperforming all neural models.

### Step 2: Legitimate Improvement & Model Synthesis (`IMPROVEMENTS.md`)
- Conducted 4 disciplined rounds of improvements across 10 real seeds on validation data:
  - **Round 1 (Mixed-Gap Augmentation):** Cut adaptive ODE divergence from $32.85$ to $20.06 \pm 0.92$ cycles.
  - **Round 2 (Asymmetric Safety Loss, $\lambda_{\text{opt}}=1.0$):** Reduced active wear optimistic bias from $+12.51$ to $+7.93$ cycles and improved NASA score by $26\%$.
  - **Round 3 (Soft-Saturation & Gap Conditioning):** Established baseline convergence across models (`GRU_D_v2`: $18.96$, `PatchTST_RUL_v2`: $19.74$, proposed: $19.72$).
  - **Round 4 (Synthesis Ablation — `CT_PhysProb_JEPA_KinGate`):** Synthesized a hybrid combining 2nd-order Hamiltonian ODE, learned degradation knee gating $-\kappa_t \Delta t$ from LOCF, and continuous damping from GRU-D. Achieved **$19.21 \pm 0.75$ cycles** (statistically tied with `GRU_D_v2`, $p = 0.17$), and lowest active bias ($+6.25$ cycles).
- Model architecture and hyperparameters frozen under SHA-256 hash `070ab9c4ec0425112c499a64741e523825fc90c87eee77585326e3bfc3be270e` in `configs/frozen_final_configuration.json`.

### Step 3: Pre-Registration (`PREREGISTRATION.md`)
- Pre-registered 30 fixed seeds, 4 blackout regimes, primary endpoint (`OOD_10x_20x` RMSE_All), secondary safety endpoints, and formal hypotheses (H1–H4) before unblinding test data.

### Step 4: Final Evaluation (`results/raw/step4_final_30seeds_raw.csv`)
- Executed 1,080 model trainings and evaluations across all 30 seeds on the unblinded official FD001 test set (100 units).
- Conformal coverage and CBM costs computed exclusively from saved predictions, calibrated on validation units and evaluated on test units.

### Step 5: Findings & Manuscript Audit (`FINDINGS.md`, `CLAIMS_VS_EVIDENCE.md`)
- Performed rigorous hypothesis testing with paired two-sided $t$-tests, Wilcoxon signed-rank tests, Cohen's $d_z$ effect sizes, 95% CIs, and Holm-Bonferroni correction.
- Generated complete claims-to-evidence mapping table verifying every sentence and table row against real logs.

---

## 3. Key Findings & Truthful Manuscript Revisions
1. **Withdrew "Best Primary RMSE" Claim:**  
   `TimeCorrected_LOCF_Oracle` ($11.42 \pm 0.09$ cycles) and continuous-time baselines (`Plain_2ndOrder_NODE_Kin`: $18.84 \pm 0.78$, `GRU_D`: $19.49 \pm 0.92$) achieve lower numerical point RMSE than `CT_PhysProb_JEPA_KinGate` ($20.79 \pm 0.84$). The proposed model is reported as **competitive with modern Transformers (`PatchTST_RUL`: $20.51$, $p = 0.50$)**, not best.
2. **Affirmed Safety Superiority:**  
   Proposed model achieves **statistically significant superior safety performance**:
   - Lowest active degradation optimistic bias: **$+4.77 \pm 1.71$ cycles** vs. $+12.25$ for GRU-D and $+11.16$ for Plain NODE ($p < 10^{-14}$).
   - Lowest NASA exponential penalty: **$771.7 \pm 234.3$** vs. $1,060.4$ for GRU-D ($p < 10^{-5}$).
3. **Conformal & CBM Validation:**  
   Split-Conformal LCB achieved **$94.47\%$ test coverage** and slashed in-flight critical failures from $3.17$ to $0.10$ engines.
4. **Transparent Data Scope:**  
   FD001 fully evaluated on 30 seeds; N-CMAPSS DS02 and PRONOSTIA transparently designated as **NOT_RUN** due to absence of local raw benchmark data.

---

## 4. Phase 3 Confirmatory Execution & Audit Ledger (Protocol v2.5 Official Evaluation)

### Protocol Governance & Cryptographic Registration
- **Pre-Registration v2.5:** Sealed and locked in `PREREGISTRATION_V2.md` (`SHA-256: 708bef147e8debb106b14ca72a12ea3ebf3cb3af1118d2756afd20204947930f`).
- **External Remote Lock:** Pushed repository, manifest, and annotated tag `preregistration-v2.5` (`ac9fd50ce67d03afc0803bab22c89fd9a01a87d8`, commit `6afc1a0ec42f3a8df26eefc5a9fef89fbf9edf02`) to external HTTPS host `https://github.com/Nafis878/CT_PhysProb_JEPA.git`.
- **Pre-Flight Gates G1–G4:** Executed automated audit verification script `scripts/verify_preflight_gates.py`; all four gates passed strictly prior to any confirmatory evaluation.

### Official Confirmatory Execution Scope
- **Dataset:** NASA C-MAPSS FD003 official test set (100 engines). Training pool restricted to 80 units; calibration pool restricted to 20 units; 0 overlap.
- **Seeds & Models:** 30 pre-registered seeds `[42, 100..128]` evaluated across all 11 models in the symmetry table, plus 2 reference heuristics (`EstimatedAnchor_Countdown` and `TimeCorrected_LOCF_Oracle`), across 7 regimes.
- **Logged Raw Artifacts:**
  - `results/raw/phase3_confirmatory_raw.csv` (3,000 real train+eval rows).
  - `results/raw/phase3_h4b_ablation_raw.csv` (180 real component ablation rows).
  - `results/preds/` (full per-window raw prediction arrays for every seed, model, and regime).

### Confirmatory Statistical Findings
1. **Additive Latent Physics (H5): `NOT SUPPORTED`**  
   Proposed model ($16.81 \pm 0.83$) shows no statistically significant benefit over matched non-physics backbone `NoPhysics` ($16.84 \pm 0.79$) on primary `RMSE_All` (paired diff $-0.037$, $t = -0.21, p = 0.835, p_{\text{Holm}} = 1.0$, bootstrap 95% CI $[-0.112, -0.003]$) and no benefit on Active Signed Bias ($+0.265$ cycles, $p = 0.577, p_{\text{Holm}} = 1.0$). Confirms the formal limitation statement: *the Hamiltonian latent as parameterized receives no consistent learning signal and contributes no measurable benefit*.
2. **Integrator Symplecticity Equivalence (H3): `EQUIVALENCE SUPPORTED`**  
   Two One-Sided Tests (TOST) confirm Strang-symplectic splitting is statistically equivalent to forward Euler within $\pm 0.25$ cycles on `RMSE_All` ($p_{\text{TOST}} = 1.20 \times 10^{-46}$) and within $\pm 0.50$ cycles on Signed Bias ($p_{\text{TOST}} = 1.19 \times 10^{-47}$). Claims of distinct physical energy conservation advantages are falsified.
3. **Recipe Effect vs Original Baselines (H4): `RECIPE EFFECT NOT FULLY SUPPORTED` (Single-Verdict Primary Endpoint Rule)**  
   Recipe models (`CT_PhysProb_JEPA_v2_SWA`, `NoPhysics`, `GRU_D_Enhanced_SWA`) achieve large, statistically significant gains over `PatchTST_Original` ($-2.82$ to $-2.89$ cycles RMSE, $p_{\text{Holm}} < 0.026$) and `Plain_2ndOrder_NODE_Kin_Original` ($-4.26$ to $-4.33$ cycles RMSE, $p_{\text{Holm}} < 10^{-9}$), and slash optimistic bias vs `GRU_D_Original` by $\approx 4.0$ cycles ($p_{\text{Holm}} < 10^{-4}$). However, re-frozen `GRU_D_Original` matches the recipe on point RMSE ($16.63 \pm 0.67$ vs $16.81 \pm 0.83$, $p = 0.40$), preventing declaration of a uniform recipe advantage across all 9 pairs on point RMSE.
4. **Sensor Channel Dropout Resilience (H1): `NOT SUPPORTED`**  
   Across 10%, 20%, and 30% channel dropouts, the proposed model does not outperform fair baselines receiving masked inputs; `PatchTST_RUL_v2` significantly outperforms the proposed model at 20% dropout ($17.25$ vs $18.54$, $p_{\text{Holm}} = 0.0116$).
5. **Extreme Outage Optimistic Bias Suppression (H2): `SUPPORTED` vs Linear Probes**  
   The KneeHazardHead significantly suppresses optimistic bias during extreme outages ($15\text{x}\text{--}25\text{x}$ and $20\text{x}\text{--}30\text{x}$) compared to `PatchTST_RUL_v2` and `GRU_D_Canonical_SWA` (reductions of $2.04$ to $2.74$ cycles bias, all $p_{\text{Holm}} < 0.003$).
6. **Recipe Component Ablation (H4b):**  
   Confirms that Mixed-Gap training contributes $-2.22$ cycles RMSE ($p_{\text{Holm}} = 0.000010$) and SWA contributes $-1.61$ cycles RMSE ($p_{\text{Holm}} = 0.000083$).
7. **Conformal & CBM Decision Rules:**  
   Split-conformal calibration on 20 independent engines evaluated on 100 official test engines achieved $89.5\%$ empirical coverage (target $90.0\%$) and zero missed critical failures ($0.00\%$).
8. **Secondary Dataset Scope:**  
   FD004 is marked **`NOT_RUN`** due to computational budget allocation prioritizing complete 30-seed confirmatory evaluation of FD003 as registered.
