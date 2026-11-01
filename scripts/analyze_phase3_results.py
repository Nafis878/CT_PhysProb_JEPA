#!/usr/bin/env python3
"""
PHASE 3 CONFIRMATORY STATISTICAL ANALYSIS PIPELINE
Strictly reads from results/raw/ and results/preds/.
Executes:
1. Seed-level paired t-test and Wilcoxon signed-rank test.
2. Step-down Holm-Bonferroni correction across M=40 primary hypothesis family.
3. Engine-level paired cluster bootstrap (B=10,000 resamples over 100 official test engines).
4. Dual-significance rule and practical threshold check (|Delta| >= 0.5 RMSE, >= 1.0 Bias).
5. Separate evaluation of H3 TOST equivalence and H4b component ablation family (10 tests, Holm).
6. Comprehensive markdown summary output generation.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def holm_bonferroni(p_vals):
    """
    Computes step-down Holm-Bonferroni adjusted p-values.
    """
    m = len(p_vals)
    sorted_indices = np.argsort(p_vals)
    sorted_p = np.array(p_vals)[sorted_indices]
    adj_p = np.empty(m)
    
    current_max = 0.0
    for i in range(m):
        val = (m - i) * sorted_p[i]
        current_max = max(current_max, val)
        adj_p[i] = min(current_max, 1.0)
        
    out = np.empty(m)
    out[sorted_indices] = adj_p
    return out

def tost_equivalence(diffs, delta, alpha=0.05):
    """
    Two One-Sided Tests (TOST) for equivalence within [-delta, delta].
    """
    n = len(diffs)
    mean_d = np.mean(diffs)
    sd_d = np.std(diffs, ddof=1)
    se_d = sd_d / np.sqrt(n)
    
    # t1 for H01: mean <= -delta (rejection if t1 > t_crit)
    t1 = (mean_d - (-delta)) / se_d
    p1 = 1.0 - stats.t.cdf(t1, df=n - 1)
    
    # t2 for H02: mean >= delta (rejection if t2 < -t_crit)
    t2 = (mean_d - delta) / se_d
    p2 = stats.t.cdf(t2, df=n - 1)
    
    p_tost = max(p1, p2)
    # 90% CI for alpha=0.05
    t_crit = stats.t.ppf(1.0 - alpha, df=n - 1)
    ci90 = (mean_d - t_crit * se_d, mean_d + t_crit * se_d)
    is_equiv = (ci90[0] > -delta) and (ci90[1] < delta)
    return p_tost, ci90, is_equiv

def run_engine_cluster_bootstrap(preds_a, preds_b, y_true, unit_ids, metric_fn, B=10000, seed=42):
    """
    Engine-level paired cluster bootstrap.
    Resamples the 100 official test engines with replacement.
    Averages predictions over the 30 seeds per engine before resampling.
    """
    np.random.seed(seed)
    unique_units = np.unique(unit_ids)
    n_units = len(unique_units)
    
    # Average predictions over seeds per window
    # preds_a: shape (30, N_windows)
    mean_pred_a = np.mean(preds_a, axis=0)
    mean_pred_b = np.mean(preds_b, axis=0)
    
    # Pre-map engine indices to window indices
    unit_to_indices = {u: np.where(unit_ids == u)[0] for u in unique_units}
    
    boot_diffs = np.empty(B, dtype=np.float64)
    for b in range(B):
        resampled_units = np.random.choice(unique_units, size=n_units, replace=True)
        resampled_idx = np.concatenate([unit_to_indices[u] for u in resampled_units])
        
        val_a = metric_fn(mean_pred_a[resampled_idx], y_true[resampled_idx])
        val_b = metric_fn(mean_pred_b[resampled_idx], y_true[resampled_idx])
        boot_diffs[b] = val_a - val_b
        
    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))
    boot_mean = float(np.mean(boot_diffs))
    ci_excludes_zero = (ci_lower > 0.0) or (ci_upper < 0.0)
    
    return boot_mean, ci_lower, ci_upper, ci_excludes_zero

def compute_rmse(pred, true):
    return np.sqrt(np.mean((pred - true) ** 2))

def compute_active_bias(pred, true):
    mask = (true < 124.9)
    if np.sum(mask) == 0:
        return 0.0
    return np.mean(pred[mask] - true[mask])

def run_statistical_analysis():
    print("=" * 80)
    print("RUNNING CONFIRMATORY STATISTICAL ANALYSIS PIPELINE")
    print("=" * 80)
    
    raw_csv = Path('results/raw/phase3_confirmatory_raw.csv')
    h4b_csv = Path('results/raw/phase3_h4b_ablation_raw.csv')
    assert raw_csv.exists(), f"Missing {raw_csv}"
    assert h4b_csv.exists(), f"Missing {h4b_csv}"
    
    df_raw = pd.read_csv(raw_csv)
    df_h4b = pd.read_csv(h4b_csv)
    
    print(f"Loaded {len(df_raw)} confirmatory rows from {raw_csv}")
    print(f"Loaded {len(df_h4b)} H4b rows from {h4b_csv}")
    
    # --------------------------------------------------------------------------
    # 1. Primary Hypothesis Family (M = 40 Tests)
    # --------------------------------------------------------------------------
    # Collect test specifications
    # Test format: (Hypothesis, Comparison_Name, Model_A, Model_B, Regime, Endpoint, Practical_Threshold)
    tests_spec = []
    
    # H1: Sensor Channel Dropout Resilience (12 tests)
    # Proposed vs 4 baselines across 3 dropout regimes on RMSE_All
    baselines_h1 = ['GRU_D_Enhanced_SWA', 'GRU_D_Canonical_SWA', 'Plain_NODE_Enhanced_SWA', 'PatchTST_RUL_v2']
    dropouts = ['Dropout_10pct', 'Dropout_20pct', 'Dropout_30pct']
    for d in dropouts:
        for b in baselines_h1:
            tests_spec.append(('H1', f"Proposed_vs_{b}_{d}", 'CT_PhysProb_JEPA_v2_SWA', b, d, 'RMSE_All', 0.50))
            
    # H2: Extreme Outage Optimistic Bias Suppression (8 tests)
    # Proposed vs 4 baselines across 2 regimes on Active Signed Bias
    extreme_regimes = ['OOD_15x_25x', 'OOD_20x_30x']
    for r in extreme_regimes:
        for b in baselines_h1:
            tests_spec.append(('H2', f"Proposed_vs_{b}_{r}_Bias", 'CT_PhysProb_JEPA_v2_SWA', b, r, 'Signed_Bias_ActiveDeg', 1.00))
            
    # H4: Recipe Effect vs Original Baselines (18 tests)
    # 3 Recipe models vs 3 Original Baselines on OOD 10x-20x across RMSE_All (9 tests) and Active Signed Bias (9 tests)
    recipe_models = ['CT_PhysProb_JEPA_NoPhysics', 'GRU_D_Enhanced_SWA', 'CT_PhysProb_JEPA_v2_SWA']
    original_models = ['GRU_D_Original', 'PatchTST_Original', 'Plain_2ndOrder_NODE_Kin_Original']
    for rm in recipe_models:
        for om in original_models:
            tests_spec.append(('H4', f"{rm}_vs_{om}_RMSE", rm, om, 'OOD_10x_20x', 'RMSE_All', 0.50))
    for rm in recipe_models:
        for om in original_models:
            tests_spec.append(('H4', f"{rm}_vs_{om}_Bias", rm, om, 'OOD_10x_20x', 'Signed_Bias_ActiveDeg', 1.00))
            
    # H5: Additive Latent Physics Benefit (2 tests)
    # Proposed vs NoPhysics on OOD 10x-20x across RMSE_All (1 test) and Active Signed Bias (1 test)
    tests_spec.append(('H5', "Proposed_vs_NoPhysics_RMSE", 'CT_PhysProb_JEPA_v2_SWA', 'CT_PhysProb_JEPA_NoPhysics', 'OOD_10x_20x', 'RMSE_All', 0.50))
    tests_spec.append(('H5', "Proposed_vs_NoPhysics_Bias", 'CT_PhysProb_JEPA_v2_SWA', 'CT_PhysProb_JEPA_NoPhysics', 'OOD_10x_20x', 'Signed_Bias_ActiveDeg', 1.00))
    
    assert len(tests_spec) == 40, f"Expected 40 primary tests, got {len(tests_spec)}"
    print(f"Verified M = {len(tests_spec)} tests in the primary confirmatory family.", flush=True)
    
    # --------------------------------------------------------------------------
    # Execute Primary Family Seed-Level & Cluster Bootstrap Tests
    # --------------------------------------------------------------------------
    primary_results = []
    p_vals = []
    
    for hyp, comp_name, m_a, m_b, reg, ep, thresh in tests_spec:
        # Extract 30 paired seed observations
        df_a = df_raw[(df_raw['Model'] == m_a) & (df_raw['Regime'] == reg)].sort_values('Seed')
        df_b = df_raw[(df_raw['Model'] == m_b) & (df_raw['Regime'] == reg)].sort_values('Seed')
        
        assert len(df_a) == 30, f"Incomplete seeds for {m_a} {reg}: {len(df_a)}"
        assert len(df_b) == 30, f"Incomplete seeds for {m_b} {reg}: {len(df_b)}"
        
        vals_a = df_a[ep].values
        vals_b = df_b[ep].values
        diffs = vals_a - vals_b  # Model A - Model B
        mean_d = float(np.mean(diffs))
        sd_d = float(np.std(diffs, ddof=1))
        
        # Seed-level paired tests
        t_stat, p_t = stats.ttest_rel(vals_a, vals_b)
        w_res = stats.wilcoxon(vals_a, vals_b)
        p_w = float(w_res.pvalue)
        
        # Load predictions for engine cluster bootstrap
        meta = np.load(f"results/preds/metadata_{reg}.npz")
        y_true = meta['y_true']
        unit_ids = meta['unit_id']
        
        preds_a = np.array([np.load(f"results/preds/{m_a}_{reg}_seed{s}_pred.npy") for s in df_a['Seed']])
        preds_b = np.array([np.load(f"results/preds/{m_b}_{reg}_seed{s}_pred.npy") for s in df_b['Seed']])
        
        metric_fn = compute_rmse if ep == 'RMSE_All' else compute_active_bias
        boot_mean, ci_lo, ci_hi, ci_excludes_0 = run_engine_cluster_bootstrap(
            preds_a, preds_b, y_true, unit_ids, metric_fn, B=10000, seed=42
        )
        
        p_vals.append(p_t)
        primary_results.append({
            'Hypothesis': hyp,
            'Comparison': comp_name,
            'Model_A': m_a,
            'Model_B': m_b,
            'Regime': reg,
            'Endpoint': ep,
            'Practical_Threshold': thresh,
            'Mean_A': float(np.mean(vals_a)),
            'SD_A': float(np.std(vals_a, ddof=1)),
            'Mean_B': float(np.mean(vals_b)),
            'SD_B': float(np.std(vals_b, ddof=1)),
            'Mean_Diff': mean_d,
            'SD_Diff': sd_d,
            't_stat': float(t_stat),
            'p_raw_t': float(p_t),
            'p_raw_w': float(p_w),
            'Boot_Mean': boot_mean,
            'Boot_CI_95_Lo': ci_lo,
            'Boot_CI_95_Hi': ci_hi,
            'Boot_Excludes_Zero': ci_excludes_0
        })
        
    # Step-down Holm correction
    adj_p = holm_bonferroni(p_vals)
    for i, r in enumerate(primary_results):
        r['p_Holm'] = float(adj_p[i])
        
        # Assign Pre-registered Verdict Label
        # Dual significance: p_Holm < 0.05 AND Boot_Excludes_Zero
        dual_sig = (r['p_Holm'] < 0.05) and r['Boot_Excludes_Zero']
        abs_diff = abs(r['Mean_Diff'])
        is_hypothesized_dir = (r['Mean_Diff'] < 0.0) if r['Endpoint'] == 'RMSE_All' else (r['Mean_Diff'] < 0.0)
        
        if not dual_sig:
            verdict = "NOT SUPPORTED"
        else:
            if abs_diff < r['Practical_Threshold']:
                verdict = "STATISTICALLY DETECTABLE BUT NOT PRACTICALLY RELEVANT"
            else:
                if is_hypothesized_dir:
                    verdict = "SUPPORTED"
                else:
                    verdict = "HARMS"
                    
        r['Dual_Significant'] = dual_sig
        r['Verdict'] = verdict
        
    df_primary_res = pd.DataFrame(primary_results)
    df_primary_res.to_csv('results/raw/phase3_primary_family_statistical_results.csv', index=False)
    print(f"Saved primary family statistical results to results/raw/phase3_primary_family_statistical_results.csv")
    
    # --------------------------------------------------------------------------
    # 2. Hypothesis H3: Integrator Symplecticity Equivalence (TOST Protocol)
    # --------------------------------------------------------------------------
    print("\nEvaluating Hypothesis H3: Integrator Symplecticity Equivalence (TOST)...")
    df_strang = df_raw[(df_raw['Model'] == 'CT_PhysProb_JEPA_v2_SWA') & (df_raw['Regime'] == 'OOD_10x_20x')].sort_values('Seed')
    df_euler  = df_raw[(df_raw['Model'] == 'Euler_Damped') & (df_raw['Regime'] == 'OOD_10x_20x')].sort_values('Seed')
    
    # H3a: RMSE_All equivalence (delta = +/- 0.25)
    diff_rmse = df_strang['RMSE_All'].values - df_euler['RMSE_All'].values
    p_tost_rmse, ci90_rmse, is_eq_rmse = tost_equivalence(diff_rmse, delta=0.25)
    
    # H3b: Signed Bias equivalence (delta = +/- 0.50)
    diff_bias = df_strang['Signed_Bias_ActiveDeg'].values - df_euler['Signed_Bias_ActiveDeg'].values
    p_tost_bias, ci90_bias, is_eq_bias = tost_equivalence(diff_bias, delta=0.50)
    
    h3_records = [
        {
            'Endpoint': 'RMSE_All',
            'Delta_Bound': 0.25,
            'Mean_Diff': float(np.mean(diff_rmse)),
            'SD_Diff': float(np.std(diff_rmse, ddof=1)),
            'CI_90_Lo': float(ci90_rmse[0]),
            'CI_90_Hi': float(ci90_rmse[1]),
            'p_TOST': float(p_tost_rmse),
            'Equivalent': is_eq_rmse,
            'Verdict': 'EQUIVALENCE SUPPORTED' if is_eq_rmse else 'EQUIVALENCE NOT SUPPORTED'
        },
        {
            'Endpoint': 'Signed_Bias_ActiveDeg',
            'Delta_Bound': 0.50,
            'Mean_Diff': float(np.mean(diff_bias)),
            'SD_Diff': float(np.std(diff_bias, ddof=1)),
            'CI_90_Lo': float(ci90_bias[0]),
            'CI_90_Hi': float(ci90_bias[1]),
            'p_TOST': float(p_tost_bias),
            'Equivalent': is_eq_bias,
            'Verdict': 'EQUIVALENCE SUPPORTED' if is_eq_bias else 'EQUIVALENCE NOT SUPPORTED'
        }
    ]
    df_h3_res = pd.DataFrame(h3_records)
    df_h3_res.to_csv('results/raw/phase3_h3_tost_results.csv', index=False)
    print(f"Saved H3 TOST results to results/raw/phase3_h3_tost_results.csv")
    
    # --------------------------------------------------------------------------
    # 3. Hypothesis H4b: Component Ablation Family (10 Tests, Holm at alpha=0.05)
    # --------------------------------------------------------------------------
    print("\nEvaluating Hypothesis H4b: Component Ablations on GRU_D_Enhanced Backbone...")
    ablations = ['H4b_All_Off', 'H4b_No_Mask', 'H4b_No_MixedGap', 'H4b_No_HazardHead', 'H4b_No_SWA']
    df_all_on = df_h4b[df_h4b['Ablation'] == 'H4b_All_On'].sort_values('Seed')
    
    h4b_tests = []
    for ab in ablations:
        h4b_tests.append((ab, 'RMSE_All', 0.50))
    for ab in ablations:
        h4b_tests.append((ab, 'Signed_Bias_ActiveDeg', 1.00))
        
    h4b_results = []
    h4b_pvals = []
    for ab, ep, thresh in h4b_tests:
        df_ab = df_h4b[df_h4b['Ablation'] == ab].sort_values('Seed')
        vals_on = df_all_on[ep].values
        vals_ab = df_ab[ep].values
        diffs = vals_on - vals_ab  # All_On - Ablation
        mean_d = float(np.mean(diffs))
        sd_d = float(np.std(diffs, ddof=1))
        
        t_stat, p_t = stats.ttest_rel(vals_on, vals_ab)
        w_res = stats.wilcoxon(vals_on, vals_ab)
        p_w = float(w_res.pvalue)
        
        # Engine cluster bootstrap for H4b
        meta = np.load(f"results/preds/metadata_OOD_10x_20x.npz")
        y_true = meta['y_true']
        unit_ids = meta['unit_id']
        
        preds_on = np.array([np.load(f"results/preds/h4b_H4b_All_On_seed{s}_pred.npy") for s in df_all_on['Seed']])
        preds_ab = np.array([np.load(f"results/preds/h4b_{ab}_seed{s}_pred.npy") for s in df_ab['Seed']])
        
        metric_fn = compute_rmse if ep == 'RMSE_All' else compute_active_bias
        boot_mean, ci_lo, ci_hi, ci_excludes_0 = run_engine_cluster_bootstrap(
            preds_on, preds_ab, y_true, unit_ids, metric_fn, B=10000, seed=42
        )
        
        h4b_pvals.append(p_t)
        h4b_results.append({
            'Ablation': ab,
            'Endpoint': ep,
            'Practical_Threshold': thresh,
            'Mean_All_On': float(np.mean(vals_on)),
            'SD_All_On': float(np.std(vals_on, ddof=1)),
            'Mean_Ablation': float(np.mean(vals_ab)),
            'SD_Ablation': float(np.std(vals_ab, ddof=1)),
            'Mean_Diff': mean_d,
            'SD_Diff': sd_d,
            't_stat': float(t_stat),
            'p_raw_t': float(p_t),
            'p_raw_w': float(p_w),
            'Boot_Mean': boot_mean,
            'Boot_CI_95_Lo': ci_lo,
            'Boot_CI_95_Hi': ci_hi,
            'Boot_Excludes_Zero': ci_excludes_0
        })
        
    adj_p_h4b = holm_bonferroni(h4b_pvals)
    for i, r in enumerate(h4b_results):
        r['p_Holm'] = float(adj_p_h4b[i])
        dual_sig = (r['p_Holm'] < 0.05) and r['Boot_Excludes_Zero']
        abs_diff = abs(r['Mean_Diff'])
        is_benefit = (r['Mean_Diff'] < 0.0) if r['Endpoint'] == 'RMSE_All' else (r['Mean_Diff'] < 0.0)
        
        if not dual_sig:
            verdict = "NOT SUPPORTED"
        else:
            if abs_diff < r['Practical_Threshold']:
                verdict = "STATISTICALLY DETECTABLE BUT NOT PRACTICALLY RELEVANT"
            else:
                verdict = "COMPONENT BENEFICIAL" if is_benefit else "COMPONENT HARMFUL"
        r['Dual_Significant'] = dual_sig
        r['Verdict'] = verdict
        
    df_h4b_res = pd.DataFrame(h4b_results)
    df_h4b_res.to_csv('results/raw/phase3_h4b_ablation_statistical_results.csv', index=False)
    print(f"Saved H4b statistical results to results/raw/phase3_h4b_ablation_statistical_results.csv")
    
    print("=" * 80)
    print("STATISTICAL ANALYSIS COMPLETE.")
    print("=" * 80)

if __name__ == '__main__':
    run_statistical_analysis()
