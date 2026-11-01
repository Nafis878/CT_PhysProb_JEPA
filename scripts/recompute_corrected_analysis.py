#!/usr/bin/env python3
"""
MASTER SCRIPT: RECOMPUTE CORRECTED CONFIRMATORY ANALYSIS (STRICTLY FROM RAW LOGS)
Zero re-training. Uses existing results/raw/*.csv and results/preds/*.npy.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def holm_bonferroni(p_vals):
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
    n = len(diffs)
    mean_d = np.mean(diffs)
    sd_d = np.std(diffs, ddof=1)
    se_d = sd_d / np.sqrt(n)
    t1 = (mean_d - (-delta)) / se_d
    p1 = 1.0 - stats.t.cdf(t1, df=n - 1)
    t2 = (mean_d - delta) / se_d
    p2 = stats.t.cdf(t2, df=n - 1)
    p_tost = max(p1, p2)
    t_crit = stats.t.ppf(1.0 - alpha, df=n - 1)
    ci90 = (mean_d - t_crit * se_d, mean_d + t_crit * se_d)
    is_equiv = (ci90[0] > -delta) and (ci90[1] < delta)
    return p_tost, ci90, is_equiv

def compute_rmse(pred, true):
    return np.sqrt(np.mean((pred - true) ** 2))

def compute_active_bias(pred, true):
    mask = (true < 124.9)
    if np.sum(mask) == 0:
        return 0.0
    return np.mean(pred[mask] - true[mask])

def run_both_bootstrap_estimands(preds_a, preds_b, y_true, unit_ids, ep, B=10000, seed=42):
    """
    Vectorized dual bootstrap estimand calculation.
    preds_a, preds_b: shape (30, N_windows)
    y_true: shape (N_windows,)
    """
    np.random.seed(seed)
    unique_units = np.unique(unit_ids)
    n_units = len(unique_units)
    unit_to_indices = {u: np.where(unit_ids == u)[0] for u in unique_units}
    
    mean_pred_a = np.mean(preds_a, axis=0)
    mean_pred_b = np.mean(preds_b, axis=0)
    
    boot_est1 = np.empty(B, dtype=np.float64)
    boot_est2 = np.empty(B, dtype=np.float64)
    
    resampled_units_all = np.random.choice(unique_units, size=(B, n_units), replace=True)
    is_rmse = (ep == 'RMSE_All')
    act_mask_global = (y_true < 124.9)
    
    for b in range(B):
        ru = resampled_units_all[b]
        idx = np.concatenate([unit_to_indices[u] for u in ru])
        
        if is_rmse:
            y_sub = y_true[idx]
            # Est 1: seed-averaged predictions
            val_a_e1 = np.sqrt(np.mean((mean_pred_a[idx] - y_sub) ** 2))
            val_b_e1 = np.sqrt(np.mean((mean_pred_b[idx] - y_sub) ** 2))
            boot_est1[b] = val_a_e1 - val_b_e1
            
            # Est 2: vectorized per-seed evaluation
            err_a_sq = (preds_a[:, idx] - y_sub) ** 2
            err_b_sq = (preds_b[:, idx] - y_sub) ** 2
            rmse_a_seeds = np.sqrt(np.mean(err_a_sq, axis=1))  # shape (30,)
            rmse_b_seeds = np.sqrt(np.mean(err_b_sq, axis=1))  # shape (30,)
            boot_est2[b] = np.mean(rmse_a_seeds - rmse_b_seeds)
        else:
            # Signed Bias Active Degradation
            m_act = act_mask_global[idx]
            if np.sum(m_act) == 0:
                boot_est1[b] = 0.0
                boot_est2[b] = 0.0
            else:
                idx_act = idx[m_act]
                y_act = y_true[idx_act]
                # Est 1
                bias_a_e1 = np.mean(mean_pred_a[idx_act] - y_act)
                bias_b_e1 = np.mean(mean_pred_b[idx_act] - y_act)
                boot_est1[b] = bias_a_e1 - bias_b_e1
                
                # Est 2
                bias_a_seeds = np.mean(preds_a[:, idx_act] - y_act, axis=1)
                bias_b_seeds = np.mean(preds_b[:, idx_act] - y_act, axis=1)
                boot_est2[b] = np.mean(bias_a_seeds - bias_b_seeds)
                
    ci_e1 = (float(np.percentile(boot_est1, 2.5)), float(np.percentile(boot_est1, 97.5)))
    ci_e2 = (float(np.percentile(boot_est2, 2.5)), float(np.percentile(boot_est2, 97.5)))
    mean_e1 = float(np.mean(boot_est1))
    mean_e2 = float(np.mean(boot_est2))
    
    return mean_e1, ci_e1, mean_e2, ci_e2

def main():
    print("=" * 80)
    print("EXECUTING RIGOROUS ANALYSIS AUDIT AND CORRECTION")
    print("=" * 80)
    
    df_raw = pd.read_csv('results/raw/phase3_confirmatory_raw.csv')
    df_h4b = pd.read_csv('results/raw/phase3_h4b_ablation_raw.csv')
    
    # --------------------------------------------------------------------------
    # 1. Full Pre-Registered Primary Hypothesis Family (M = 40 Tests)
    # --------------------------------------------------------------------------
    # Per PREREGISTRATION_V2.md line-by-line:
    # H1: 12 tests = 4 pairs x 3 dropouts (10%, 20%, 30%) on RMSE_All
    # Pairs: Proposed vs NoPhysics, Proposed vs GRU_D_Canonical, NoPhysics vs GRU_D_Canonical, GRU_D_Enhanced vs GRU_D_Canonical
    # H2: 8 tests = 4 pairs x 2 extreme outages (15x-25x, 20x-30x) on Active Signed Bias
    # Same 4 pairs.
    # H4: 18 tests = 3 Recipe models x 3 Original baselines on OOD 10x-20x across RMSE_All (9) and Active Signed Bias (9)
    # H5: 2 tests = Proposed vs NoPhysics on OOD 10x-20x across RMSE_All (1) and Active Signed Bias (1)
    
    primary_specs = []
    
    # H1
    h1_pairs = [
        ('CT_PhysProb_JEPA_v2_SWA', 'CT_PhysProb_JEPA_NoPhysics'),
        ('CT_PhysProb_JEPA_v2_SWA', 'GRU_D_Canonical_SWA'),
        ('CT_PhysProb_JEPA_NoPhysics', 'GRU_D_Canonical_SWA'),
        ('GRU_D_Enhanced_SWA', 'GRU_D_Canonical_SWA')
    ]
    for d in ['Dropout_10pct', 'Dropout_20pct', 'Dropout_30pct']:
        for ma, mb in h1_pairs:
            primary_specs.append(('H1', f"{ma}_vs_{mb}_{d}", ma, mb, d, 'RMSE_All', 0.50))
            
    # H2
    for r in ['OOD_15x_25x', 'OOD_20x_30x']:
        for ma, mb in h1_pairs:
            primary_specs.append(('H2', f"{ma}_vs_{mb}_{r}_Bias", ma, mb, r, 'Signed_Bias_ActiveDeg', 1.00))
            
    # H4
    recipe_models = ['CT_PhysProb_JEPA_NoPhysics', 'GRU_D_Enhanced_SWA', 'CT_PhysProb_JEPA_v2_SWA']
    original_models = ['GRU_D_Original', 'PatchTST_Original', 'Plain_2ndOrder_NODE_Kin_Original']
    for rm in recipe_models:
        for om in original_models:
            primary_specs.append(('H4', f"{rm}_vs_{om}_RMSE", rm, om, 'OOD_10x_20x', 'RMSE_All', 0.50))
    for rm in recipe_models:
        for om in original_models:
            primary_specs.append(('H4', f"{rm}_vs_{om}_Bias", rm, om, 'OOD_10x_20x', 'Signed_Bias_ActiveDeg', 1.00))
            
    # H5
    primary_specs.append(('H5', "CT_PhysProb_JEPA_v2_SWA_vs_CT_PhysProb_JEPA_NoPhysics_RMSE", 'CT_PhysProb_JEPA_v2_SWA', 'CT_PhysProb_JEPA_NoPhysics', 'OOD_10x_20x', 'RMSE_All', 0.50))
    primary_specs.append(('H5', "CT_PhysProb_JEPA_v2_SWA_vs_CT_PhysProb_JEPA_NoPhysics_Bias", 'CT_PhysProb_JEPA_v2_SWA', 'CT_PhysProb_JEPA_NoPhysics', 'OOD_10x_20x', 'Signed_Bias_ActiveDeg', 1.00))
    
    assert len(primary_specs) == 40, f"Expected exactly 40 primary tests, got {len(primary_specs)}"
    print(f"Constructed exact pre-registered Primary Confirmatory Family: M = {len(primary_specs)} tests.")
    
    # --------------------------------------------------------------------------
    # 2. Exploratory Comparisons Suite
    # --------------------------------------------------------------------------
    exploratory_specs = []
    # Proposed vs other models on dropouts
    for d in ['Dropout_10pct', 'Dropout_20pct', 'Dropout_30pct']:
        exploratory_specs.append(('Exploratory_Dropout', f"Proposed_vs_GRU_D_Enhanced_SWA_{d}", 'CT_PhysProb_JEPA_v2_SWA', 'GRU_D_Enhanced_SWA', d, 'RMSE_All', 0.50))
        exploratory_specs.append(('Exploratory_Dropout', f"Proposed_vs_Plain_NODE_Enhanced_SWA_{d}", 'CT_PhysProb_JEPA_v2_SWA', 'Plain_NODE_Enhanced_SWA', d, 'RMSE_All', 0.50))
        exploratory_specs.append(('Exploratory_Dropout', f"Proposed_vs_PatchTST_RUL_v2_{d}", 'CT_PhysProb_JEPA_v2_SWA', 'PatchTST_RUL_v2', d, 'RMSE_All', 0.50))
    # Proposed vs other models on extreme outages
    for r in ['OOD_15x_25x', 'OOD_20x_30x']:
        exploratory_specs.append(('Exploratory_ExtremeOutage', f"Proposed_vs_GRU_D_Enhanced_SWA_{r}_Bias", 'CT_PhysProb_JEPA_v2_SWA', 'GRU_D_Enhanced_SWA', r, 'Signed_Bias_ActiveDeg', 1.00))
        exploratory_specs.append(('Exploratory_ExtremeOutage', f"Proposed_vs_Plain_NODE_Enhanced_SWA_{r}_Bias", 'CT_PhysProb_JEPA_v2_SWA', 'Plain_NODE_Enhanced_SWA', r, 'Signed_Bias_ActiveDeg', 1.00))
        exploratory_specs.append(('Exploratory_ExtremeOutage', f"Proposed_vs_PatchTST_RUL_v2_{r}_Bias", 'CT_PhysProb_JEPA_v2_SWA', 'PatchTST_RUL_v2', r, 'Signed_Bias_ActiveDeg', 1.00))
        
    print(f"Constructed Exploratory Comparisons Suite: {len(exploratory_specs)} tests.")
    
    # --------------------------------------------------------------------------
    # Compute Statistics for Primary Family
    # --------------------------------------------------------------------------
    print("\nComputing Primary Family Statistics with Dual Estimands (B=10,000)...")
    primary_records = []
    raw_pvals = []
    
    for hyp, comp, ma, mb, reg, ep, thresh in primary_specs:
        df_a = df_raw[(df_raw['Model'] == ma) & (df_raw['Regime'] == reg)].sort_values('Seed')
        df_b = df_raw[(df_raw['Model'] == mb) & (df_raw['Regime'] == reg)].sort_values('Seed')
        
        vals_a = df_a[ep].values
        vals_b = df_b[ep].values
        diffs = vals_a - vals_b  # Model A - Model B
        mean_d = float(np.mean(diffs))
        sd_d = float(np.std(diffs, ddof=1))
        se_d = sd_d / np.sqrt(30)
        t_crit = stats.t.ppf(0.975, df=29)
        ci_seed = (float(mean_d - t_crit * se_d), float(mean_d + t_crit * se_d))
        
        t_stat, p_t = stats.ttest_rel(vals_a, vals_b)
        w_res = stats.wilcoxon(vals_a, vals_b)
        p_w = float(w_res.pvalue)
        raw_pvals.append(float(p_t))
        
        # Load predictions for bootstrap
        meta = np.load(f"results/preds/metadata_{reg}.npz")
        y_true = meta['y_true']
        unit_ids = meta['unit_id']
        preds_a = np.array([np.load(f"results/preds/{ma}_{reg}_seed{s}_pred.npy") for s in df_a['Seed']])
        preds_b = np.array([np.load(f"results/preds/{mb}_{reg}_seed{s}_pred.npy") for s in df_b['Seed']])
        
        mean_e1, ci_e1, mean_e2, ci_e2 = run_both_bootstrap_estimands(
            preds_a, preds_b, y_true, unit_ids, ep, B=10000, seed=42
        )
        
        primary_records.append({
            'Hypothesis': hyp,
            'Comparison': comp,
            'Model_A': ma,
            'Model_B': mb,
            'Regime': reg,
            'Endpoint': ep,
            'Practical_Threshold': thresh,
            'Mean_A': float(np.mean(vals_a)),
            'SD_A': float(np.std(vals_a, ddof=1)),
            'Mean_B': float(np.mean(vals_b)),
            'SD_B': float(np.std(vals_b, ddof=1)),
            'Mean_Diff': mean_d,
            'SD_Diff': sd_d,
            'Seed_CI_95_Lo': ci_seed[0],
            'Seed_CI_95_Hi': ci_seed[1],
            't_stat': float(t_stat),
            'p_raw_t': float(p_t),
            'p_raw_w': p_w,
            'Boot_E1_Mean': mean_e1,
            'Boot_E1_CI_95_Lo': ci_e1[0],
            'Boot_E1_CI_95_Hi': ci_e1[1],
            'Boot_E1_Excludes_Zero': (ci_e1[0] > 0.0 or ci_e1[1] < 0.0),
            'Boot_E2_Mean': mean_e2,
            'Boot_E2_CI_95_Lo': ci_e2[0],
            'Boot_E2_CI_95_Hi': ci_e2[1],
            'Boot_E2_Excludes_Zero': (ci_e2[0] > 0.0 or ci_e2[1] < 0.0)
        })
        
    adj_p = holm_bonferroni(raw_pvals)
    for i, r in enumerate(primary_records):
        r['p_Holm'] = float(adj_p[i])
        
        # Verdict under Estimand 1 (Pre-Registered)
        dual_sig_e1 = (r['p_Holm'] < 0.05) and r['Boot_E1_Excludes_Zero']
        abs_diff = abs(r['Mean_Diff'])
        is_benefit = (r['Mean_Diff'] < 0.0)
        
        if not dual_sig_e1:
            verd_e1 = "NOT SUPPORTED"
        else:
            if abs_diff < r['Practical_Threshold']:
                verd_e1 = "STATISTICALLY DETECTABLE BUT NOT PRACTICALLY RELEVANT"
            else:
                verd_e1 = "SUPPORTED" if is_benefit else "HARMS"
                
        # Verdict under Estimand 2 (Sensitivity)
        dual_sig_e2 = (r['p_Holm'] < 0.05) and r['Boot_E2_Excludes_Zero']
        if not dual_sig_e2:
            verd_e2 = "NOT SUPPORTED"
        else:
            if abs_diff < r['Practical_Threshold']:
                verd_e2 = "STATISTICALLY DETECTABLE BUT NOT PRACTICALLY RELEVANT"
            else:
                verd_e2 = "SUPPORTED" if is_benefit else "HARMS"
                
        r['Verdict_E1_PreReg'] = verd_e1
        r['Verdict_E2_Sensitivity'] = verd_e2
        r['Verdict_Changed'] = (verd_e1 != verd_e2)
        
    df_prim_res = pd.DataFrame(primary_records)
    df_prim_res.to_csv('results/raw/phase3_preregistered_40family_statistical_results.csv', index=False)
    print("Saved results/raw/phase3_preregistered_40family_statistical_results.csv")
    
    # Check verdict shifts
    shifts = df_prim_res[df_prim_res['Verdict_Changed']]
    print(f"Number of verdict shifts between Estimand 1 and Estimand 2: {len(shifts)}")
    if len(shifts) > 0:
        print(shifts[['Comparison', 'Verdict_E1_PreReg', 'Verdict_E2_Sensitivity']])
        
    # --------------------------------------------------------------------------
    # Compute Statistics for Exploratory Comparisons Suite
    # --------------------------------------------------------------------------
    print("\nComputing Exploratory Comparisons Statistics...")
    exploratory_records = []
    exp_pvals = []
    for grp, comp, ma, mb, reg, ep, thresh in exploratory_specs:
        df_a = df_raw[(df_raw['Model'] == ma) & (df_raw['Regime'] == reg)].sort_values('Seed')
        df_b = df_raw[(df_raw['Model'] == mb) & (df_raw['Regime'] == reg)].sort_values('Seed')
        
        vals_a = df_a[ep].values
        vals_b = df_b[ep].values
        diffs = vals_a - vals_b
        mean_d = float(np.mean(diffs))
        sd_d = float(np.std(diffs, ddof=1))
        se_d = sd_d / np.sqrt(30)
        t_crit = stats.t.ppf(0.975, df=29)
        ci_seed = (float(mean_d - t_crit * se_d), float(mean_d + t_crit * se_d))
        
        t_stat, p_t = stats.ttest_rel(vals_a, vals_b)
        w_res = stats.wilcoxon(vals_a, vals_b)
        exp_pvals.append(float(p_t))
        
        meta = np.load(f"results/preds/metadata_{reg}.npz")
        y_true = meta['y_true']
        unit_ids = meta['unit_id']
        preds_a = np.array([np.load(f"results/preds/{ma}_{reg}_seed{s}_pred.npy") for s in df_a['Seed']])
        preds_b = np.array([np.load(f"results/preds/{mb}_{reg}_seed{s}_pred.npy") for s in df_b['Seed']])
        
        mean_e1, ci_e1, mean_e2, ci_e2 = run_both_bootstrap_estimands(
            preds_a, preds_b, y_true, unit_ids, ep, B=10000, seed=42
        )
        
        exploratory_records.append({
            'Group': grp,
            'Comparison': comp,
            'Model_A': ma,
            'Model_B': mb,
            'Regime': reg,
            'Endpoint': ep,
            'Practical_Threshold': thresh,
            'Mean_A': float(np.mean(vals_a)),
            'SD_A': float(np.std(vals_a, ddof=1)),
            'Mean_B': float(np.mean(vals_b)),
            'SD_B': float(np.std(vals_b, ddof=1)),
            'Mean_Diff': mean_d,
            'SD_Diff': sd_d,
            'Seed_CI_95_Lo': ci_seed[0],
            'Seed_CI_95_Hi': ci_seed[1],
            't_stat': float(t_stat),
            'p_raw_t': float(p_t),
            'p_raw_w': float(w_res.pvalue),
            'Boot_E1_Mean': mean_e1,
            'Boot_E1_CI_95_Lo': ci_e1[0],
            'Boot_E1_CI_95_Hi': ci_e1[1],
            'Boot_E2_Mean': mean_e2,
            'Boot_E2_CI_95_Lo': ci_e2[0],
            'Boot_E2_CI_95_Hi': ci_e2[1],
        })
        
    adj_p_exp = holm_bonferroni(exp_pvals)
    for i, r in enumerate(exploratory_records):
        r['p_Holm'] = float(adj_p_exp[i])
        
    df_exp_res = pd.DataFrame(exploratory_records)
    df_exp_res.to_csv('results/raw/phase3_exploratory_comparisons_statistical_results.csv', index=False)
    print("Saved results/raw/phase3_exploratory_comparisons_statistical_results.csv")
    
    # --------------------------------------------------------------------------
    # 3. Full H4b Audit Table Calculation
    # --------------------------------------------------------------------------
    print("\nComputing Full H4b Audit Table...")
    df_on = df_h4b[df_h4b['Ablation'] == 'H4b_All_On'].sort_values('Seed')
    ablations = ['H4b_All_Off', 'H4b_No_Mask', 'H4b_No_MixedGap', 'H4b_No_HazardHead', 'H4b_No_SWA']
    
    meta_ood = np.load('results/preds/metadata_OOD_10x_20x.npz')
    y_true_ood = meta_ood['y_true']
    unit_ids_ood = meta_ood['unit_id']
    
    h4b_audit_records = []
    h4b_pvals = []
    
    for ab in ablations:
        for ep, thresh in [('RMSE_All', 0.50), ('Signed_Bias_ActiveDeg', 1.00)]:
            df_ab = df_h4b[df_h4b['Ablation'] == ab].sort_values('Seed')
            vals_on = df_on[ep].values
            vals_ab = df_ab[ep].values
            diffs = vals_on - vals_ab  # Sign convention: All_On minus Ablated
            mean_d = float(np.mean(diffs))
            sd_d = float(np.std(diffs, ddof=1))
            se_d = sd_d / np.sqrt(30)
            t_crit = stats.t.ppf(0.975, df=29)
            ci_seed = (float(mean_d - t_crit * se_d), float(mean_d + t_crit * se_d))
            
            t_stat, p_t = stats.ttest_rel(vals_on, vals_ab)
            w_res = stats.wilcoxon(vals_on, vals_ab)
            h4b_pvals.append(float(p_t))
            
            preds_on = np.array([np.load(f"results/preds/h4b_H4b_All_On_seed{s}_pred.npy") for s in df_on['Seed']])
            preds_ab = np.array([np.load(f"results/preds/h4b_{ab}_seed{s}_pred.npy") for s in df_ab['Seed']])
            
            mean_e1, ci_e1, mean_e2, ci_e2 = run_both_bootstrap_estimands(
                preds_on, preds_ab, y_true_ood, unit_ids_ood, ep, B=10000, seed=42
            )
            
            h4b_audit_records.append({
                'Ablation': ab,
                'Endpoint': ep,
                'Practical_Threshold': thresh,
                'Mean_All_On': float(np.mean(vals_on)),
                'SD_All_On': float(np.std(vals_on, ddof=1)),
                'Mean_Ablation': float(np.mean(vals_ab)),
                'SD_Ablation': float(np.std(vals_ab, ddof=1)),
                'Mean_Diff': mean_d,
                'SD_Diff': sd_d,
                'Seed_CI_95_Lo': ci_seed[0],
                'Seed_CI_95_Hi': ci_seed[1],
                't_stat': float(t_stat),
                'p_raw_t': float(p_t),
                'p_raw_w': float(w_res.pvalue),
                'Boot_E1_Mean': mean_e1,
                'Boot_E1_CI_95_Lo': ci_e1[0],
                'Boot_E1_CI_95_Hi': ci_e1[1],
                'Boot_E1_Excludes_Zero': (ci_e1[0] > 0.0 or ci_e1[1] < 0.0),
                'Boot_E2_Mean': mean_e2,
                'Boot_E2_CI_95_Lo': ci_e2[0],
                'Boot_E2_CI_95_Hi': ci_e2[1],
                'Boot_E2_Excludes_Zero': (ci_e2[0] > 0.0 or ci_e2[1] < 0.0),
            })
            
    adj_p_h4b = holm_bonferroni(h4b_pvals)
    for i, r in enumerate(h4b_audit_records):
        r['p_Holm'] = float(adj_p_h4b[i])
        dual_sig_e1 = (r['p_Holm'] < 0.05) and r['Boot_E1_Excludes_Zero']
        abs_diff = abs(r['Mean_Diff'])
        is_benefit = (r['Mean_Diff'] < 0.0)
        
        if not dual_sig_e1:
            verd_e1 = "NOT SUPPORTED"
        else:
            if abs_diff < r['Practical_Threshold']:
                verd_e1 = "STATISTICALLY DETECTABLE BUT NOT PRACTICALLY RELEVANT"
            else:
                verd_e1 = "COMPONENT BENEFICIAL" if is_benefit else "COMPONENT HARMFUL"
                
        dual_sig_e2 = (r['p_Holm'] < 0.05) and r['Boot_E2_Excludes_Zero']
        if not dual_sig_e2:
            verd_e2 = "NOT SUPPORTED"
        else:
            if abs_diff < r['Practical_Threshold']:
                verd_e2 = "STATISTICALLY DETECTABLE BUT NOT PRACTICALLY RELEVANT"
            else:
                verd_e2 = "COMPONENT BENEFICIAL" if is_benefit else "COMPONENT HARMFUL"
                
        r['Verdict_E1_PreReg'] = verd_e1
        r['Verdict_E2_Sensitivity'] = verd_e2
        r['Verdict_Changed'] = (verd_e1 != verd_e2)
        
    df_h4b_audit = pd.DataFrame(h4b_audit_records)
    df_h4b_audit.to_csv('results/raw/phase3_h4b_ablation_corrected_audit_results.csv', index=False)
    print("Saved results/raw/phase3_h4b_ablation_corrected_audit_results.csv")
    
    print("=" * 80)
    print("RECOMPUTATION COMPLETE SUCCESSFULLY.")
    print("=" * 80)

if __name__ == '__main__':
    main()
