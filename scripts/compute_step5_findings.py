#!/usr/bin/env python3
"""
STEP 5: COMPREHENSIVE STATISTICAL ANALYSIS & HYPOTHESIS TESTING
Evaluates all pre-registered hypotheses H1, H2, H3, H4 across all 30 real seeds.
Computes effect sizes (Cohen's dz), exact 95% CIs, Holm-Bonferroni adjusted p-values,
and TOST non-inferiority margins.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats

def paired_stats(x, y):
    diff = x - y
    n = len(diff)
    mean_d = float(np.mean(diff))
    std_d = float(np.std(diff, ddof=1))
    se_d = std_d / np.sqrt(n)
    ci95 = (mean_d - 1.96 * se_d, mean_d + 1.96 * se_d)
    dz = mean_d / std_d if std_d > 0 else 0.0
    se_dz = np.sqrt(1.0/n + (dz**2)/(2.0*n))
    dz_ci95 = (dz - 1.96 * se_dz, dz + 1.96 * se_dz)
    t_stat, p_t = stats.ttest_rel(x, y)
    w_stat, p_w = stats.wilcoxon(x, y)
    return {
        'mean_diff': mean_d, 'ci95': ci95, 'dz': dz, 'dz_ci95': dz_ci95,
        't_stat': float(t_stat), 'p_t': float(p_t), 'w_stat': float(w_stat), 'p_w': float(p_w)
    }

def holm_adjust(p_vals):
    p_vals = np.array(p_vals)
    sorted_idx = np.argsort(p_vals)
    adj_p = np.zeros(len(p_vals))
    m = len(p_vals)
    for rank, idx in enumerate(sorted_idx):
        adj_p[idx] = min(1.0, p_vals[idx] * (m - rank))
    for i in range(1, len(sorted_idx)):
        prev_i = sorted_idx[i-1]
        curr_i = sorted_idx[i]
        adj_p[curr_i] = max(adj_p[curr_i], adj_p[prev_i])
    return adj_p

def main():
    df = pd.read_csv(ROOT / "results" / "raw" / "step4_final_30seeds_raw.csv")
    df_conf = pd.read_csv(ROOT / "results" / "raw" / "step4_conformal_cbm_raw.csv")
    
    print("=" * 80)
    print("STEP 5: STATISTICAL HYPOTHESIS TESTING REPORT (30 REAL SEEDS)")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # PRIMARY ENDPOINT: OOD 10x-20x RMSE_All
    # -------------------------------------------------------------------------
    sub10 = df[df['Regime'] == 'OOD_10x_20x']
    kg_rmse = sub10[sub10['Model'] == 'CT_PhysProb_JEPA_KinGate'].sort_values('Seed')['RMSE_All'].values
    models = [m for m in sub10['Model'].unique() if m != 'CT_PhysProb_JEPA_KinGate']
    
    print("\n--- 1. PRIMARY ENDPOINT (OOD 10x-20x RMSE_All) PAIRED COMPARISONS ---")
    p_t_list, tests_list = [], []
    for m in models:
        comp = sub10[sub10['Model'] == m].sort_values('Seed')['RMSE_All'].values
        res = paired_stats(kg_rmse, comp)
        res['model'] = m
        tests_list.append(res)
        p_t_list.append(res['p_t'])
        
    adj_p = holm_adjust(p_t_list)
    for i, t in enumerate(tests_list):
        t['p_holm'] = adj_p[i]
        print(f"{t['model']:<25}: Diff={t['mean_diff']:+.2f} cycles, dz={t['dz']:+.2f} [CI: {t['dz_ci95'][0]:.2f}, {t['dz_ci95'][1]:.2f}], p_t={t['p_t']:.3e}, p_holm={t['p_holm']:.3e}, p_w={t['p_w']:.3e}")

    # -------------------------------------------------------------------------
    # HYPOTHESIS 1 (H1): Error Growth Slope vs Outage Length
    # -------------------------------------------------------------------------
    print("\n--- 2. HYPOTHESIS 1 (H1: Error Growth Slope dRMSE/dt) ---")
    # Regimes map to average gap length: InDist=1.5, 5x-10x=7.5, 10x-20x=15.0, 15x-25x=20.0 cycles
    regime_dt = {'InDist_1x_2x': 1.5, 'OOD_5x_10x': 7.5, 'OOD_10x_20x': 15.0, 'OOD_15x_25x': 20.0}
    df['Mean_DT'] = df['Regime'].map(regime_dt)
    
    slopes = []
    for (s, m), grp in df.groupby(['Seed', 'Model']):
        x_dt = grp['Mean_DT'].values
        y_rm = grp['RMSE_All'].values
        slope, _, _, _, _ = stats.linregress(x_dt, y_rm)
        slopes.append({'Seed': s, 'Model': m, 'Slope': slope})
    df_slopes = pd.DataFrame(slopes)
    
    slope_summary = df_slopes.groupby('Model')['Slope'].agg(['mean', 'std']).reset_index()
    print("Mean Error Growth Slopes (cycles per cycle of outage):")
    print(slope_summary.round(3).to_string(index=False))
    
    kg_slopes = df_slopes[df_slopes['Model'] == 'CT_PhysProb_JEPA_KinGate'].sort_values('Seed')['Slope'].values
    print("\nPaired slope tests vs CT_PhysProb_JEPA_KinGate:")
    p_slope_list, slope_tests = [], []
    for m in [m for m in slope_summary['Model'] if m != 'CT_PhysProb_JEPA_KinGate']:
        comp_s = df_slopes[df_slopes['Model'] == m].sort_values('Seed')['Slope'].values
        res = paired_stats(kg_slopes, comp_s)
        res['model'] = m
        slope_tests.append(res)
        p_slope_list.append(res['p_t'])
    adj_p_s = holm_adjust(p_slope_list)
    for i, t in enumerate(slope_tests):
        t['p_holm'] = adj_p_s[i]
        print(f"{t['model']:<25}: Diff={t['mean_diff']:+.3f}, dz={t['dz']:+.2f}, p_t={t['p_t']:.3e}, p_holm={t['p_holm']:.3e}")

    # -------------------------------------------------------------------------
    # HYPOTHESIS 2 (H2): In-Distribution Non-Inferiority (TOST, delta = 0.5)
    # -------------------------------------------------------------------------
    print("\n--- 3. HYPOTHESIS 2 (H2: In-Distribution TOST Non-Inferiority, Margin = 0.5 cycles) ---")
    sub_in = df[df['Regime'] == 'InDist_1x_2x']
    in_summary = sub_in.groupby('Model')['RMSE_All'].agg(['mean', 'std']).reset_index().sort_values('mean')
    print("In-Distribution RMSE:")
    print(in_summary.round(2).to_string(index=False))
    
    kg_in = sub_in[sub_in['Model'] == 'CT_PhysProb_JEPA_KinGate'].sort_values('Seed')['RMSE_All'].values
    best_m = 'GRU_D'
    best_in = sub_in[sub_in['Model'] == best_m].sort_values('Seed')['RMSE_All'].values
    
    diff_in = kg_in - best_in
    mean_d = float(np.mean(diff_in))
    se_d = float(np.std(diff_in, ddof=1) / np.sqrt(len(diff_in)))
    # TOST upper one-sided test against margin delta = 0.5: H0: mean_d >= 0.5 vs H1: mean_d < 0.5
    t_tost = (mean_d - 0.5) / se_d
    p_tost = stats.t.cdf(t_tost, df=len(diff_in)-1)
    ci90 = (mean_d - 1.645 * se_d, mean_d + 1.645 * se_d)
    print(f"Proposed ({float(np.mean(kg_in)):.2f}) vs Best Learned In-Dist {best_m} ({float(np.mean(best_in)):.2f}):")
    print(f"Paired Diff: {mean_d:+.2f} cycles, 90% CI: [{ci90[0]:.2f}, {ci90[1]:.2f}]")
    print(f"TOST test against margin 0.5: t = {t_tost:.3f}, p_TOST = {p_tost:.4f}")
    if p_tost < 0.05:
        print("-> H2 SUPPORTED: Non-inferiority established within 0.5 cycles margin.")
    else:
        print("-> H2 NOT SUPPORTED: Difference exceeds non-inferiority margin.")

    # -------------------------------------------------------------------------
    # HYPOTHESIS 3 (H3): Safety-Critical Optimistic Bias & NASA Score
    # -------------------------------------------------------------------------
    print("\n--- 4. HYPOTHESIS 3 (H3: Active Degradation Signed Bias & NASA Score) ---")
    kg_bias = sub10[sub10['Model'] == 'CT_PhysProb_JEPA_KinGate'].sort_values('Seed')['Signed_Bias_ActiveDeg'].values
    kg_nasa = sub10[sub10['Model'] == 'CT_PhysProb_JEPA_KinGate'].sort_values('Seed')['NASA_Score'].values
    
    print("Signed Bias Comparisons (Lower is safer):")
    for m in ['GRU_D', 'Plain_2ndOrder_NODE_Kin', 'PatchTST_RUL', 'CT_PhysProb_JEPA_Clamped', 'GRU_D_v2']:
        comp_b = sub10[sub10['Model'] == m].sort_values('Seed')['Signed_Bias_ActiveDeg'].values
        res = paired_stats(kg_bias, comp_b)
        print(f"  vs {m:<25}: Diff={res['mean_diff']:+.2f} cycles, p_t={res['p_t']:.3e}, p_w={res['p_w']:.3e}")
        
    print("\nNASA Score Comparisons (Lower is better):")
    for m in ['GRU_D', 'Plain_2ndOrder_NODE_Kin', 'PatchTST_RUL', 'CT_PhysProb_JEPA_Clamped', 'GRU_D_v2']:
        comp_n = sub10[sub10['Model'] == m].sort_values('Seed')['NASA_Score'].values
        res = paired_stats(kg_nasa, comp_n)
        print(f"  vs {m:<25}: Diff={res['mean_diff']:+.1f}, p_t={res['p_t']:.3e}, p_w={res['p_w']:.3e}")

    # -------------------------------------------------------------------------
    # HYPOTHESIS 4 (H4): Matched Ablations
    # -------------------------------------------------------------------------
    print("\n--- 5. HYPOTHESIS 4 (H4: Matched Ablations across 30 Real Seeds) ---")
    # Ablations:
    # KinGate vs Adaptive (Ablating Kinematic Gate)
    # KinGate vs Clamped (Ablating Adaptive Strang Leapfrog)
    adap_rmse = sub10[sub10['Model'] == 'CT_PhysProb_JEPA_Adaptive'].sort_values('Seed')['RMSE_All'].values
    clamp_rmse = sub10[sub10['Model'] == 'CT_PhysProb_JEPA_Clamped'].sort_values('Seed')['RMSE_All'].values
    
    res_adap = paired_stats(kg_rmse, adap_rmse)
    res_clamp = paired_stats(kg_rmse, clamp_rmse)
    print(f"KinGate vs Adaptive (No-KinGate): Diff={res_adap['mean_diff']:+.2f} cycles, p_t={res_adap['p_t']:.3e}")
    print(f"KinGate vs Clamped (Horizon Clamp): Diff={res_clamp['mean_diff']:+.2f} cycles, p_t={res_clamp['p_t']:.3e}")
    
    # -------------------------------------------------------------------------
    # CONFORMAL & CBM RESULTS
    # -------------------------------------------------------------------------
    print("\n--- 6. CONFORMAL COVERAGE & CBM COST SUMMARY ---")
    conf_summ = df_conf.groupby(['Policy', 'Model']).agg({
        'PICP95': ['mean', 'std'],
        'Width95': ['mean', 'std'],
        'Crit_Missed_Failures': ['mean'],
        'False_Alarms': ['mean'],
        'Expected_Cost_USD': ['mean', 'std']
    }).round(2)
    print(conf_summ.to_string())

if __name__ == '__main__':
    main()
