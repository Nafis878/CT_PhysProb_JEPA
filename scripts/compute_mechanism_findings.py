import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats

def analyze_mechanism_ablations():
    raw_csv = ROOT / "results" / "raw" / "mechanism_ablation_raw.csv"
    if not raw_csv.exists():
        print(f"File {raw_csv} does not exist yet.")
        return
        
    df = pd.read_csv(raw_csv)
    print("=" * 90)
    print("MECHANISM ABLATION SUMMARY (10 REAL SEEDS, VALIDATION UNITS)")
    print("=" * 90)
    
    regimes = df['Regime'].unique()
    models = df['Model'].unique()
    
    # Summary Table: Mean +/- Std for Total_RMSE and Active_Bias
    summary_rmse = df.groupby(['Regime', 'Model'])['Total_RMSE'].agg(['mean', 'std']).round(2)
    summary_bias = df.groupby(['Regime', 'Model'])['Active_Bias'].agg(['mean', 'std']).round(2)
    
    print("\n--- TOTAL RMSE (Cycles) across Regimes ---")
    piv_rmse = df.pivot_table(index='Regime', columns='Model', values='Total_RMSE', aggfunc='mean').round(2)
    print(piv_rmse.to_string())
    
    print("\n--- ACTIVE DEGRADATION BIAS (Cycles) across Regimes ---")
    piv_bias = df.pivot_table(index='Regime', columns='Model', values='Active_Bias', aggfunc='mean').round(2)
    print(piv_bias.to_string())
    
    # Paired Statistical Tests against Proposed Model (Strang_Symplectic_Damped)
    ref_model = 'Strang_Symplectic_Damped'
    p_values_all = []
    test_records = []
    
    for r in regimes:
        sub = df[df['Regime'] == r]
        ref_vals_rmse = sub[sub['Model'] == ref_model].sort_values('Seed')['Total_RMSE'].values
        ref_vals_bias = sub[sub['Model'] == ref_model].sort_values('Seed')['Active_Bias'].values
        
        for m in models:
            if m == ref_model:
                continue
            m_vals_rmse = sub[sub['Model'] == m].sort_values('Seed')['Total_RMSE'].values
            m_vals_bias = sub[sub['Model'] == m].sort_values('Seed')['Active_Bias'].values
            
            # RMSE test
            diff_rmse = ref_vals_rmse - m_vals_rmse
            mean_d_r = float(np.mean(diff_rmse))
            std_d_r = float(np.std(diff_rmse, ddof=1))
            n = len(diff_rmse)
            se_r = std_d_r / np.sqrt(n)
            ci_r = (mean_d_r - 2.262 * se_r, mean_d_r + 2.262 * se_r)
            t_r, pt_r = stats.ttest_rel(ref_vals_rmse, m_vals_rmse)
            try:
                w_r, pw_r = stats.wilcoxon(diff_rmse)
            except:
                pw_r = 1.0
                
            test_records.append({
                'Regime': r, 'Metric': 'Total_RMSE', 'Competitor': m,
                'Proposed_Mean': float(np.mean(ref_vals_rmse)), 'Competitor_Mean': float(np.mean(m_vals_rmse)),
                'Paired_Diff': mean_d_r, 'CI_95_Low': ci_r[0], 'CI_95_High': ci_r[1],
                'p_t': pt_r, 'p_w': pw_r
            })
            p_values_all.append(pt_r)
            
            # Active bias test (specifically relevant for extreme outages)
            if 'Extreme' in r:
                diff_bias = ref_vals_bias - m_vals_bias
                mean_d_b = float(np.mean(diff_bias))
                std_d_b = float(np.std(diff_bias, ddof=1))
                se_b = std_d_b / np.sqrt(n)
                ci_b = (mean_d_b - 2.262 * se_b, mean_d_b + 2.262 * se_b)
                t_b, pt_b = stats.ttest_rel(ref_vals_bias, m_vals_bias)
                try:
                    w_b, pw_b = stats.wilcoxon(diff_bias)
                except:
                    pw_b = 1.0
                    
                test_records.append({
                    'Regime': r, 'Metric': 'Active_Bias', 'Competitor': m,
                    'Proposed_Mean': float(np.mean(ref_vals_bias)), 'Competitor_Mean': float(np.mean(m_vals_bias)),
                    'Paired_Diff': mean_d_b, 'CI_95_Low': ci_b[0], 'CI_95_High': ci_b[1],
                    'p_t': pt_b, 'p_w': pw_b
                })
                p_values_all.append(pt_b)

    # Multiplicity control: Holm-Bonferroni correction
    df_tests = pd.DataFrame(test_records)
    # Sort by p_t
    df_tests = df_tests.sort_values('p_t').reset_index(drop=True)
    M = len(df_tests)
    df_tests['p_Holm'] = [min(1.0, df_tests.loc[i, 'p_t'] * (M - i)) for i in range(M)]
    # Ensure monotonicity of Holm adjusted p-values
    for i in range(1, M):
        df_tests.loc[i, 'p_Holm'] = max(df_tests.loc[i, 'p_Holm'], df_tests.loc[i-1, 'p_Holm'])
        
    print("\n" + "=" * 90)
    print(f"PAIRED STATISTICAL COMPARISONS vs PROPOSED (Strang_Symplectic_Damped) [Total Comparisons M={M}]")
    print("=" * 90)
    cols = ['Regime', 'Metric', 'Competitor', 'Paired_Diff', 'CI_95_Low', 'CI_95_High', 'p_t', 'p_w', 'p_Holm']
    print(df_tests[cols].round(4).to_string(index=False))
    
    out_csv = ROOT / "results" / "mechanism_ablation_summary.csv"
    df_tests.to_csv(out_csv, index=False)
    print(f"\nSummary table saved to {out_csv}")

if __name__ == '__main__':
    analyze_mechanism_ablations()
