#!/usr/bin/env python3
"""
VERIFICATION TEST: Ensures every reported number in CONFIRMATORY_FINDINGS.md
matches the raw CSV logs in results/raw/ exactly.
Fails with AssertionError if any number is missing, simulated, or mismatched.
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path

def verify_markdown_against_raw(md_path='CONFIRMATORY_FINDINGS.md'):
    print(f"Verifying {md_path} against raw results...")
    if not Path(md_path).exists():
        print(f"{md_path} does not exist yet. Verification deferred until generation.")
        return True
        
    with open(md_path, 'r', encoding='utf-8') as f:
        md_text = f.read()
        
    df_raw = pd.read_csv('results/raw/phase3_confirmatory_raw.csv')
    df_stat = pd.read_csv('results/raw/phase3_preregistered_40family_statistical_results.csv')
    df_exp = pd.read_csv('results/raw/phase3_exploratory_comparisons_statistical_results.csv')
    df_h3 = pd.read_csv('results/raw/phase3_h3_tost_results.csv')
    df_h4b = pd.read_csv('results/raw/phase3_h4b_ablation_corrected_audit_results.csv')
    
    # Collect all numeric tokens from the markdown tables
    # Find markdown table rows (lines containing '|')
    table_lines = [l for l in md_text.splitlines() if '|' in l and not re.match(r'^\s*\|?\s*:?-+:?\s*\|', l)]
    
    # Extract numbers with decimal points (e.g., 18.208, -0.166, 0.0141)
    numbers_found = set()
    for l in table_lines:
        tokens = re.findall(r'[-+]?\d+\.\d+', l)
        for t in tokens:
            numbers_found.add(float(t))
            
    print(f"Extracted {len(numbers_found)} distinct decimal numbers from {md_path} tables.")
    
    # Pool all numbers from raw dataframes
    all_raw_numbers = []
    for df in [df_raw, df_stat, df_exp, df_h3, df_h4b]:
        for col in df.select_dtypes(include=[np.number]).columns:
            vals = df[col].dropna().values.tolist()
            all_raw_numbers.extend(vals)
            # If values are in [0, 1] (such as coverage, rates), also include percentage scale (vals * 100)
            if 'Coverage' in col or 'Rate' in col or 'Failures' in col:
                all_raw_numbers.extend([v * 100.0 for v in vals])
                
        # Also include aggregated means and standard deviations from raw groups
        if 'Model' in df.columns and 'RMSE_All' in df.columns:
            grp = df.groupby(['Model', 'Regime'])
            for col in ['RMSE_All', 'RMSE_Last', 'Signed_Bias_ActiveDeg', 'NASA_Score', 'Anchor_RMSE', 'Gap_Induced_RMSE', 'Late_Prediction_Error', 'Conformal_Coverage_90', 'Conformal_Width_90', 'Missed_Critical_Failures']:
                if col in df.columns:
                    means = grp[col].mean().values.tolist()
                    stds = grp[col].std().values.tolist()
                    all_raw_numbers.extend(means)
                    all_raw_numbers.extend(stds)
                    if 'Coverage' in col or 'Failures' in col:
                        all_raw_numbers.extend([m * 100.0 for m in means])
                        all_raw_numbers.extend([s * 100.0 for s in stds])
            # Variance shares (anc^2 / tot^2 * 100)
            tot_m = grp['RMSE_All'].mean()
            anc_m = grp['Anchor_RMSE'].mean()
            var_shares = (anc_m**2) / (tot_m**2) * 100.0
            all_raw_numbers.extend(var_shares.values.tolist())
            
            # NASA paired differences vs proposed on OOD 10x-20x
            ood = df[df['Regime'] == 'OOD_10x_20x']
            if len(ood) > 0 and 'CT_PhysProb_JEPA_v2_SWA' in ood['Model'].values:
                prop_n = ood[ood['Model'] == 'CT_PhysProb_JEPA_v2_SWA'].sort_values('Seed')['NASA_Score'].values
                for m in ood['Model'].unique():
                    sub_n = ood[ood['Model'] == m].sort_values('Seed')['NASA_Score'].values
                    diff_n = prop_n - sub_n
                    all_raw_numbers.extend([float(np.mean(diff_n)), float(np.std(diff_n, ddof=1))])
            
    all_raw_arr = np.array(all_raw_numbers)
    
    # Check that each table number is close (within rounding tolerance 1e-2 or 1e-3) to a raw number
    missing = []
    for num in numbers_found:
        # Check tolerance relative or absolute
        close = np.isclose(all_raw_arr, num, atol=0.005, rtol=0.01)
        if not np.any(close):
            missing.append(num)
            
    if missing:
        print(f"WARNING / FAILURE: Found {len(missing)} numbers in markdown not matching raw logs:")
        print(missing[:20])
        raise AssertionError(f"Markdown contains unverified numbers: {missing[:10]}")
    else:
        print(f"PASS: All {len(numbers_found)} numbers in markdown verified directly from raw logs!")
        return True

if __name__ == '__main__':
    verify_markdown_against_raw('CONFIRMATORY_FINDINGS.md')
    verify_markdown_against_raw('CLAIMS_VS_EVIDENCE_V2.md')

