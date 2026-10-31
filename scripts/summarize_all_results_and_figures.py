import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'figure.dpi': 300
})

def paired_stats(x, y):
    # x = proposed, y = baseline (lower is better for RMSE)
    diff = np.asarray(y) - np.asarray(x)
    t_stat, p_t = stats.ttest_rel(x, y)
    try:
        w_stat, p_w = stats.wilcoxon(x, y)
    except Exception:
        p_w = 1.0
    sd_diff = float(np.std(diff, ddof=1))
    d_cohen = float(np.mean(diff) / sd_diff) if sd_diff > 1e-8 else 0.0
    return float(p_t), float(p_w), d_cohen

def summarize_all():
    df_fd001 = pd.read_csv('results/ress_revision/10Seed_FD001_Benchmark_Raw_Logs.csv')
    df_ood   = pd.read_csv('results/ress_revision/10Seed_OOD_10xGap_Extrapolation.csv')
    df_abl   = pd.read_csv('results/ress_revision/10Seed_FD001_Ablations.csv')
    df_sd    = pd.read_csv('results/ress_revision/10Seed_SmallData_Fleet_Regimes.csv')
    df_stiff = pd.read_csv('results/ress_revision/Stiffness_NFE_WallClock_Benchmark.csv')
    df_fd002 = pd.read_csv('results/ress_revision/5Seed_FD002_MultiCondition_Benchmark.csv')
    df_ds02  = pd.read_csv('results/ress_revision/5Seed_NCMAPSS_DS02_Equalized_Benchmark.csv')
    df_lobo  = pd.read_csv('results/ress_revision/PRONOSTIA_17Bearings_LOBO.csv')
    df_cc    = pd.read_csv('results/ress_revision/Conditional_Coverage_By_RUL_Bin.csv')
    df_ec    = pd.read_csv('results/ress_revision/Conditional_Coverage_Per_Engine.csv')

    print("=== 1. FD001 10-SEED BENCHMARK (RMSE_All / RMSE_Last / NASA_Score) ===")
    for reg in ['MCAR_0', 'MCAR_30', 'MCAR_50', 'MCAR_70', 'MCAR_90', 'Burst_30', 'Burst_50', 'Burst_70', 'MNAR_50', 'MNAR_70']:
        print(f"\n--- Regime: {reg} ---")
        sub = df_fd001[df_fd001['Regime'] == reg]
        prop = sub[sub['Model'] == 'CT_PhysProb_JEPA'].sort_values('Seed')
        for m in sub['Model'].unique():
            m_df = sub[sub['Model'] == m].sort_values('Seed')
            pt, pw, dc = (1.0, 1.0, 0.0) if m == 'CT_PhysProb_JEPA' else paired_stats(prop['RMSE_All'].values, m_df['RMSE_All'].values)
            print(f"  {m:18s}: All={m_df['RMSE_All'].mean():.2f}±{m_df['RMSE_All'].std():.2f} | Last={m_df['RMSE_Last'].mean():.2f}±{m_df['RMSE_Last'].std():.2f} | NASA={m_df['NASA_Score'].mean():.0f}±{m_df['NASA_Score'].std():.0f} | Cov={m_df['Cov95_Post'].mean():.1f}% | p_t={pt:.4f}, p_w={pw:.4f}, d={dc:.2f}")

    print("\n=== 2. ROUTE A: 10-SEED OOD GAP EXTRAPOLATION (FD001) ===")
    for reg in ['InDist_1x_2x', 'OOD_5x_10x', 'OOD_10x_20x', 'OOD_15x_25x']:
        print(f"\n--- OOD Regime: {reg} ---")
        sub = df_ood[df_ood['Regime'] == reg]
        prop = sub[sub['Model'] == 'CT_PhysProb_JEPA'].sort_values('Seed')
        for m in sub['Model'].unique():
            m_df = sub[sub['Model'] == m].sort_values('Seed')
            pt, pw, dc = (1.0, 1.0, 0.0) if m == 'CT_PhysProb_JEPA' else paired_stats(prop['RMSE_All'].values, m_df['RMSE_All'].values)
            print(f"  {m:18s}: All={m_df['RMSE_All'].mean():.2f}±{m_df['RMSE_All'].std():.2f} | Last={m_df['RMSE_Last'].mean():.2f}±{m_df['RMSE_Last'].std():.2f} | Norm={m_df['Latent_Norm'].mean():.2f} | p_t={pt:.2e}, p_w={pw:.4f}, d={dc:.2f}")

    print("\n=== 3. 12-VARIANT ABLATIONS (MCAR_70 vs OOD_10x_20x) ===")
    for reg in ['MCAR_70', 'OOD_10x_20x']:
        print(f"\n--- Ablation Regime: {reg} ---")
        sub = df_abl[df_abl['Regime'] == reg]
        prop = sub[sub['Variant'] == 'Full_CT_PhysProb_JEPA'].sort_values('Seed')
        for v in sub['Variant'].unique():
            v_df = sub[sub['Variant'] == v].sort_values('Seed')
            pt, pw, dc = (1.0, 1.0, 0.0) if v == 'Full_CT_PhysProb_JEPA' else paired_stats(prop['RMSE_All'].values, v_df['RMSE_All'].values)
            print(f"  {v:28s}: All={v_df['RMSE_All'].mean():.2f}±{v_df['RMSE_All'].std():.2f} | Last={v_df['RMSE_Last'].mean():.2f}±{v_df['RMSE_Last'].std():.2f} | Rank={v_df['Effective_Rank'].mean():.2f} | Norm={v_df['Latent_Norm'].mean():.2f} | p_t={pt:.2e}, p_w={pw:.4f}")

    print("\n=== 4. SMALL-DATA FLEET REGIMES (N_train = 8, 16, 40, 80) ===")
    for reg in ['MCAR_50', 'OOD_10x_20x']:
        print(f"\n--- Small-Data Regime: {reg} ---")
        for n_e in [8, 16, 40, 80]:
            sub = df_sd[(df_sd['Regime'] == reg) & (df_sd['N_Train_Engines'] == n_e)]
            prop = sub[sub['Model'] == 'CT_PhysProb_JEPA'].sort_values('Seed')
            line_items = []
            for m in ['CT_PhysProb_JEPA', 'Latent_NODE', 'Neural_CDE', 'mTAN', 'GRU_D', 'Obs_PINN']:
                m_df = sub[sub['Model'] == m].sort_values('Seed')
                line_items.append(f"{m}={m_df['RMSE_All'].mean():.2f}±{m_df['RMSE_All'].std():.2f}")
            print(f"  N={n_e:2d}: " + " | ".join(line_items))

    print("\n=== 5. STIFFNESS, NFE & WALL-CLOCK SOLVER BENCHMARK ===")
    print(df_stiff.to_string(index=False))

    print("\n=== 6. MULTI-DATASET EQUALIZED BENCHMARKS (FD002, N-CMAPSS DS02, PRONOSTIA 17-BEARING LOBO) ===")
    for name, df_m in [('FD002 (5 Seeds)', df_fd002), ('N-CMAPSS DS02 (5 Seeds)', df_ds02)]:
        print(f"\n--- {name} ---")
        for reg in ['MCAR_0', 'MCAR_30', 'MCAR_70', 'OOD_10x_20x']:
            sub = df_m[df_m['Regime'] == reg]
            items = [f"{m}={sub[sub['Model']==m]['RMSE_All'].mean():.2f}±{sub[sub['Model']==m]['RMSE_All'].std():.2f}" for m in ['CT_PhysProb_JEPA', 'Latent_NODE', 'Neural_CDE', 'mTAN', 'TimeAware_JEPA', 'GRU_D', 'ODE_RNN', 'Obs_PINN', 'Vanilla_JEPA']]
            print(f"  {reg:12s}: " + " | ".join(items))

    print("\n--- PRONOSTIA 17-Bearing LOBO (17 Folds, RMSE in seconds & minutes) ---")
    for reg in ['MCAR_0', 'MCAR_30', 'MCAR_70', 'OOD_10x_20x']:
        sub = df_lobo[df_lobo['Regime'] == reg]
        prop = sub[sub['Model'] == 'CT_PhysProb_JEPA'].sort_values('Fold')
        print(f"  Regime {reg}:")
        for m in sub['Model'].unique():
            m_df = sub[sub['Model'] == m].sort_values('Fold')
            pt, pw, dc = (1.0, 1.0, 0.0) if m == 'CT_PhysProb_JEPA' else paired_stats(prop['RMSE_s'].values, m_df['RMSE_s'].values)
            print(f"    {m:18s}: RMSE_s={m_df['RMSE_s'].mean():.1f}±{m_df['RMSE_s'].std():.1f}s ({m_df['RMSE_min'].mean():.2f}±{m_df['RMSE_min'].std():.2f}m) | Cov95={m_df['Cov95_Post'].mean():.1f}% | Width={m_df['Width95_s'].mean():.1f}s | p_t={pt:.4f}, p_w={pw:.4f}")

    print("\n=== 7. CONDITIONAL COVERAGE BY RUL BIN & PER ENGINE ===")
    print(df_cc.groupby(['Regime', 'RUL_Bin'])[['Cov95_Pre', 'Cov95_Post', 'Width95_Post', 'RMSE_Bin', 'Vacuous_Frac_gt62_5']].mean().round(2).to_string())
    print("\n--- Per-Engine Conditional Coverage ---")
    print(df_ec.groupby('Regime')[['Engine_Cov_Mean', 'Engine_Cov_Std', 'Engine_Cov_P10', 'Engine_Cov_Median', 'Engine_Cov_Min', 'Frac_Engines_ge90', 'Frac_Vacuous_gt62_5']].mean().round(2).to_string())

if __name__ == '__main__':
    summarize_all()
