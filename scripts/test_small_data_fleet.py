import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats
import torch

from scripts.run_ood_and_small_data import load_cmapss_subset, build_seq_data
from scripts.run_step1_diagnosis import to_tensor_data
from scripts.run_step2_development import train_model_swa, CT_PhysProb_JEPA_v2
from scripts.run_step2_improvements import GRU_D_v2

def test_small_data_fleet():
    print("=" * 80)
    print("SMALL DATA / FLEET-SCARCITY TEST (15 Training Engines vs Standard 80 Engines)")
    print("Evaluating 5 real seeds on OOD 10x-20x validation")
    print("=" * 80)
    
    df_tr_full, df_va, _, stats1 = load_cmapss_subset('FD001')
    seeds = [42, 100, 101, 777, 2024]
    
    records = []
    for s in seeds:
        # Sample only 15 training units
        np.random.seed(s)
        avail_units = df_tr_full['unit'].unique()
        sampled_units = np.random.choice(avail_units, size=15, replace=False)
        df_tr_small = df_tr_full[df_tr_full['unit'].isin(sampled_units)].copy()
        
        tr_short = build_seq_data(df_tr_small, stats1, drop_prob=0.25, seed=s)
        tr_long  = build_seq_data(df_tr_small, stats1, fixed_stride_range=(3, 15), seed=s + 1000)
        tr_mixed = {k: np.concatenate([tr_short[k], tr_long[k]], axis=0) for k in ['obs_t', 'ctrl_t', 'dt', 'obs_next', 'y_norm', 'y_raw', 'mem_fast', 'mem_slow']}
        x_tr, u_tr, dt_tr, y_tr = to_tensor_data(tr_mixed)
        
        m_jepa, _ = train_model_swa(CT_PhysProb_JEPA_v2, dict(), x_tr, u_tr, dt_tr, y_tr, seed=s, epochs=8, use_swa=True)
        m_grud, _ = train_model_swa(GRU_D_v2, dict(), x_tr, u_tr, dt_tr, y_tr, seed=s, epochs=8, use_swa=False)
        
        val_ood = build_seq_data(df_va, stats1, fixed_stride_range=(10, 20), seed=s + 4000)
        vx, vu, vdt, _ = to_tensor_data(val_ood)
        y_true = val_ood['y_raw']
        
        m_jepa.eval()
        m_grud.eval()
        with torch.no_grad():
            pj_norm, _, _ = m_jepa(vx, vu, vdt)
            pg_norm, _, _ = m_grud(vx, vu, vdt)
            pj = np.clip(pj_norm[:, 0].numpy() * stats1['tgt_std'] + stats1['tgt_mean'], 0.0, 125.0)
            pg = np.clip(pg_norm[:, 0].numpy() * stats1['tgt_std'] + stats1['tgt_mean'], 0.0, 125.0)
            
        rj = float(np.sqrt(np.mean((pj - y_true)**2)))
        rg = float(np.sqrt(np.mean((pg - y_true)**2)))
        print(f"Seed {s}: Small Data 15-Engines JEPA RMSE={rj:.2f} vs GRU_D RMSE={rg:.2f} (Diff: {rj - rg:+.2f})")
        records.append({'Seed': s, 'JEPA_RMSE': rj, 'GRUD_RMSE': rg, 'Diff_RMSE': rj - rg})
        
    df_small = pd.DataFrame(records)
    t_stat, p_val = stats.ttest_rel(df_small['JEPA_RMSE'], df_small['GRUD_RMSE'])
    print("\n" + "=" * 80)
    print(f"Small Data (15 engines) Mean RMSE: JEPA={df_small['JEPA_RMSE'].mean():.2f} +/- {df_small['JEPA_RMSE'].std():.2f} vs GRU_D={df_small['GRUD_RMSE'].mean():.2f} +/- {df_small['GRUD_RMSE'].std():.2f}")
    print(f"Paired Difference: {df_small['Diff_RMSE'].mean():.2f} cycles (t={t_stat:.3f}, p={p_val:.4f})")
    print("=" * 80)

if __name__ == '__main__':
    test_small_data_fleet()
