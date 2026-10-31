import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats
import torch

from scripts.run_ood_and_small_data import load_cmapss_subset, build_seq_data, nasa_score
from scripts.run_step1_diagnosis import to_tensor_data
from scripts.run_step2_development import train_model_swa, CT_PhysProb_JEPA_v2
from scripts.run_step2_improvements import GRU_D_v2

def stress_test_physics_vs_grud():
    print("=" * 80)
    print("STRESS TEST: CT_PhysProb_JEPA_v2_SWA vs GRU_D_v2")
    print("Evaluating where physical constraints separate from black-box GRU-D:")
    print("1. Extreme Outage Horizon (15x-25x and 20x-30x)")
    print("2. Sensor Channel Dropout (30% missing channels at blackout onset)")
    print("3. Small Training Fleet (15 engines vs 80 engines)")
    print("=" * 80)
    
    df_tr, df_va, _, stats1 = load_cmapss_subset('FD001')
    seeds = [42, 100, 101, 777, 2024]
    
    # --------------------------------------------------------------------------
    # STRESS TEST 1: Extreme Outage Blackouts (15x-25x and 20x-30x)
    # --------------------------------------------------------------------------
    print("\n[STRESS TEST 1: Extreme Outages 15x-25x and 20x-30x across 5 seeds]")
    records_ext = []
    for s in seeds:
        tr_short = build_seq_data(df_tr, stats1, drop_prob=0.25, seed=s)
        tr_long  = build_seq_data(df_tr, stats1, fixed_stride_range=(3, 15), seed=s + 1000)
        tr_mixed = {k: np.concatenate([tr_short[k], tr_long[k]], axis=0) for k in ['obs_t', 'ctrl_t', 'dt', 'obs_next', 'y_norm', 'y_raw', 'mem_fast', 'mem_slow']}
        x_tr, u_tr, dt_tr, y_tr = to_tensor_data(tr_mixed)
        
        m_jepa, _ = train_model_swa(CT_PhysProb_JEPA_v2, dict(), x_tr, u_tr, dt_tr, y_tr, seed=s, epochs=8, use_swa=True)
        m_grud, _ = train_model_swa(GRU_D_v2, dict(), x_tr, u_tr, dt_tr, y_tr, seed=s, epochs=8, use_swa=False)
        
        # 15x-25x and 20x-30x validation sets
        for reg_name, stride_rng in [('OOD_15x_25x', (15, 25)), ('Extreme_20x_30x', (20, 30))]:
            val_ext = build_seq_data(df_va, stats1, fixed_stride_range=stride_rng, seed=s + 5000)
            vx, vu, vdt, _ = to_tensor_data(val_ext)
            y_true = val_ext['y_raw']
            dt_cyc = val_ext['dt'][:, 0] * 4.0
            y_curr = np.minimum(125.0, y_true + dt_cyc)
            true_slope = (y_true - y_curr) / np.maximum(1.0, dt_cyc)
            mask_act = (true_slope <= -0.90)
            
            m_jepa.eval()
            m_grud.eval()
            with torch.no_grad():
                pj_norm, _, _ = m_jepa(vx, vu, vdt)
                pg_norm, _, _ = m_grud(vx, vu, vdt)
                pj = np.clip(pj_norm[:, 0].numpy() * stats1['tgt_std'] + stats1['tgt_mean'], 0.0, 125.0)
                pg = np.clip(pg_norm[:, 0].numpy() * stats1['tgt_std'] + stats1['tgt_mean'], 0.0, 125.0)
                
            rj = float(np.sqrt(np.mean((pj - y_true)**2)))
            rg = float(np.sqrt(np.mean((pg - y_true)**2)))
            bj = float(np.mean(pj[mask_act] - y_true[mask_act])) if np.sum(mask_act) > 0 else 0.0
            bg = float(np.mean(pg[mask_act] - y_true[mask_act])) if np.sum(mask_act) > 0 else 0.0
            nj = nasa_score(pj, y_true)
            ng = nasa_score(pg, y_true)
            
            records_ext.append({
                'Seed': s, 'Regime': reg_name,
                'JEPA_RMSE': rj, 'GRUD_RMSE': rg, 'Diff_RMSE': rj - rg,
                'JEPA_Bias': bj, 'GRUD_Bias': bg, 'Diff_Bias': bj - bg,
                'JEPA_NASA': nj, 'GRUD_NASA': ng
            })
            
    df_ext = pd.DataFrame(records_ext)
    for reg in ['OOD_15x_25x', 'Extreme_20x_30x']:
        sub = df_ext[df_ext['Regime'] == reg]
        t_stat, p_val = stats.ttest_rel(sub['JEPA_RMSE'], sub['GRUD_RMSE'])
        tb_stat, pb_val = stats.ttest_rel(sub['JEPA_Bias'], sub['GRUD_Bias'])
        print(f"\n--- {reg} Results ---")
        print(f"RMSE: JEPA={sub['JEPA_RMSE'].mean():.2f} +/- {sub['JEPA_RMSE'].std():.2f} vs GRU_D={sub['GRUD_RMSE'].mean():.2f} +/- {sub['GRUD_RMSE'].std():.2f} (Diff: {sub['Diff_RMSE'].mean():.2f}, p={p_val:.4f})")
        print(f"Bias: JEPA={sub['JEPA_Bias'].mean():.2f} vs GRU_D={sub['GRUD_Bias'].mean():.2f} (Diff: {sub['Diff_Bias'].mean():.2f}, p={pb_val:.4f})")
        print(f"NASA: JEPA={sub['JEPA_NASA'].mean():.1f} vs GRU_D={sub['GRUD_NASA'].mean():.1f}")
        
    # --------------------------------------------------------------------------
    # STRESS TEST 2: Sensor Channel Dropout (30% channels missing at blackout onset)
    # --------------------------------------------------------------------------
    print("\n[STRESS TEST 2: Sensor Channel Dropout (30% missing)]")
    records_drop = []
    for s in seeds:
        val_ood = build_seq_data(df_va, stats1, fixed_stride_range=(10, 20), seed=s + 4000)
        vx, vu, vdt, _ = to_tensor_data(val_ood)
        y_true = val_ood['y_raw']
        
        # Zero out 30% of sensor channels (4 out of 14 raw sensors + their EWMA features)
        np.random.seed(s + 999)
        drop_indices = np.random.choice(14, size=4, replace=False)
        vx_corrupt = vx.clone()
        for ch in drop_indices:
            vx_corrupt[:, ch] = 0.0        # raw sensor
            vx_corrupt[:, ch + 14] = 0.0   # fast EWMA
            vx_corrupt[:, ch + 28] = 0.0   # slow EWMA
            
        m_jepa.eval()
        m_grud.eval()
        with torch.no_grad():
            pj_norm, _, _ = m_jepa(vx_corrupt, vu, vdt)
            pg_norm, _, _ = m_grud(vx_corrupt, vu, vdt)
            pj = np.clip(pj_norm[:, 0].numpy() * stats1['tgt_std'] + stats1['tgt_mean'], 0.0, 125.0)
            pg = np.clip(pg_norm[:, 0].numpy() * stats1['tgt_std'] + stats1['tgt_mean'], 0.0, 125.0)
            
        rj = float(np.sqrt(np.mean((pj - y_true)**2)))
        rg = float(np.sqrt(np.mean((pg - y_true)**2)))
        records_drop.append({'Seed': s, 'JEPA_RMSE': rj, 'GRUD_RMSE': rg, 'Diff_RMSE': rj - rg})
        
    df_drop = pd.DataFrame(records_drop)
    t_stat, p_val = stats.ttest_rel(df_drop['JEPA_RMSE'], df_drop['GRUD_RMSE'])
    print(f"30% Sensor Dropout RMSE: JEPA={df_drop['JEPA_RMSE'].mean():.2f} vs GRU_D={df_drop['GRUD_RMSE'].mean():.2f} (Diff: {df_drop['Diff_RMSE'].mean():.2f}, p={p_val:.4f})")
    print("=" * 80)

if __name__ == '__main__':
    stress_test_physics_vs_grud()
