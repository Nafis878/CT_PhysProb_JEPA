import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from scripts.run_ood_and_small_data import load_cmapss_subset, build_seq_data, nasa_score
from scripts.run_step1_diagnosis import to_tensor_data
from scripts.run_step2_development import train_model_swa, CT_PhysProb_JEPA_v2, GRU_D_Enhanced

def test_ensemble_3seeds():
    df_tr, df_va, _, stats1 = load_cmapss_subset('FD001')
    seeds = [42, 100, 101]
    
    preds_jepa = []
    preds_grud = []
    
    # Ground truth on validation set for Seed 42 evaluation
    val_ood = build_seq_data(df_va, stats1, fixed_stride_range=(10, 20), seed=4042)
    vx, vu, vdt, _ = to_tensor_data(val_ood)
    y_true = val_ood['y_raw']
    dt_cyc = val_ood['dt'][:, 0] * 4.0
    y_curr = np.minimum(125.0, y_true + dt_cyc)
    
    for s in seeds:
        print(f"Training Seed {s}...")
        tr_short = build_seq_data(df_tr, stats1, drop_prob=0.25, seed=s)
        tr_long  = build_seq_data(df_tr, stats1, fixed_stride_range=(3, 15), seed=s + 1000)
        tr_mixed = {k: np.concatenate([tr_short[k], tr_long[k]], axis=0) for k in ['obs_t', 'ctrl_t', 'dt', 'obs_next', 'y_norm', 'y_raw', 'mem_fast', 'mem_slow']}
        x_tr, u_tr, dt_tr, y_tr = to_tensor_data(tr_mixed)
        
        m_jepa, _ = train_model_swa(CT_PhysProb_JEPA_v2, dict(), x_tr, u_tr, dt_tr, y_tr, seed=s, epochs=8, use_swa=True)
        m_grud, _ = train_model_swa(GRU_D_Enhanced, dict(), x_tr, u_tr, dt_tr, y_tr, seed=s, epochs=8, use_swa=True)
        
        m_jepa.eval()
        m_grud.eval()
        with torch.no_grad():
            pj_norm, _, _ = m_jepa(vx, vu, vdt)
            pg_norm, _, _ = m_grud(vx, vu, vdt)
            pj = np.clip(pj_norm[:, 0].numpy() * stats1['tgt_std'] + stats1['tgt_mean'], 0.0, 125.0)
            pg = np.clip(pg_norm[:, 0].numpy() * stats1['tgt_std'] + stats1['tgt_mean'], 0.0, 125.0)
            
            preds_jepa.append(pj)
            preds_grud.append(pg)
            
        print(f"  Seed {s} Single Model RMSE: JEPA={np.sqrt(np.mean((pj - y_true)**2)):.2f}, GRUD={np.sqrt(np.mean((pg - y_true)**2)):.2f}")
        
    ens_jepa = np.mean(preds_jepa, axis=0)
    ens_grud = np.mean(preds_grud, axis=0)
    
    rmse_ens_jepa = np.sqrt(np.mean((ens_jepa - y_true)**2))
    rmse_ens_grud = np.sqrt(np.mean((ens_grud - y_true)**2))
    
    print("\n" + "=" * 50)
    print(f"3-Seed Deep Ensemble JEPA Total RMSE: {rmse_ens_jepa:.2f} cycles")
    print(f"3-Seed Deep Ensemble GRUD Total RMSE: {rmse_ens_grud:.2f} cycles")
    print(f"Difference (JEPA - GRUD):             {rmse_ens_jepa - rmse_ens_grud:+.2f} cycles")
    print("=" * 50)

if __name__ == '__main__':
    test_ensemble_3seeds()
