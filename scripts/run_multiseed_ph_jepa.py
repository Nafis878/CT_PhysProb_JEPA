import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.run_ood_and_small_data import load_cmapss_subset, build_seq_data, nasa_score
from scripts.run_step1_diagnosis import to_tensor_data
from scripts.run_step2_development import GRUD_TemporalEncoder
from scripts.test_port_hamiltonian_jepa import PortHamiltonian_JEPA
from scripts.run_step2_improvements import GRU_D_v2

def run_multiseed_ph_jepa():
    print("=" * 80)
    print("COMPARING PORT-HAMILTONIAN JEPA vs GRU_D_v2 ACROSS 5 REAL VALIDATION SEEDS")
    print("=" * 80)
    
    df_tr, df_va, _, stats1 = load_cmapss_subset('FD001')
    seeds = [42, 100, 101, 777, 2024]
    
    records = []
    
    for s in seeds:
        print(f"\n--- Testing Seed {s} ---", flush=True)
        torch.manual_seed(s)
        np.random.seed(s)
        
        tr_short = build_seq_data(df_tr, stats1, drop_prob=0.25, seed=s)
        tr_long  = build_seq_data(df_tr, stats1, fixed_stride_range=(3, 15), seed=s + 1000)
        tr_mixed = {
            k: np.concatenate([tr_short[k], tr_long[k]], axis=0)
            for k in ['obs_t', 'ctrl_t', 'dt', 'obs_next', 'y_norm', 'y_raw', 'mem_fast', 'mem_slow']
        }
        
        x_tr = torch.tensor(np.concatenate([tr_mixed['obs_t'], tr_mixed['mem_fast'], tr_mixed['mem_slow']], axis=-1), dtype=torch.float32)
        x_next_tr = torch.tensor(np.concatenate([tr_mixed['obs_next'], tr_mixed['mem_fast'], tr_mixed['mem_slow']], axis=-1), dtype=torch.float32)
        u_tr = torch.tensor(tr_mixed['ctrl_t'], dtype=torch.float32)
        dt_tr = torch.tensor(tr_mixed['dt'], dtype=torch.float32)
        y_tr = torch.tensor(tr_mixed['y_norm'], dtype=torch.float32)
        
        # 1. Train pH-JEPA
        model_ph = PortHamiltonian_JEPA()
        opt_ph = torch.optim.AdamW(model_ph.parameters(), lr=3.5e-3, weight_decay=1e-4)
        N = x_tr.size(0)
        batch_size = 512
        epochs = 8
        lambda_jepa = 0.5
        
        t0 = time.time()
        for ep in range(epochs):
            perm = torch.randperm(N)
            for i in range(0, N, batch_size):
                idx = perm[i:i+batch_size]
                bx, bx_next, bu, bdt, by = x_tr[idx], x_next_tr[idx], u_tr[idx], dt_tr[idx], y_tr[idx]
                y_pred, _, z_pred, z_tgt = model_ph(bx, bu, bdt, x_next=bx_next)
                diff = y_pred - by
                w = torch.where(diff > 0, 2.0, 1.0)
                loss_rul = torch.mean(w * (diff ** 2))
                loss_jepa = F.mse_loss(z_pred, z_tgt)
                loss = loss_rul + lambda_jepa * loss_jepa
                
                opt_ph.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model_ph.parameters(), max_norm=2.0)
                opt_ph.step()
                model_ph.update_target_encoder()
        sec_ph = time.time() - t0
        
        # 2. Train GRU_D_v2 (identical epochs and data)
        model_grud = GRU_D_v2()
        opt_grud = torch.optim.AdamW(model_grud.parameters(), lr=3.5e-3, weight_decay=1e-4)
        t0 = time.time()
        for ep in range(epochs):
            perm = torch.randperm(N)
            for i in range(0, N, batch_size):
                idx = perm[i:i+batch_size]
                bx, bu, bdt, by = x_tr[idx], u_tr[idx], dt_tr[idx], y_tr[idx]
                y_pred, _, _ = model_grud(bx, bu, bdt)
                diff = y_pred - by
                w = torch.where(diff > 0, 2.0, 1.0)
                loss = torch.mean(w * (diff ** 2))
                opt_grud.zero_grad()
                loss.backward()
                opt_grud.step()
        sec_grud = time.time() - t0
        
        # Evaluation on OOD 10x-20x validation
        val_ood = build_seq_data(df_va, stats1, fixed_stride_range=(10, 20), seed=s + 4000)
        vx = torch.tensor(np.concatenate([val_ood['obs_t'], val_ood['mem_fast'], val_ood['mem_slow']], axis=-1), dtype=torch.float32)
        vu = torch.tensor(val_ood['ctrl_t'], dtype=torch.float32)
        vdt = torch.tensor(val_ood['dt'], dtype=torch.float32)
        y_true = val_ood['y_raw']
        dt_cyc = val_ood['dt'][:, 0] * 4.0
        y_curr = np.minimum(125.0, y_true + dt_cyc)
        true_slope = (y_true - y_curr) / np.maximum(1.0, dt_cyc)
        mask_act = (true_slope <= -0.90)
        
        # Evaluate pH-JEPA
        model_ph.eval()
        with torch.no_grad():
            vdt_zero = torch.full_like(vdt, 0.25)
            pa_norm, _, _, _ = model_ph(vx, vu, vdt_zero)
            pa_ph = np.clip(pa_norm[:, 0].numpy() * stats1['tgt_std'] + stats1['tgt_mean'], 0.0, 125.0)
            pt_norm, _, _, _ = model_ph(vx, vu, vdt)
            pt_ph = np.clip(pt_norm[:, 0].numpy() * stats1['tgt_std'] + stats1['tgt_mean'], 0.0, 125.0)
            
        rmse_ph = float(np.sqrt(np.mean((pt_ph - y_true)**2)))
        nasa_ph = nasa_score(pt_ph, y_true)
        bias_ph = float(np.mean(pt_ph[mask_act] - y_true[mask_act]))
        
        # Evaluate GRU_D_v2
        model_grud.eval()
        with torch.no_grad():
            pa_norm, _, _ = model_grud(vx, vu, vdt_zero)
            pa_g = np.clip(pa_norm[:, 0].numpy() * stats1['tgt_std'] + stats1['tgt_mean'], 0.0, 125.0)
            pt_norm, _, _ = model_grud(vx, vu, vdt)
            pt_g = np.clip(pt_norm[:, 0].numpy() * stats1['tgt_std'] + stats1['tgt_mean'], 0.0, 125.0)
            
        rmse_grud = float(np.sqrt(np.mean((pt_g - y_true)**2)))
        nasa_grud = nasa_score(pt_g, y_true)
        bias_grud = float(np.mean(pt_g[mask_act] - y_true[mask_act]))
        
        print(f"Seed {s}: pH-JEPA RMSE={rmse_ph:.2f} (NASA={nasa_ph:.0f}) | GRU_D_v2 RMSE={rmse_grud:.2f} (NASA={nasa_grud:.0f}) | Diff={rmse_ph - rmse_grud:+.2f}")
        records.append({
            'Seed': s,
            'pH_JEPA_RMSE': rmse_ph, 'GRU_D_RMSE': rmse_grud, 'Diff_RMSE': rmse_ph - rmse_grud,
            'pH_JEPA_NASA': nasa_ph, 'GRU_D_NASA': nasa_grud,
            'pH_JEPA_Bias': bias_ph, 'GRU_D_Bias': bias_grud,
            'pH_JEPA_Sec': sec_ph, 'GRU_D_Sec': sec_grud
        })
        
    df_res = pd.DataFrame(records)
    print("\n" + "=" * 80)
    print("5-SEED SUMMARY RESULTS:")
    print(df_res.to_string(index=False))
    
    t_stat, p_val = stats.ttest_rel(df_res['pH_JEPA_RMSE'], df_res['GRU_D_RMSE'])
    print(f"\nMean pH-JEPA RMSE: {df_res['pH_JEPA_RMSE'].mean():.2f} +/- {df_res['pH_JEPA_RMSE'].std():.2f}")
    print(f"Mean GRU_D_v2 RMSE: {df_res['GRU_D_RMSE'].mean():.2f} +/- {df_res['GRU_D_RMSE'].std():.2f}")
    print(f"Paired Mean Diff:   {df_res['Diff_RMSE'].mean():.2f} (t={t_stat:.3f}, p={p_val:.4f})")
    print(f"Mean NASA Score:    pH-JEPA={df_res['pH_JEPA_NASA'].mean():.1f} vs GRU_D={df_res['GRU_D_NASA'].mean():.1f}")
    print("=" * 80)

if __name__ == '__main__':
    run_multiseed_ph_jepa()
