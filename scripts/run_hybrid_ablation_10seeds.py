#!/usr/bin/env python3
"""
Ablation & Synthesis: CT_PhysProb_JEPA_KinGate
Evaluating whether incorporating Kinematic Countdown (from LOCF) and Continuous Damping (from GRU-D)
allows CT_PhysProb_JEPA to achieve the top performance across all 10 real validation seeds.

NON-NEGOTIABLE PROTOCOL:
- 10 real seeds: [42, 100, 2024, 777, 999, 101, 202, 303, 404, 505].
- Evaluated strictly on 20 validation engines (test set untouched).
- All numbers logged live to results/raw/hybrid_ablation_raw.csv.
"""

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

from scripts.run_ood_and_small_data import load_cmapss_subset, build_seq_data, nasa_score
from scripts.run_step1_diagnosis import to_tensor_data, SEEDS_10, REGIMES
from scripts.run_step2_improvements import GRU_D_v2, PatchTST_RUL_v2, train_model_step2

class CT_PhysProb_JEPA_KinGate(nn.Module):
    """Ablated Hybrid: 2nd-order Hamiltonian ODE + Degradation-Gated Kinematic Countdown + GRU-D Continuous Damping."""
    def __init__(self, in_features=42, u_dim=3, latent_dim=16, alpha=0.10, gamma0=0.15):
        super().__init__()
        self.d = latent_dim // 2
        self.alpha = alpha
        self.gamma0 = gamma0
        
        self.encoder = nn.Sequential(nn.Linear(in_features, 48), nn.Tanh(), nn.Linear(48, latent_dim))
        self.knee_head = nn.Sequential(nn.Linear(latent_dim, 1), nn.Sigmoid())
        self.decay_w = nn.Parameter(torch.tensor([0.25]))
        
        self.W_pot = nn.Parameter(torch.randn(self.d, self.d) * 0.25)
        self.b_pot = nn.Parameter(torch.zeros(self.d))
        self.mode_signs = nn.Parameter(torch.tensor([1.0 if j % 3 != 0 else -0.65 for j in range(self.d)]), requires_grad=False)
        
        cond_in = latent_dim + u_dim + 1
        self.force_net = nn.Sequential(nn.Linear(cond_in, 32), nn.Tanh(), nn.Linear(32, self.d))
        
        self.probe0 = nn.Sequential(nn.Linear(latent_dim + u_dim, 48), nn.Tanh(), nn.Linear(48, 1))
        self.probe_dyn = nn.Sequential(nn.Linear(latent_dim + u_dim + 1, 32), nn.Tanh(), nn.Linear(32, 1))

    def force_tot(self, q, q0, p, u, dt_eff):
        hq = torch.tanh(q @ self.W_pot + self.b_pot) * self.mode_signs
        grad_v_mlp = (hq @ self.W_pot.T) / float(self.d)
        grad_v_conf = 3.5 * self.alpha * (q - q0)
        cond_vec = torch.cat([q0, p, u, dt_eff], dim=-1)
        f_ext = 0.45 * self.force_net(cond_vec)
        return -(grad_v_mlp + grad_v_conf) + f_ext

    def forward(self, x, u, dt):
        z0 = self.encoder(x)
        q0 = z0[:, :self.d]
        p0 = z0[:, self.d:]
        dt_eff = torch.clamp(dt, 0.01, 10.0)
        
        kappa = self.knee_head(z0)
        gamma = self.gamma0 * (1.0 + torch.relu(self.decay_w * dt_eff))
        
        max_dt = float(dt_eff.max().item())
        m_steps = max(4, int(np.ceil(max_dt / 0.0875)))
        sub_dt = dt_eff / float(m_steps)
        
        q, p = q0.clone(), p0.clone()
        decay = torch.exp(-0.5 * gamma * sub_dt)
        for _ in range(m_steps):
            p = decay * p
            p = p + 0.5 * sub_dt * self.force_tot(q, q0, p, u, dt_eff)
            q = q + sub_dt * p
            p = p + 0.5 * sub_dt * self.force_tot(q, q0, p, u, dt_eff)
            p = decay * p
            
        z_pred = torch.cat([q, p], dim=-1)
        y0_norm = self.probe0(torch.cat([z0, u], dim=-1))
        
        dt_norm = (dt_eff * 4.0) / 41.6
        kinematic_count = - kappa * dt_norm
        dyn_corr = 0.25 * self.probe_dyn(torch.cat([z_pred - z0, u, dt_eff], dim=-1))
        
        y_pred = y0_norm + kinematic_count + dyn_corr
        return y_pred, z0, z_pred


def run_hybrid_ablation():
    print("=" * 80)
    print("ABLATION & SYNTHESIS: TESTING CT_PhysProb_JEPA_KinGate ACROSS 10 SEEDS")
    print("=" * 80)
    
    df_tr, df_va, df_te, stats1 = load_cmapss_subset('FD001')
    raw_log_path = ROOT / "results" / "raw" / "hybrid_ablation_raw.csv"
    
    records = []
    models_to_test = [
        ('CT_PhysProb_JEPA_KinGate', CT_PhysProb_JEPA_KinGate, dict(), True),
        ('GRU_D_v2', GRU_D_v2, dict(), False),
        ('PatchTST_RUL_v2', PatchTST_RUL_v2, dict(), False),
    ]

    for s_idx, s in enumerate(SEEDS_10):
        print(f"\n--- Seed {s} ({s_idx + 1}/10) ---")
        tr_short = build_seq_data(df_tr, stats1, drop_prob=0.25, seed=s)
        tr_long  = build_seq_data(df_tr, stats1, fixed_stride_range=(3, 15), seed=s + 1000)
        tr_mixed = {k: np.concatenate([tr_short[k], tr_long[k]], axis=0) for k in ['obs_t', 'ctrl_t', 'dt', 'obs_next', 'y_norm', 'y_raw', 'mem_fast', 'mem_slow']}
        x_tr, u_tr, dt_tr, y_tr = to_tensor_data(tr_mixed)
        
        val_regimes = {
            'InDist_1x_2x': build_seq_data(df_va, stats1, drop_prob=0.25, seed=s + 2000),
            'OOD_5x_10x': build_seq_data(df_va, stats1, fixed_stride_range=(5, 10), seed=s + 3000),
            'OOD_10x_20x': build_seq_data(df_va, stats1, fixed_stride_range=(10, 20), seed=s + 4000),
            'OOD_15x_25x': build_seq_data(df_va, stats1, fixed_stride_range=(15, 25), seed=s + 5000),
        }
        for r_name, vd in val_regimes.items():
            dt_cyc = vd['dt'][:, 0] * 4.0
            vd['y_curr'] = np.minimum(125.0, vd['y_raw'] + dt_cyc)

        for m_name, m_cls, m_kw, has_kin in models_to_test:
            model, tr_sec = train_model_step2(m_cls, m_kw, x_tr, u_tr, dt_tr, y_tr, seed=s, epochs=8, has_kin=has_kin, loss_type='asym', lambda_opt=1.0)
            for r_name in REGIMES:
                vd = val_regimes[r_name]
                vx, vu, vdt, _ = to_tensor_data(vd)
                with torch.no_grad():
                    yp_norm, _, _ = model(vx, vu, vdt)
                    y_pred = torch.clamp(yp_norm[:, 0] * stats1['tgt_std'] + stats1['tgt_mean'], 0.0, 125.0).numpy()
                y_true = vd['y_raw']
                dt_cyc = vd['dt'][:, 0] * 4.0
                true_slope = (y_true - vd['y_curr']) / np.maximum(1.0, dt_cyc)
                mask_act = (true_slope <= -0.90)
                
                records.append({
                    'Seed': s, 'Model': m_name, 'Regime': r_name,
                    'RMSE_All': float(np.sqrt(np.mean((y_pred - y_true)**2))),
                    'MAE_All': float(np.mean(np.abs(y_pred - y_true))),
                    'Signed_Bias_ActiveDeg': float(np.mean(y_pred[mask_act] - y_true[mask_act])) if np.sum(mask_act) > 0 else 0.0,
                    'NASA_Score': nasa_score(y_pred, y_true),
                    'Train_Sec': tr_sec
                })
        print(f"Seed {s} complete.")

    df_res = pd.DataFrame(records)
    df_res.to_csv(raw_log_path, index=False)
    print(f"\nRaw results saved to {raw_log_path}")
    
    # Statistical analysis on primary endpoint
    sub_10 = df_res[df_res['Regime'] == 'OOD_10x_20x']
    v_hybrid = sub_10[sub_10['Model'] == 'CT_PhysProb_JEPA_KinGate'].sort_values('Seed')['RMSE_All'].values
    v_gru    = sub_10[sub_10['Model'] == 'GRU_D_v2'].sort_values('Seed')['RMSE_All'].values
    v_pt     = sub_10[sub_10['Model'] == 'PatchTST_RUL_v2'].sort_values('Seed')['RMSE_All'].values
    
    d_gru = v_hybrid - v_gru
    pt_gru = stats.ttest_rel(v_hybrid, v_gru).pvalue
    pw_gru = stats.wilcoxon(v_hybrid, v_gru).pvalue
    
    d_pt = v_hybrid - v_pt
    pt_pt = stats.ttest_rel(v_hybrid, v_pt).pvalue
    pw_pt = stats.wilcoxon(v_hybrid, v_pt).pvalue
    
    print("\n" + "=" * 80)
    print("PRIMARY ENDPOINT (OOD 10x-20x) HYBRID ABLATION RESULTS (10 REAL SEEDS)")
    print("=" * 80)
    summary_df = sub_10.groupby('Model').agg(
        RMSE_Mean=('RMSE_All', 'mean'),
        RMSE_SD=('RMSE_All', 'std'),
        Signed_Bias_Active=('Signed_Bias_ActiveDeg', 'mean'),
        NASA_Score=('NASA_Score', 'mean')
    ).reset_index().sort_values('RMSE_Mean')
    print(summary_df.to_string(index=False))
    
    print("\nPaired Statistical Tests vs Competitors:")
    print(f"  CT_PhysProb_JEPA_KinGate vs GRU_D_v2: Diff = {np.mean(d_gru):+.2f} cycles, p_t = {pt_gru:.4f}, p_w = {pw_gru:.4f}")
    print(f"  CT_PhysProb_JEPA_KinGate vs PatchTST_RUL_v2: Diff = {np.mean(d_pt):+.2f} cycles, p_t = {pt_pt:.4f}, p_w = {pw_pt:.4f}")

    return df_res, summary_df

if __name__ == '__main__':
    run_hybrid_ablation()
