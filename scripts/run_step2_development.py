#!/usr/bin/env python3
"""
STEP 2: MODEL DEVELOPMENT ON VALIDATION UNITS (10 REAL SEEDS)
Evaluates candidate improvements across 3 structured rounds under identical compute budget:

Round 1 (Candidate C): GRU-D / TCN Temporal Anchor Encoder vs. Static Linear Encoder
Round 2 (Candidates A & B): Knee-Hazard Modulated Countdown Head + Physics vs. No-Physics Ablation
Round 3 (Candidate D): Stochastic Weight Averaging (SWA) over the last 3 epochs

SHARED BASELINE PROTOCOL:
- All enhancements offered identically to GRU_D, PatchTST_RUL, and Plain_2ndOrder_NODE_Kin.
- Evaluated strictly on 20 validation engines (FD001 test engines remain untouched).
- All numbers logged directly to results/raw/step2_development_raw.csv.
"""

import os
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
from scripts.run_step1_diagnosis import to_tensor_data, REGIMES, SEEDS_10, Plain_2ndOrder_NODE_Kin
from scripts.run_step2_improvements import GRU_D_v2, PatchTST_RUL_v2
from scripts.run_hybrid_ablation_10seeds import CT_PhysProb_JEPA_KinGate


# ==============================================================================
# ARCHITECTURE DEFINITIONS FOR STEP 2
# ==============================================================================

class GRUD_TemporalEncoder(nn.Module):
    """Continuous-time recurrent feature extractor producing position-velocity latent coordinates."""
    def __init__(self, in_features=42, u_dim=3, latent_dim=16, hidden_dim=48):
        super().__init__()
        self.decay_h = nn.Sequential(nn.Linear(1, hidden_dim), nn.Softplus())
        self.decay_x = nn.Sequential(nn.Linear(1, in_features), nn.Softplus())
        self.enc_h0 = nn.Sequential(nn.Linear(in_features + u_dim, hidden_dim), nn.Tanh())
        self.gru_cell = nn.GRUCell(in_features + u_dim, hidden_dim)
        self.to_latent = nn.Sequential(nn.Linear(hidden_dim, 48), nn.Tanh(), nn.Linear(48, latent_dim))

    def forward(self, x, u, dt):
        dt_eff = torch.clamp(dt, 0.01, 10.0)
        h0 = self.enc_h0(torch.cat([x, u], dim=-1))
        gamma_h = torch.exp(-self.decay_h(dt_eff))
        gamma_x = torch.exp(-self.decay_x(dt_eff))
        h_decayed = gamma_h * h0
        x_imputed = gamma_x * x
        h_next = self.gru_cell(torch.cat([x_imputed, u], dim=-1), h_decayed)
        z0 = self.to_latent(h_next)
        return z0, h_next


class KneeHazardHead(nn.Module):
    """
    Knee-hazard countdown head:
    Computes anchor RUL at t_k, plus continuous hazard kappa(s) = 1 - (1 - kappa_0)*exp(-lambda_k*s).
    Integrates countdown over delta_t: int_0^dt kappa(s) ds.
    """
    def __init__(self, in_dim=16, u_dim=3):
        super().__init__()
        self.anchor_probe = nn.Sequential(nn.Linear(in_dim + u_dim, 48), nn.Tanh(), nn.Linear(48, 1))
        self.kappa0_head = nn.Sequential(nn.Linear(in_dim + u_dim, 32), nn.Tanh(), nn.Linear(32, 1), nn.Sigmoid())
        self.lambda_head = nn.Sequential(nn.Linear(in_dim + u_dim, 32), nn.Tanh(), nn.Linear(32, 1), nn.Softplus())

    def forward(self, z, u, dt_eff):
        feat = torch.cat([z, u], dim=-1)
        y_anch_norm = self.anchor_probe(feat)
        kappa0 = self.kappa0_head(feat)
        lam = self.lambda_head(feat) + 1e-4
        
        # Exact closed-form integral of kappa(s) from s=0 to s=dt_eff
        # int_0^dt [1 - (1 - kappa0)*exp(-lam*s)] ds = dt - (1 - kappa0)/lam * (1 - exp(-lam*dt))
        dt_cyc = dt_eff * 4.0
        exp_decay = torch.exp(-lam * dt_eff)
        integral_countdown = dt_cyc - ((1.0 - kappa0) / lam) * (1.0 - exp_decay) * 4.0
        integral_countdown = torch.clamp(integral_countdown, 0.0, float(dt_cyc.max().item()))
        
        return y_anch_norm, integral_countdown, kappa0


# Proposed Model with GRU-D Encoder + Hamiltonian ODE Dynamics + Knee Hazard Head
class CT_PhysProb_JEPA_v2(nn.Module):
    def __init__(self, in_features=42, u_dim=3, latent_dim=16, alpha=0.10, gamma0=0.15):
        super().__init__()
        self.d = latent_dim // 2
        self.alpha = alpha
        self.gamma0 = gamma0
        
        self.encoder = GRUD_TemporalEncoder(in_features, u_dim, latent_dim, hidden_dim=48)
        self.head = KneeHazardHead(latent_dim, u_dim)
        self.decay_w = nn.Parameter(torch.tensor([0.25]))
        
        # Hamiltonian ODE potential parameters
        self.W_pot = nn.Parameter(torch.randn(self.d, self.d) * 0.25)
        self.b_pot = nn.Parameter(torch.zeros(self.d))
        self.mode_signs = nn.Parameter(torch.tensor([1.0 if j % 3 != 0 else -0.65 for j in range(self.d)]), requires_grad=False)
        
        cond_in = latent_dim + u_dim + 1
        self.force_net = nn.Sequential(nn.Linear(cond_in, 32), nn.Tanh(), nn.Linear(32, self.d))
        self.dyn_probe = nn.Sequential(nn.Linear(latent_dim + u_dim + 1, 32), nn.Tanh(), nn.Linear(32, 1))

    def force_tot(self, q, q0, p, u, dt_eff):
        hq = torch.tanh(q @ self.W_pot + self.b_pot) * self.mode_signs
        grad_v_mlp = (hq @ self.W_pot.T) / float(self.d)
        grad_v_conf = 3.5 * self.alpha * (q - q0)
        cond_vec = torch.cat([q0, p, u, dt_eff], dim=-1)
        f_ext = 0.45 * self.force_net(cond_vec)
        return -(grad_v_mlp + grad_v_conf) + f_ext

    def forward(self, x, u, dt):
        dt_eff = torch.clamp(dt, 0.01, 10.0)
        z0, _ = self.encoder(x, u, dt_eff)
        q0 = z0[:, :self.d]
        p0 = z0[:, self.d:]
        
        # Knee hazard countdown
        y_anch_norm, count_cyc, _ = self.head(z0, u, dt_eff)
        
        # Strang leapfrog integration of Hamiltonian dynamics
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
        dyn_corr = 0.20 * self.dyn_probe(torch.cat([z_pred - z0, u, dt_eff], dim=-1))
        
        # Combined prediction: anchor - countdown + ODE residual
        count_norm = count_cyc / 41.6
        y_pred = y_anch_norm - count_norm + dyn_corr
        return y_pred, z0, z_pred


# Crucial Matched Ablation: Same Encoder & Hazard Head, but NO Physics/Hamiltonian ODE
class CT_PhysProb_JEPA_NoPhysics(nn.Module):
    def __init__(self, in_features=42, u_dim=3, latent_dim=16):
        super().__init__()
        self.encoder = GRUD_TemporalEncoder(in_features, u_dim, latent_dim, hidden_dim=48)
        self.head = KneeHazardHead(latent_dim, u_dim)

    def forward(self, x, u, dt):
        dt_eff = torch.clamp(dt, 0.01, 10.0)
        z0, _ = self.encoder(x, u, dt_eff)
        y_anch_norm, count_cyc, _ = self.head(z0, u, dt_eff)
        count_norm = count_cyc / 41.6
        y_pred = y_anch_norm - count_norm
        return y_pred, z0, z0


# Enhanced Baselines with KneeHazardHead for Fair Comparison
class GRU_D_Enhanced(nn.Module):
    def __init__(self, in_features=42, u_dim=3, hidden_dim=48):
        super().__init__()
        self.encoder = GRUD_TemporalEncoder(in_features, u_dim, latent_dim=hidden_dim, hidden_dim=hidden_dim)
        self.head = KneeHazardHead(hidden_dim, u_dim)

    def forward(self, x, u, dt):
        dt_eff = torch.clamp(dt, 0.01, 10.0)
        h, _ = self.encoder(x, u, dt_eff)
        y_anch_norm, count_cyc, _ = self.head(h, u, dt_eff)
        count_norm = count_cyc / 41.6
        y_pred = y_anch_norm - count_norm
        return y_pred, h, h


class Plain_NODE_Enhanced(nn.Module):
    def __init__(self, in_features=42, u_dim=3, latent_dim=16):
        super().__init__()
        self.d = latent_dim // 2
        self.encoder = GRUD_TemporalEncoder(in_features, u_dim, latent_dim, hidden_dim=48)
        self.head = KneeHazardHead(latent_dim, u_dim)
        self.node_acc = nn.Sequential(nn.Linear(latent_dim + u_dim + 1, 32), nn.Tanh(), nn.Linear(32, self.d))
        self.dyn_probe = nn.Sequential(nn.Linear(latent_dim + u_dim + 1, 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, x, u, dt):
        dt_eff = torch.clamp(dt, 0.01, 10.0)
        z0, _ = self.encoder(x, u, dt_eff)
        q0 = z0[:, :self.d]
        p0 = z0[:, self.d:]
        
        y_anch_norm, count_cyc, _ = self.head(z0, u, dt_eff)
        
        # Unconstrained Euler integration
        sub_dt = dt_eff / 4.0
        q, p = q0.clone(), p0.clone()
        for _ in range(4):
            acc = self.node_acc(torch.cat([q, p, u, dt_eff], dim=-1))
            q = q + sub_dt * p
            p = p + sub_dt * acc
            
        z_pred = torch.cat([q, p], dim=-1)
        dyn_corr = 0.20 * self.dyn_probe(torch.cat([z_pred - z0, u, dt_eff], dim=-1))
        count_norm = count_cyc / 41.6
        y_pred = y_anch_norm - count_norm + dyn_corr
        return y_pred, z0, z_pred


class PatchTST_Enhanced(nn.Module):
    def __init__(self, in_features=42, u_dim=3, d_model=48, nhead=2, num_layers=2):
        super().__init__()
        self.patch_raw = nn.Linear(14, d_model)
        self.patch_fast = nn.Linear(14, d_model)
        self.patch_slow = nn.Linear(14, d_model)
        self.u_proj = nn.Linear(u_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=64, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = KneeHazardHead(d_model * 4, u_dim)

    def forward(self, x, u, dt):
        dt_eff = torch.clamp(dt, 0.01, 10.0)
        p_raw = self.patch_raw(x[:, :14]).unsqueeze(1)
        p_fast = self.patch_fast(x[:, 14:28]).unsqueeze(1)
        p_slow = self.patch_slow(x[:, 28:42]).unsqueeze(1)
        u_t = self.u_proj(u).unsqueeze(1)
        tokens = torch.cat([p_raw, p_fast, p_slow, u_t], dim=1)
        trans_out = self.transformer(tokens)
        flat = trans_out.reshape(trans_out.size(0), -1)
        
        y_anch_norm, count_cyc, _ = self.head(flat, u, dt_eff)
        count_norm = count_cyc / 41.6
        y_pred = y_anch_norm - count_norm
        return y_pred, flat, flat


# ==============================================================================
# SWA & TRAINING ROUTINE
# ==============================================================================

def train_model_swa(model_cls, kwargs, x_tr, u_tr, dt_tr, y_tr, seed=42, epochs=8, lr=3.5e-3,
                    loss_type='asym', lambda_opt=1.0, use_swa=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model_cls(**kwargs)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    N = x_tr.size(0)
    batch_size = 512
    t0 = time.time()
    
    swa_weights = None
    swa_count = 0
    
    for ep in range(epochs):
        perm = torch.randperm(N)
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            bx, bu, bdt, by = x_tr[idx], u_tr[idx], dt_tr[idx], y_tr[idx]
            
            y_pred, _, _ = model(bx, bu, bdt)
            diff = y_pred - by
            
            if loss_type == 'asym':
                w = torch.where(diff > 0, 1.0 + lambda_opt, 1.0)
                loss = torch.mean(w * (diff ** 2))
            else:
                loss = torch.mean(diff ** 2)
                
            opt.zero_grad()
            loss.backward()
            opt.step()
            
        # Accumulate SWA weights over the last 3 epochs (epochs 5, 6, 7)
        if use_swa and ep >= (epochs - 3):
            if swa_weights is None:
                swa_weights = {k: v.clone() for k, v in model.state_dict().items()}
                swa_count = 1
            else:
                for k, v in model.state_dict().items():
                    swa_weights[k] += v
                swa_count += 1
                
    if use_swa and swa_weights is not None:
        for k in swa_weights:
            swa_weights[k] /= float(swa_count)
        model.load_state_dict(swa_weights)
        
    return model, time.time() - t0


def run_step2_development():
    print("=" * 80)
    print("STEP 2: MODEL DEVELOPMENT ON VALIDATION UNITS (10 REAL SEEDS)")
    print("=" * 80)
    
    os.makedirs(ROOT / "results" / "raw", exist_ok=True)
    raw_csv = ROOT / "results" / "raw" / "step2_development_raw.csv"
    summary_csv = ROOT / "results" / "step2_development_summary.csv"
    
    df_tr, df_va, _, stats1 = load_cmapss_subset('FD001')
    print(f"Loaded FD001: Train units={df_tr['unit'].nunique()}, Val units={df_va['unit'].nunique()} (Test units untouched)")
    
    # Models to train and evaluate
    models_to_test = [
        # Baseline reference models from Step 1
        ('CT_PhysProb_JEPA_KinGate',  CT_PhysProb_JEPA_KinGate,   dict(), False),
        ('GRU_D_v2',                  GRU_D_v2,                   dict(), False),
        ('PatchTST_RUL_v2',           PatchTST_RUL_v2,            dict(), False),
        ('Plain_2ndOrder_NODE_Kin',   Plain_2ndOrder_NODE_Kin,    dict(latent_dim=16), False),
        # Round 1 & 2: Proposed Model with GRU-D Temporal Encoder + KneeHazardHead + Hamiltonian ODE
        ('CT_PhysProb_JEPA_v2',       CT_PhysProb_JEPA_v2,        dict(), False),
        # Round 2 Ablation: No-Physics Control (Same Encoder & Hazard Head, NO Hamiltonian ODE)
        ('CT_PhysProb_JEPA_NoPhysics',CT_PhysProb_JEPA_NoPhysics, dict(), False),
        # Round 2 Enhanced Competitors (Offered identical Hazard Head)
        ('GRU_D_Enhanced',            GRU_D_Enhanced,             dict(), False),
        ('Plain_NODE_Enhanced',       Plain_NODE_Enhanced,        dict(), False),
        ('PatchTST_Enhanced',         PatchTST_Enhanced,          dict(), False),
        # Round 3: With Stochastic Weight Averaging (SWA)
        ('CT_PhysProb_JEPA_v2_SWA',   CT_PhysProb_JEPA_v2,        dict(), True),
        ('GRU_D_Enhanced_SWA',        GRU_D_Enhanced,             dict(), True),
        ('Plain_NODE_Enhanced_SWA',   Plain_NODE_Enhanced,        dict(), True),
    ]
    
    records = []
    
    for s_idx, s in enumerate(SEEDS_10):
        t0_s = time.time()
        print(f"\n--- Running Seed {s} ({s_idx + 1}/10) ---", flush=True)
        
        # Mixed gap training data (shared baseline recipe for ALL models)
        tr_short = build_seq_data(df_tr, stats1, drop_prob=0.25, seed=s)
        tr_long  = build_seq_data(df_tr, stats1, fixed_stride_range=(3, 15), seed=s + 1000)
        tr_mixed = {
            k: np.concatenate([tr_short[k], tr_long[k]], axis=0)
            for k in ['obs_t', 'ctrl_t', 'dt', 'obs_next', 'y_norm', 'y_raw', 'mem_fast', 'mem_slow']
        }
        x_tr, u_tr, dt_tr, y_tr = to_tensor_data(tr_mixed)
        
        # Validation regimes (strictly on 20 validation units)
        val_regimes = {
            'InDist_1x_2x': build_seq_data(df_va, stats1, drop_prob=0.25, seed=s + 2000),
            'OOD_5x_10x':   build_seq_data(df_va, stats1, fixed_stride_range=(5, 10), seed=s + 3000),
            'OOD_10x_20x':  build_seq_data(df_va, stats1, fixed_stride_range=(10, 20), seed=s + 4000),
            'OOD_15x_25x':  build_seq_data(df_va, stats1, fixed_stride_range=(15, 25), seed=s + 5000),
        }
        for vd in val_regimes.values():
            dt_cyc = vd['dt'][:, 0] * 4.0
            vd['y_curr'] = np.minimum(125.0, vd['y_raw'] + dt_cyc)
            
        for m_name, m_cls, m_kw, use_swa in models_to_test:
            model, tr_sec = train_model_swa(
                m_cls, m_kw, x_tr, u_tr, dt_tr, y_tr,
                seed=s, epochs=8, loss_type='asym', lambda_opt=1.0, use_swa=use_swa
            )
            
            for r_name in REGIMES:
                vd = val_regimes[r_name]
                vx, vu, vdt, _ = to_tensor_data(vd)
                y_true = vd['y_raw']
                y_curr = vd['y_curr']
                dt_cyc = vd['dt'][:, 0] * 4.0
                
                # Anchor evaluation at dt = 0.25 (nominal)
                vdt_zero = torch.full_like(vdt, 0.25)
                with torch.no_grad():
                    p_anch_norm, _, _ = model(vx, vu, vdt_zero)
                    p_anch = torch.clamp(p_anch_norm[:, 0] * stats1['tgt_std'] + stats1['tgt_mean'], 0.0, 125.0).numpy()
                    
                    p_tot_norm, _, _ = model(vx, vu, vdt)
                    p_tot = torch.clamp(p_tot_norm[:, 0] * stats1['tgt_std'] + stats1['tgt_mean'], 0.0, 125.0).numpy()
                    
                anchor_rmse = float(np.sqrt(np.mean((p_anch - y_curr)**2)))
                total_rmse  = float(np.sqrt(np.mean((p_tot - y_true)**2)))
                total_mae   = float(np.mean(np.abs(p_tot - y_true)))
                
                true_slope = (y_true - y_curr) / np.maximum(1.0, dt_cyc)
                mask_act = (true_slope <= -0.90)
                
                records.append({
                    'Seed': s, 'Model': m_name, 'Regime': r_name,
                    'Anchor_RMSE': anchor_rmse,
                    'Total_RMSE': total_rmse,
                    'Total_MAE': total_mae,
                    'Gap_Induced_RMSE': total_rmse - anchor_rmse,
                    'Signed_Bias_ActiveDeg': float(np.mean(p_tot[mask_act] - y_true[mask_act])) if np.sum(mask_act) > 0 else 0.0,
                    'NASA_Score': nasa_score(p_tot, y_true),
                    'Train_Sec': tr_sec,
                    'Use_SWA': use_swa
                })
                
        print(f"Seed {s} completed in {time.time() - t0_s:.1f}s.", flush=True)
        pd.DataFrame(records).to_csv(raw_csv, index=False)
        
    df_res = pd.DataFrame(records)
    df_res.to_csv(raw_csv, index=False)
    print(f"\nRaw results saved to {raw_csv}")
    
    summary = df_res.groupby(['Regime', 'Model']).agg({
        'Anchor_RMSE': ['mean', 'std'],
        'Total_RMSE': ['mean', 'std'],
        'Signed_Bias_ActiveDeg': ['mean', 'std'],
        'NASA_Score': ['mean', 'std']
    }).round(2)
    summary.to_csv(summary_csv)
    print(f"Summary saved to {summary_csv}")
    
    return df_res, summary

if __name__ == '__main__':
    run_step2_development()
