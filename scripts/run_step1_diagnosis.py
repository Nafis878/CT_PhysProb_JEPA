#!/usr/bin/env python3
"""
STEP 1: DIAGNOSE (Validation Only, 10 Real Seeds)
Strict Empirical Audit of CT_PhysProb_JEPA and Competitors on NASA C-MAPSS FD001.

NON-NEGOTIABLE PROTOCOL:
- Evaluates strictly on 20 validation engines from data/splits/cmapss_FD001.json.
- Official test engines (test_FD001.txt, RUL_FD001.txt) remain completely UNTOUCHED.
- Every metric originates from a live train+evaluate loop logged to results/raw/.
- Zero synthetic sampling, zero hardcoded constants, zero seed reconstruction.
- 10 real seeds: [42, 100, 2024, 777, 999, 101, 202, 303, 404, 505].
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

SEEDS_10 = [42, 100, 2024, 777, 999, 101, 202, 303, 404, 505]
REGIMES = ['InDist_1x_2x', 'OOD_5x_10x', 'OOD_10x_20x', 'OOD_15x_25x']

# ==============================================================================
# MODEL ARCHITECTURES (Equalized Tuning Budget: 8 Epochs, AdamW, H=48, Latent=16)
# ==============================================================================

class CT_PhysProb_JEPA(nn.Module):
    """Proposed model with controllable horizon policy: clamped (gap-blind) vs adaptive (full-gap)."""
    def __init__(self, in_features=42, u_dim=3, latent_dim=16, alpha=0.10, gamma=0.15, clamped=True):
        super().__init__()
        self.d = latent_dim // 2
        self.alpha = alpha
        self.gamma = gamma
        self.clamped = clamped
        
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 48),
            nn.Tanh(),
            nn.Linear(48, latent_dim)
        )
        self.W_pot = nn.Parameter(torch.randn(self.d, self.d) * 0.25)
        self.b_pot = nn.Parameter(torch.zeros(self.d))
        self.mode_signs = nn.Parameter(torch.tensor([1.0 if j % 3 != 0 else -0.65 for j in range(self.d)]), requires_grad=False)
        
        cond_in = latent_dim + u_dim + 1
        self.force_net = nn.Sequential(
            nn.Linear(cond_in, 32),
            nn.Tanh(),
            nn.Linear(32, self.d)
        )
        self.probe = nn.Sequential(
            nn.Linear(latent_dim + u_dim, 48),
            nn.Tanh(),
            nn.Linear(48, 1)
        )

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
        
        if self.clamped:
            dt_eff = torch.clamp(dt, 0.01, 10.0)
            h_int = torch.clamp(dt_eff, max=0.35)
            dt_cond = torch.clamp(dt_eff, max=1.0)
            m_steps = 4
            sub_dt = h_int / float(m_steps)
        else:
            dt_eff = torch.clamp(dt, 0.01, 10.0)
            h_int = dt_eff
            dt_cond = dt_eff
            max_dt = float(dt_eff.max().item())
            m_steps = max(4, int(np.ceil(max_dt / 0.0875)))
            sub_dt = h_int / float(m_steps)
            
        q, p = q0.clone(), p0.clone()
        decay = torch.exp(-0.5 * self.gamma * sub_dt)
        for _ in range(m_steps):
            p = decay * p
            p = p + 0.5 * sub_dt * self.force_tot(q, q0, p, u, dt_cond)
            q = q + sub_dt * p
            p = p + 0.5 * sub_dt * self.force_tot(q, q0, p, u, dt_cond)
            p = decay * p
            
        z_pred = torch.cat([q, p], dim=-1)
        y_pred = self.probe(torch.cat([z_pred, u], dim=-1))
        return y_pred, z0, z_pred


class Plain_2ndOrder_NODE_Kin(nn.Module):
    """Causal control: 2nd-order ODE with kinematic loss, no EL potential, gamma=0, Euler M=1."""
    def __init__(self, in_features=42, u_dim=3, latent_dim=16, alpha=0.10):
        super().__init__()
        self.d = latent_dim // 2
        self.alpha = alpha
        
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 48),
            nn.Tanh(),
            nn.Linear(48, latent_dim)
        )
        cond_in = latent_dim + u_dim + 1
        self.force_net = nn.Sequential(
            nn.Linear(cond_in, 32),
            nn.Tanh(),
            nn.Linear(32, self.d)
        )
        self.probe = nn.Sequential(
            nn.Linear(latent_dim + u_dim, 48),
            nn.Tanh(),
            nn.Linear(48, 1)
        )

    def forward(self, x, u, dt):
        z0 = self.encoder(x)
        q0 = z0[:, :self.d]
        p0 = z0[:, self.d:]
        dt_eff = torch.clamp(dt, 0.01, 10.0)
        
        cond_vec = torch.cat([q0, p0, u, dt_eff], dim=-1)
        f_ext = 0.45 * self.force_net(cond_vec)
        q1 = q0 + dt_eff * p0
        p1 = p0 + dt_eff * f_ext
        
        z_pred = torch.cat([q1, p1], dim=-1)
        y_pred = self.probe(torch.cat([z_pred, u], dim=-1))
        return y_pred, z0, z_pred


class GRU_D(nn.Module):
    """Continuous-Time Gated Recurrent Unit with exponential decay of hidden state and inputs."""
    def __init__(self, in_features=42, u_dim=3, hidden_dim=48):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.decay_h = nn.Sequential(nn.Linear(1, hidden_dim), nn.Softplus())
        self.decay_x = nn.Sequential(nn.Linear(1, in_features), nn.Softplus())
        self.enc_h0 = nn.Sequential(nn.Linear(in_features + u_dim, hidden_dim), nn.Tanh())
        self.gru_cell = nn.GRUCell(in_features + u_dim, hidden_dim)
        self.probe = nn.Sequential(
            nn.Linear(hidden_dim + u_dim, 48),
            nn.Tanh(),
            nn.Linear(48, 1)
        )

    def forward(self, x, u, dt):
        dt_eff = torch.clamp(dt, 0.01, 10.0)
        h0 = self.enc_h0(torch.cat([x, u], dim=-1))
        gamma_h = torch.exp(-self.decay_h(dt_eff))
        gamma_x = torch.exp(-self.decay_x(dt_eff))
        h_decayed = gamma_h * h0
        x_imputed = gamma_x * x
        h_next = self.gru_cell(torch.cat([x_imputed, u], dim=-1), h_decayed)
        y_pred = self.probe(torch.cat([h_next, u], dim=-1))
        return y_pred, h0, h_next


class Latent_SDE(nn.Module):
    """Continuous-Time Neural SDE with Euler-Maruyama drift and diffusion."""
    def __init__(self, in_features=42, u_dim=3, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 48),
            nn.Tanh(),
            nn.Linear(48, latent_dim)
        )
        self.drift_net = nn.Sequential(
            nn.Linear(latent_dim + u_dim + 1, 48),
            nn.Tanh(),
            nn.Linear(48, latent_dim)
        )
        self.diff_net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid()
        )
        self.probe = nn.Sequential(
            nn.Linear(latent_dim + u_dim, 48),
            nn.Tanh(),
            nn.Linear(48, 1)
        )

    def forward(self, x, u, dt):
        z0 = self.encoder(x)
        dt_eff = torch.clamp(dt, 0.01, 10.0)
        max_dt = float(dt_eff.max().item())
        m_steps = max(4, int(np.ceil(max_dt / 0.10)))
        sub_dt = dt_eff / float(m_steps)
        sqrt_sub_dt = torch.sqrt(sub_dt)
        
        z = z0.clone()
        for _ in range(m_steps):
            f = self.drift_net(torch.cat([z, u, dt_eff], dim=-1))
            g = 0.15 * self.diff_net(z)
            eps = torch.randn_like(z)
            z = z + sub_dt * f + sqrt_sub_dt * g * eps
            
        y_pred = self.probe(torch.cat([z, u], dim=-1))
        return y_pred, z0, z


class PatchTST_RUL(nn.Module):
    """Channel-Independent Patched Transformer for continuous-time sequence RUL regression."""
    def __init__(self, in_features=42, u_dim=3, d_model=48, nhead=2, num_layers=2):
        super().__init__()
        self.patch_raw = nn.Linear(14, d_model)
        self.patch_fast = nn.Linear(14, d_model)
        self.patch_slow = nn.Linear(14, d_model)
        self.u_proj = nn.Linear(u_dim, d_model)
        self.time_proj = nn.Sequential(nn.Linear(1, d_model), nn.Tanh())
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=64, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.probe = nn.Sequential(
            nn.Linear(d_model * 4, 48),
            nn.Tanh(),
            nn.Linear(48, 1)
        )

    def forward(self, x, u, dt):
        dt_eff = torch.clamp(dt, 0.01, 10.0)
        p_raw = self.patch_raw(x[:, :14]).unsqueeze(1)
        p_fast = self.patch_fast(x[:, 14:28]).unsqueeze(1)
        p_slow = self.patch_slow(x[:, 28:42]).unsqueeze(1)
        p_u = self.u_proj(u).unsqueeze(1)
        
        t_emb = self.time_proj(dt_eff).unsqueeze(1)
        tokens = torch.cat([p_raw + t_emb, p_fast + t_emb, p_slow + t_emb, p_u], dim=1)
        trans_out = self.transformer(tokens)
        flat = trans_out.reshape(trans_out.size(0), -1)
        y_pred = self.probe(flat)
        return y_pred, flat, flat


# ==============================================================================
# DATA CONVERTERS & TRAINER
# ==============================================================================

def to_tensor_data(d):
    x = torch.cat([
        torch.tensor(d['obs_t'], dtype=torch.float32),
        torch.tensor(d['mem_fast'], dtype=torch.float32),
        torch.tensor(d['mem_slow'], dtype=torch.float32)
    ], dim=-1)
    u = torch.tensor(d['ctrl_t'], dtype=torch.float32)
    dt = torch.tensor(d['dt'], dtype=torch.float32)
    y = torch.tensor(d['y_norm'], dtype=torch.float32)
    return x, u, dt, y


def train_single_model(model_cls, kwargs, x_tr, u_tr, dt_tr, y_tr, seed=42, epochs=8, lr=3.5e-3, has_kin=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model_cls(**kwargs)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    N = x_tr.size(0)
    batch_size = 512
    t0 = time.time()
    
    for ep in range(epochs):
        perm = torch.randperm(N)
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            bx, bu, bdt, by = x_tr[idx], u_tr[idx], dt_tr[idx], y_tr[idx]
            
            y_pred, z0, z_pred = model(bx, bu, bdt)
            loss_mse = nn.functional.mse_loss(y_pred, by)
            
            if has_kin and hasattr(model, 'd'):
                d = model.d
                q0 = z0[:, :d]
                p0 = z0[:, d:]
                q_next = z_pred[:, :d]
                dt_kin = torch.clamp(bdt, min=0.1)
                v_sec = (q_next - q0) / dt_kin
                loss_kin = 0.30 * nn.functional.mse_loss(p0, v_sec.detach())
                loss = loss_mse + loss_kin
            else:
                loss = loss_mse
                
            opt.zero_grad()
            loss.backward()
            opt.step()
            
    train_sec = time.time() - t0
    model.eval()
    return model, train_sec


# ==============================================================================
# MAIN DIAGNOSTIC AUDIT RUNNER
# ==============================================================================

def run_step1_diagnosis():
    print("=" * 80)
    print("STEP 1: RIGOROUS DIAGNOSTIC AUDIT (Validation Units Only, 10 Real Seeds)")
    print("=" * 80)
    
    # Create raw logging directory
    raw_dir = ROOT / "results" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_log_path = raw_dir / "step1_diagnosis_raw.csv"
    
    # Load dataset strictly into Train (80 engines) and Val (20 engines)
    df_tr, df_va, df_te, stats1 = load_cmapss_subset('FD001')
    print(f"Dataset Loaded: Train engines={df_tr['unit'].nunique()}, Val engines={df_va['unit'].nunique()}")
    print("Test engines are strictly untouched.")
    
    raw_records = []
    
    # Model specifications to run with equal tuning budget
    model_specs = [
        ('CT_PhysProb_JEPA_Clamped', CT_PhysProb_JEPA, dict(clamped=True), True),
        ('CT_PhysProb_JEPA_Adaptive', CT_PhysProb_JEPA, dict(clamped=False), True),
        ('Plain_2ndOrder_NODE_Kin', Plain_2ndOrder_NODE_Kin, dict(), True),
        ('GRU_D', GRU_D, dict(), False),
        ('Latent_SDE', Latent_SDE, dict(), False),
        ('PatchTST_RUL', PatchTST_RUL, dict(), False),
    ]

    for s_idx, s in enumerate(SEEDS_10):
        print(f"\n--- Running Seed {s} ({s_idx + 1}/{len(SEEDS_10)}) ---")
        
        # Build sequence data for train and validation regimes
        tr_d = build_seq_data(df_tr, stats1, drop_prob=0.25, seed=s)
        va_in = build_seq_data(df_va, stats1, drop_prob=0.25, seed=s + 1000)
        va_5x = build_seq_data(df_va, stats1, fixed_stride_range=(5, 10), seed=s + 2000)
        va_10x = build_seq_data(df_va, stats1, fixed_stride_range=(10, 20), seed=s + 3000)
        va_15x = build_seq_data(df_va, stats1, fixed_stride_range=(15, 25), seed=s + 4000)
        
        val_regimes = {
            'InDist_1x_2x': va_in,
            'OOD_5x_10x': va_5x,
            'OOD_10x_20x': va_10x,
            'OOD_15x_25x': va_15x,
        }
        for r_name, vd in val_regimes.items():
            dt_cyc = vd['dt'][:, 0] * 4.0
            vd['y_curr'] = np.minimum(125.0, vd['y_raw'] + dt_cyc)

        x_tr, u_tr, dt_tr, y_tr = to_tensor_data(tr_d)
        
        # Store predictions of models for later analysis and for Estimated LOCF
        preds_by_model = {}

        # Train and evaluate learned models
        for m_name, m_cls, m_kw, has_kin in model_specs:
            model, tr_time = train_single_model(
                m_cls, m_kw, x_tr, u_tr, dt_tr, y_tr, seed=s, epochs=8, has_kin=has_kin
            )
            
            for r_name in REGIMES:
                vd = val_regimes[r_name]
                vx, vu, vdt, _ = to_tensor_data(vd)
                with torch.no_grad():
                    y_pred_norm, _, _ = model(vx, vu, vdt)
                    y_pred = torch.clamp(y_pred_norm[:, 0] * stats1['tgt_std'] + stats1['tgt_mean'], 0.0, 125.0).numpy()
                
                preds_by_model[(m_name, r_name)] = y_pred
                
                # Compute metrics
                y_true = vd['y_raw']
                dt_cyc = vd['dt'][:, 0] * 4.0
                true_slope = (y_true - vd['y_curr']) / np.maximum(1.0, dt_cyc)
                mask_act = (true_slope <= -0.90)
                
                rmse_all = float(np.sqrt(np.mean((y_pred - y_true)**2)))
                mae_all = float(np.mean(np.abs(y_pred - y_true)))
                bias_all = float(np.mean(y_pred - y_true))
                bias_act = float(np.mean(y_pred[mask_act] - y_true[mask_act])) if np.sum(mask_act) > 0 else 0.0
                rmse_act = float(np.sqrt(np.mean((y_pred[mask_act] - y_true[mask_act])**2))) if np.sum(mask_act) > 0 else 0.0
                n_score = nasa_score(y_pred, y_true)
                
                raw_records.append({
                    'Seed': s, 'Model': m_name, 'Regime': r_name,
                    'RMSE_All': rmse_all, 'MAE_All': mae_all,
                    'Signed_Bias_All': bias_all, 'Signed_Bias_ActiveDeg': bias_act,
                    'RMSE_ActiveDeg': rmse_act, 'NASA_Score': n_score,
                    'Train_Sec': tr_time
                })
        
        # Evaluate Time-Corrected LOCF Controls:
        # (a) LOCF_TimeCorrected_Oracle: y_true_anchor - dt
        # (b) LOCF_TimeCorrected_Estimated: y_pred_anchor - dt (using anchor estimated by proposed model at dt=0)
        # First compute estimated anchor at dt=0 using Adaptive model
        adapt_model, _ = train_single_model(
            CT_PhysProb_JEPA, dict(clamped=False), x_tr, u_tr, dt_tr, y_tr, seed=s, epochs=8, has_kin=True
        )
        
        for r_name in REGIMES:
            vd = val_regimes[r_name]
            y_true = vd['y_raw']
            dt_cyc = vd['dt'][:, 0] * 4.0
            true_slope = (y_true - vd['y_curr']) / np.maximum(1.0, dt_cyc)
            mask_act = (true_slope <= -0.90)
            
            # (a) Oracle Anchor LOCF
            pred_locf_oracle = np.clip(vd['y_curr'] - dt_cyc, 0.0, 125.0)
            raw_records.append({
                'Seed': s, 'Model': 'TimeCorrected_LOCF_Oracle', 'Regime': r_name,
                'RMSE_All': float(np.sqrt(np.mean((pred_locf_oracle - y_true)**2))),
                'MAE_All': float(np.mean(np.abs(pred_locf_oracle - y_true))),
                'Signed_Bias_All': float(np.mean(pred_locf_oracle - y_true)),
                'Signed_Bias_ActiveDeg': float(np.mean(pred_locf_oracle[mask_act] - y_true[mask_act])),
                'RMSE_ActiveDeg': float(np.sqrt(np.mean((pred_locf_oracle[mask_act] - y_true[mask_act])**2))),
                'NASA_Score': nasa_score(pred_locf_oracle, y_true),
                'Train_Sec': 0.0
            })
            
            # (b) Estimated Anchor LOCF: advance 0 step to get anchor prediction, then subtract dt
            vx, vu, _, _ = to_tensor_data(vd)
            vdt_zero = torch.full_like(vx[:, 0:1], 0.001)
            with torch.no_grad():
                y_anchor_norm, _, _ = adapt_model(vx, vu, vdt_zero)
                y_anchor = torch.clamp(y_anchor_norm[:, 0] * stats1['tgt_std'] + stats1['tgt_mean'], 0.0, 125.0).numpy()
            pred_locf_est = np.clip(y_anchor - dt_cyc, 0.0, 125.0)
            raw_records.append({
                'Seed': s, 'Model': 'TimeCorrected_LOCF_Estimated', 'Regime': r_name,
                'RMSE_All': float(np.sqrt(np.mean((pred_locf_est - y_true)**2))),
                'MAE_All': float(np.mean(np.abs(pred_locf_est - y_true))),
                'Signed_Bias_All': float(np.mean(pred_locf_est - y_true)),
                'Signed_Bias_ActiveDeg': float(np.mean(pred_locf_est[mask_act] - y_true[mask_act])),
                'RMSE_ActiveDeg': float(np.sqrt(np.mean((pred_locf_est[mask_act] - y_true[mask_act])**2))),
                'NASA_Score': nasa_score(pred_locf_est, y_true),
                'Train_Sec': 0.0
            })
            
        print(f"Seed {s} complete.")

    # Save complete raw logs
    df_raw = pd.DataFrame(raw_records)
    df_raw.to_csv(raw_log_path, index=False)
    print(f"\nRaw per-seed logs saved to {raw_log_path} ({len(df_raw)} records)")
    
    # ==============================================================================
    # STATISTICAL ANALYSIS & SUMMARY TABLES
    # ==============================================================================
    summary_rows = []
    all_models = df_raw['Model'].unique().tolist()
    
    # We focus paired statistical tests against CT_PhysProb_JEPA_Adaptive
    ref_model = 'CT_PhysProb_JEPA_Adaptive'
    
    for r_name in REGIMES:
        sub_r = df_raw[df_raw['Regime'] == r_name]
        ref_rmse = sub_r[sub_r['Model'] == ref_model].sort_values('Seed')['RMSE_All'].values
        
        for m in all_models:
            sub_m = sub_r[sub_r['Model'] == m].sort_values('Seed')
            m_rmse = sub_m['RMSE_All'].values
            m_mae = sub_m['MAE_All'].values
            m_bias_all = sub_m['Signed_Bias_All'].values
            m_bias_act = sub_m['Signed_Bias_ActiveDeg'].values
            m_rmse_act = sub_m['RMSE_ActiveDeg'].values
            m_nasa = sub_m['NASA_Score'].values
            
            # Paired differences vs reference
            diff = m_rmse - ref_rmse
            d_mean = float(np.mean(diff))
            d_sd = float(np.std(diff, ddof=1))
            
            if m == ref_model:
                p_t = 1.0
                p_w = 1.0
            else:
                p_t = float(stats.ttest_rel(m_rmse, ref_rmse).pvalue) if d_sd > 1e-8 else 1.0
                try:
                    p_w = float(stats.wilcoxon(m_rmse, ref_rmse).pvalue)
                except:
                    p_w = 1.0
                    
            ci_half = 2.262 * d_sd / np.sqrt(len(diff)) # t_0.975 for df=9
            
            summary_rows.append({
                'Regime': r_name,
                'Model': m,
                'RMSE_Mean': float(np.mean(m_rmse)),
                'RMSE_SD': float(np.std(m_rmse, ddof=1)),
                'MAE_Mean': float(np.mean(m_mae)),
                'Signed_Bias_Active_Mean': float(np.mean(m_bias_act)),
                'RMSE_Active_Mean': float(np.mean(m_rmse_act)),
                'NASA_Score_Mean': float(np.mean(m_nasa)),
                'Paired_Diff_vs_Adaptive': d_mean,
                'Diff_95CI_Lo': d_mean - ci_half,
                'Diff_95CI_Hi': d_mean + ci_half,
                'p_t_paired': p_t,
                'p_w_paired': p_w
            })
            
    df_summary = pd.DataFrame(summary_rows)
    summary_path = ROOT / "results" / "step1_diagnosis_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"Summary table saved to {summary_path}")
    
    # Print Primary Endpoint (OOD_10x_20x)
    print("\n" + "=" * 80)
    print("PRIMARY ENDPOINT (OOD 10x-20x) VALIDATION SUMMARY (10 REAL SEEDS)")
    print("=" * 80)
    print(df_summary[df_summary['Regime'] == 'OOD_10x_20x'][[
        'Model', 'RMSE_Mean', 'RMSE_SD', 'Signed_Bias_Active_Mean', 'NASA_Score_Mean',
        'Paired_Diff_vs_Adaptive', 'p_t_paired'
    ]].sort_values('RMSE_Mean').to_string(index=False))

    return df_raw, df_summary


if __name__ == '__main__':
    run_step1_diagnosis()
