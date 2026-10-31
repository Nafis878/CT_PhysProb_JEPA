#!/usr/bin/env python3
"""
MECHANISM ABLATIONS UNDER STRESSORS (10 REAL SEEDS, FD001 VALIDATION UNITS)
Protocol:
- Evaluates whether the proposed model's behavior under stressors is genuinely driven by
  symplectic Hamiltonian integration vs. Euler integration vs. damping vs. no-physics ablation.
- Provides canonical mask inputs to GRU-D and all models to eliminate zero-fill unfairness.
- Equal compute budget, identical training recipe (mixed gaps), identical SWA across last 3 epochs.
- All numbers logged directly to results/raw/mechanism_ablation_raw.csv.
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
from scripts.run_step1_diagnosis import to_tensor_data, SEEDS_10
from scripts.run_step2_development import GRUD_TemporalEncoder, KneeHazardHead

# ==============================================================================
# MODEL DEFINITIONS FOR MECHANISM ABLATION
# ==============================================================================

class CT_PhysProb_JEPA_IntegratorAblation(nn.Module):
    """
    Parametric model allowing exact mechanism ablation:
    - integrator: 'strang' (symplectic) vs 'euler' (non-symplectic)
    - damped: True (gamma > 0) vs False (gamma = 0)
    - has_physics: True (ODE active) vs False (no ODE, pure encoder+head)
    Input includes sensor features + missingness mask: in_features = 42 + 14 = 56.
    """
    def __init__(self, in_features=56, u_dim=3, latent_dim=16, alpha=0.10, gamma0=0.15,
                 integrator='strang', damped=True, has_physics=True):
        super().__init__()
        self.d = latent_dim // 2
        self.alpha = alpha
        self.gamma0 = gamma0 if damped else 0.0
        self.integrator = integrator
        self.damped = damped
        self.has_physics = has_physics
        
        self.encoder = GRUD_TemporalEncoder(in_features, u_dim, latent_dim, hidden_dim=48)
        self.head = KneeHazardHead(latent_dim, u_dim)
        
        if self.has_physics:
            self.W_pot = nn.Parameter(torch.randn(self.d, self.d) * 0.25)
            self.b_pot = nn.Parameter(torch.zeros(self.d))
            self.mode_signs = nn.Parameter(torch.tensor([1.0 if j % 3 != 0 else -0.65 for j in range(self.d)]), requires_grad=False)
            
            cond_in = latent_dim + u_dim + 1
            self.force_net = nn.Sequential(nn.Linear(cond_in, 32), nn.Tanh(), nn.Linear(32, self.d))
            self.dyn_probe = nn.Sequential(nn.Linear(latent_dim + u_dim + 1, 32), nn.Tanh(), nn.Linear(32, 1))
            self.decay_w = nn.Parameter(torch.tensor([0.25])) if damped else None

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
        y_anch_norm, count_cyc, _ = self.head(z0, u, dt_eff)
        count_norm = count_cyc / 41.6
        
        if not self.has_physics:
            y_pred = y_anch_norm - count_norm
            return y_pred, z0, z0
            
        q0 = z0[:, :self.d]
        p0 = z0[:, self.d:]
        
        if self.damped and self.decay_w is not None:
            gamma = self.gamma0 * (1.0 + torch.relu(self.decay_w * dt_eff))
        else:
            gamma = torch.zeros_like(dt_eff)
            
        max_dt = float(dt_eff.max().item())
        m_steps = max(4, int(np.ceil(max_dt / 0.0875)))
        sub_dt = dt_eff / float(m_steps)
        
        q, p = q0.clone(), p0.clone()
        decay = torch.exp(-0.5 * gamma * sub_dt)
        
        if self.integrator == 'strang':
            # Symplectic Strang splitting
            for _ in range(m_steps):
                p = decay * p
                p = p + 0.5 * sub_dt * self.force_tot(q, q0, p, u, dt_eff)
                q = q + sub_dt * p
                p = p + 0.5 * sub_dt * self.force_tot(q, q0, p, u, dt_eff)
                p = decay * p
        elif self.integrator == 'euler':
            # Non-symplectic forward Euler
            for _ in range(m_steps):
                f = self.force_tot(q, q0, p, u, dt_eff) - gamma * p
                q = q + sub_dt * p
                p = p + sub_dt * f
                
        z_pred = torch.cat([q, p], dim=-1)
        dyn_corr = 0.20 * self.dyn_probe(torch.cat([z_pred - z0, u, dt_eff], dim=-1))
        y_pred = y_anch_norm - count_norm + dyn_corr
        return y_pred, z0, z_pred


class GRU_D_Fair_SWA(nn.Module):
    """
    Canonical GRU-D equipped with mask inputs, gap-aware decay, and SWA.
    Input: [x, mask] with in_features = 42 + 14 = 56.
    """
    def __init__(self, in_features=56, u_dim=3, hidden_dim=48, use_hazard_head=False):
        super().__init__()
        self.use_hazard_head = use_hazard_head
        self.decay_h = nn.Sequential(nn.Linear(1, hidden_dim), nn.Softplus())
        self.decay_x = nn.Sequential(nn.Linear(1, in_features), nn.Softplus())
        self.enc_h0 = nn.Sequential(nn.Linear(in_features + u_dim, hidden_dim), nn.Tanh())
        self.gru_cell = nn.GRUCell(in_features + u_dim, hidden_dim)
        
        if self.use_hazard_head:
            self.head = KneeHazardHead(hidden_dim, u_dim)
        else:
            self.probe = nn.Sequential(nn.Linear(hidden_dim + u_dim + 1, 48), nn.Tanh(), nn.Linear(48, 1))

    def forward(self, x, u, dt):
        dt_eff = torch.clamp(dt, 0.01, 10.0)
        h0 = self.enc_h0(torch.cat([x, u], dim=-1))
        gamma_h = torch.exp(-self.decay_h(dt_eff))
        gamma_x = torch.exp(-self.decay_x(dt_eff))
        h_decayed = gamma_h * h0
        x_imputed = gamma_x * x
        h_next = self.gru_cell(torch.cat([x_imputed, u], dim=-1), h_decayed)
        
        if self.use_hazard_head:
            y_anch_norm, count_cyc, _ = self.head(h_next, u, dt_eff)
            count_norm = count_cyc / 41.6
            y_pred = y_anch_norm - count_norm
        else:
            y_pred = self.probe(torch.cat([h_next, u, dt_eff], dim=-1))
            
        return y_pred, h0, h_next


# ==============================================================================
# TRAINING & EVALUATION ROUTINE
# ==============================================================================

def train_swa_model(model_cls, kwargs, x_tr, u_tr, dt_tr, y_tr, seed=42, epochs=8, lr=3.5e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model_cls(**kwargs)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    N = x_tr.size(0)
    batch_size = 512
    swa_weights = None
    swa_count = 0
    t0 = time.time()
    
    for ep in range(epochs):
        perm = torch.randperm(N)
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            bx, bu, bdt, by = x_tr[idx], u_tr[idx], dt_tr[idx], y_tr[idx]
            y_pred, _, _ = model(bx, bu, bdt)
            diff = y_pred - by
            w = torch.where(diff > 0, 2.0, 1.0)
            loss = torch.mean(w * (diff ** 2))
            opt.zero_grad()
            loss.backward()
            opt.step()
            
        if ep >= (epochs - 3):
            if swa_weights is None:
                swa_weights = {k: v.clone() for k, v in model.state_dict().items()}
                swa_count = 1
            else:
                for k, v in model.state_dict().items():
                    swa_weights[k] += v
                swa_count += 1
                
    if swa_weights is not None:
        for k in swa_weights:
            swa_weights[k] /= float(swa_count)
        model.load_state_dict(swa_weights)
        
    return model, time.time() - t0


def run_mechanism_ablations():
    print("=" * 80)
    print("RUNNING MECHANISM ABLATIONS ON VALIDATION UNITS (10 REAL SEEDS)")
    print("=" * 80)
    
    os.makedirs(ROOT / "results" / "raw", exist_ok=True)
    raw_csv = ROOT / "results" / "raw" / "mechanism_ablation_raw.csv"
    
    df_tr, df_va, _, stats1 = load_cmapss_subset('FD001')
    
    # 7 model configurations for mechanism testing
    models_to_test = [
        ('Strang_Symplectic_Damped',   CT_PhysProb_JEPA_IntegratorAblation, dict(integrator='strang', damped=True,  has_physics=True)),
        ('Strang_Symplectic_Undamped', CT_PhysProb_JEPA_IntegratorAblation, dict(integrator='strang', damped=False, has_physics=True)),
        ('Euler_Damped',               CT_PhysProb_JEPA_IntegratorAblation, dict(integrator='euler',  damped=True,  has_physics=True)),
        ('Euler_Undamped',             CT_PhysProb_JEPA_IntegratorAblation, dict(integrator='euler',  damped=False, has_physics=True)),
        ('No_Physics_Ablation',        CT_PhysProb_JEPA_IntegratorAblation, dict(has_physics=False)),
        ('GRU_D_Canonical_SWA',        GRU_D_Fair_SWA,                      dict(use_hazard_head=False)),
        ('GRU_D_Enhanced_SWA',         GRU_D_Fair_SWA,                      dict(use_hazard_head=True)),
    ]
    
    records = []
    
    for s_idx, s in enumerate(SEEDS_10):
        t0_s = time.time()
        print(f"\n--- Running Seed {s} ({s_idx + 1}/10) ---", flush=True)
        
        # Training data with full mask (all 14 sensors present: mask = 1.0)
        tr_short = build_seq_data(df_tr, stats1, drop_prob=0.25, seed=s)
        tr_long  = build_seq_data(df_tr, stats1, fixed_stride_range=(3, 15), seed=s + 1000)
        tr_mixed = {
            k: np.concatenate([tr_short[k], tr_long[k]], axis=0)
            for k in ['obs_t', 'ctrl_t', 'dt', 'obs_next', 'y_norm', 'y_raw', 'mem_fast', 'mem_slow']
        }
        
        N_tr = tr_mixed['obs_t'].shape[0]
        mask_tr = np.ones((N_tr, 14), dtype=np.float32)
        x_tr_feat = np.concatenate([tr_mixed['obs_t'], tr_mixed['mem_fast'], tr_mixed['mem_slow'], mask_tr], axis=-1)
        x_tr = torch.tensor(x_tr_feat, dtype=torch.float32)
        u_tr = torch.tensor(tr_mixed['ctrl_t'], dtype=torch.float32)
        dt_tr = torch.tensor(tr_mixed['dt'], dtype=torch.float32)
        y_tr = torch.tensor(tr_mixed['y_norm'], dtype=torch.float32)
        
        # Train all 7 models on identical data
        trained_models = {}
        for m_name, m_cls, m_kw in models_to_test:
            m_inst, tr_sec = train_swa_model(m_cls, m_kw, x_tr, u_tr, dt_tr, y_tr, seed=s, epochs=8)
            m_inst.eval()
            trained_models[m_name] = m_inst
            
        # ----------------------------------------------------------------------
        # EVALUATION SUITE: 5 Stress Regimes
        # 1. Nominal OOD 10x-20x (0% drop)
        # 2. Dropout 10% (1-2 sensors missing)
        # 3. Dropout 20% (3 sensors missing)
        # 4. Dropout 30% (4 sensors missing)
        # 5. Extreme Outage 15x-25x
        # 6. Extreme Outage 20x-30x
        # ----------------------------------------------------------------------
        val_regimes_data = {
            'OOD_10x_20x':    build_seq_data(df_va, stats1, fixed_stride_range=(10, 20), seed=s + 4000),
            'Extreme_15x_25x': build_seq_data(df_va, stats1, fixed_stride_range=(15, 25), seed=s + 5000),
            'Extreme_20x_30x': build_seq_data(df_va, stats1, fixed_stride_range=(20, 30), seed=s + 6000),
        }
        
        # Evaluate Extreme Outages (Full Sensors)
        for r_name in ['OOD_10x_20x', 'Extreme_15x_25x', 'Extreme_20x_30x']:
            vd = val_regimes_data[r_name]
            N_val = vd['obs_t'].shape[0]
            mask_val = np.ones((N_val, 14), dtype=np.float32)
            vx_feat = np.concatenate([vd['obs_t'], vd['mem_fast'], vd['mem_slow'], mask_val], axis=-1)
            vx = torch.tensor(vx_feat, dtype=torch.float32)
            vu = torch.tensor(vd['ctrl_t'], dtype=torch.float32)
            vdt = torch.tensor(vd['dt'], dtype=torch.float32)
            y_true = vd['y_raw']
            dt_cyc = vd['dt'][:, 0] * 4.0
            y_curr = np.minimum(125.0, y_true + dt_cyc)
            true_slope = (y_true - y_curr) / np.maximum(1.0, dt_cyc)
            mask_act = (true_slope <= -0.90)
            
            for m_name, model in trained_models.items():
                with torch.no_grad():
                    p_norm, _, _ = model(vx, vu, vdt)
                    p_tot = np.clip(p_norm[:, 0].numpy() * stats1['tgt_std'] + stats1['tgt_mean'], 0.0, 125.0)
                tot_rmse = float(np.sqrt(np.mean((p_tot - y_true)**2)))
                bias_act = float(np.mean(p_tot[mask_act] - y_true[mask_act])) if np.sum(mask_act) > 0 else 0.0
                nasa = nasa_score(p_tot, y_true)
                
                records.append({
                    'Seed': s, 'Model': m_name, 'Regime': r_name, 'Dropout_Pct': 0,
                    'Total_RMSE': tot_rmse, 'Active_Bias': bias_act, 'NASA_Score': nasa
                })
                
        # Evaluate Sensor Dropouts on OOD 10x-20x at 10%, 20%, 30%
        vd_1020 = val_regimes_data['OOD_10x_20x']
        N_val = vd_1020['obs_t'].shape[0]
        y_true = vd_1020['y_raw']
        dt_cyc = vd_1020['dt'][:, 0] * 4.0
        y_curr = np.minimum(125.0, y_true + dt_cyc)
        true_slope = (y_true - y_curr) / np.maximum(1.0, dt_cyc)
        mask_act = (true_slope <= -0.90)
        vu = torch.tensor(vd_1020['ctrl_t'], dtype=torch.float32)
        vdt = torch.tensor(vd_1020['dt'], dtype=torch.float32)
        
        for drop_pct, num_drop in [(10, 1), (20, 3), (30, 4)]:
            np.random.seed(s + drop_pct * 100)
            drop_ch = np.random.choice(14, size=num_drop, replace=False)
            
            # Construct corrupted inputs WITH binary mask
            obs_corrupt = vd_1020['obs_t'].copy()
            fast_corrupt = vd_1020['mem_fast'].copy()
            slow_corrupt = vd_1020['mem_slow'].copy()
            mask_corrupt = np.ones((N_val, 14), dtype=np.float32)
            
            for ch in drop_ch:
                obs_corrupt[:, ch] = 0.0
                fast_corrupt[:, ch] = 0.0
                slow_corrupt[:, ch] = 0.0
                mask_corrupt[:, ch] = 0.0  # Explicit mask indicating channel is missing
                
            vx_corrupt_feat = np.concatenate([obs_corrupt, fast_corrupt, slow_corrupt, mask_corrupt], axis=-1)
            vx_corrupt = torch.tensor(vx_corrupt_feat, dtype=torch.float32)
            
            for m_name, model in trained_models.items():
                with torch.no_grad():
                    p_norm, _, _ = model(vx_corrupt, vu, vdt)
                    p_tot = np.clip(p_norm[:, 0].numpy() * stats1['tgt_std'] + stats1['tgt_mean'], 0.0, 125.0)
                tot_rmse = float(np.sqrt(np.mean((p_tot - y_true)**2)))
                bias_act = float(np.mean(p_tot[mask_act] - y_true[mask_act])) if np.sum(mask_act) > 0 else 0.0
                nasa = nasa_score(p_tot, y_true)
                
                records.append({
                    'Seed': s, 'Model': m_name, 'Regime': f'Dropout_{drop_pct}pct', 'Dropout_Pct': drop_pct,
                    'Total_RMSE': tot_rmse, 'Active_Bias': bias_act, 'NASA_Score': nasa
                })
                
        print(f"Seed {s} completed in {time.time() - t0_s:.1f}s.", flush=True)
        pd.DataFrame(records).to_csv(raw_csv, index=False)
        
    df_all = pd.DataFrame(records)
    df_all.to_csv(raw_csv, index=False)
    print(f"\nAll mechanism ablation results saved to {raw_csv}")
    return df_all

if __name__ == '__main__':
    run_mechanism_ablations()
