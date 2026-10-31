import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scripts.run_ood_and_small_data import load_cmapss_subset, build_seq_data, nasa_score
from scripts.run_step1_diagnosis import to_tensor_data, REGIMES
from scripts.run_step2_development import GRUD_TemporalEncoder

class CT_PhysProb_JEPA_PhaseWear(nn.Module):
    """
    Physical Phase-Space Degradation JEPA:
    - GRU-D temporal encoder yields initial latent state (q0, p0).
    - Anchor head predicts initial RUL bounded strictly in [0, 125].
    - Latent Hamiltonian ODE integrates through the outage dt.
    - At each substep, instantaneous wear rate w_t in [0, 1] is integrated.
    - y_pred = clamp(y_anchor - total_wear, 0, 125).
    """
    def __init__(self, in_features=42, u_dim=3, latent_dim=16, alpha=0.10, gamma0=0.15):
        super().__init__()
        self.d = latent_dim // 2
        self.alpha = alpha
        self.gamma0 = gamma0
        
        self.encoder = GRUD_TemporalEncoder(in_features, u_dim, latent_dim, hidden_dim=48)
        
        # Anchor probe bounded strictly in [0, 125]
        # In normalized units: tgt_mean=75, tgt_std=41.6 -> [0, 125] corresponds to [-1.80, +1.20]
        # We can predict raw anchor in [0, 125] using sigmoid
        self.anchor_probe = nn.Sequential(
            nn.Linear(latent_dim + u_dim, 48),
            nn.Tanh(),
            nn.Linear(48, 1),
            nn.Sigmoid()
        )
        
        # Hamiltonian ODE potential parameters
        self.W_pot = nn.Parameter(torch.randn(self.d, self.d) * 0.25)
        self.b_pot = nn.Parameter(torch.zeros(self.d))
        self.mode_signs = nn.Parameter(torch.tensor([1.0 if j % 3 != 0 else -0.65 for j in range(self.d)]), requires_grad=False)
        
        cond_in = latent_dim + u_dim + 1
        self.force_net = nn.Sequential(nn.Linear(cond_in, 32), nn.Tanh(), nn.Linear(32, self.d))
        
        # Instantaneous wear rate: function of latent state (q, p) and control u
        # Output strictly in [0, 1.25] (cycles/cycle)
        self.wear_rate_net = nn.Sequential(
            nn.Linear(latent_dim + u_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.decay_w = nn.Parameter(torch.tensor([0.25]))

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
        
        # Anchor prediction strictly in [0, 125] cycles
        y_anch_raw = 125.0 * self.anchor_probe(torch.cat([z0, u], dim=-1))
        
        # Substepping leapfrog integration
        gamma = self.gamma0 * (1.0 + torch.relu(self.decay_w * dt_eff))
        max_dt = float(dt_eff.max().item())
        m_steps = max(4, int(np.ceil(max_dt / 0.125)))
        sub_dt = dt_eff / float(m_steps)
        sub_dt_cyc = sub_dt * 4.0  # scaled to engine cycles
        
        q, p = q0.clone(), p0.clone()
        decay = torch.exp(-0.5 * gamma * sub_dt)
        
        total_wear = torch.zeros_like(y_anch_raw)
        
        for _ in range(m_steps):
            p = decay * p
            p = p + 0.5 * sub_dt * self.force_tot(q, q0, p, u, dt_eff)
            q = q + sub_dt * p
            p = p + 0.5 * sub_dt * self.force_tot(q, q0, p, u, dt_eff)
            p = decay * p
            
            # Instantaneous wear rate at current state
            z_t = torch.cat([q, p], dim=-1)
            inst_wear = self.wear_rate_net(torch.cat([z_t, u], dim=-1)) * 1.15
            total_wear = total_wear + inst_wear * sub_dt_cyc
            
        # Predicted RUL is anchor minus integrated wear, clipped to [0, 125]
        y_pred_raw = torch.clamp(y_anch_raw - total_wear, 0.0, 125.0)
        
        # Normalize to standard target scale for loss computation
        y_pred_norm = (y_pred_raw - 75.0) / 41.6
        y_anch_norm = (y_anch_raw - 75.0) / 41.6
        
        return y_pred_norm, y_anch_norm, z0

def test_prototype():
    df_tr, df_va, _, stats1 = load_cmapss_subset('FD001')
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    tr_short = build_seq_data(df_tr, stats1, drop_prob=0.25, seed=seed)
    tr_long  = build_seq_data(df_tr, stats1, fixed_stride_range=(3, 15), seed=seed + 1000)
    tr_mixed = {
        k: np.concatenate([tr_short[k], tr_long[k]], axis=0)
        for k in ['obs_t', 'ctrl_t', 'dt', 'obs_next', 'y_norm', 'y_raw', 'mem_fast', 'mem_slow']
    }
    x_tr, u_tr, dt_tr, y_tr = to_tensor_data(tr_mixed)
    
    val_ood = build_seq_data(df_va, stats1, fixed_stride_range=(10, 20), seed=seed + 4000)
    vx, vu, vdt, _ = to_tensor_data(val_ood)
    y_true = val_ood['y_raw']
    dt_cyc = val_ood['dt'][:, 0] * 4.0
    y_curr = np.minimum(125.0, y_true + dt_cyc)
    
    model = CT_PhysProb_JEPA_PhaseWear()
    opt = torch.optim.AdamW(model.parameters(), lr=3.5e-3, weight_decay=1e-4)
    
    N = x_tr.size(0)
    batch_size = 512
    epochs = 8
    
    print("Training PhaseWear prototype on Seed 42...")
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(N)
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            bx, bu, bdt, by = x_tr[idx], u_tr[idx], dt_tr[idx], y_tr[idx]
            
            y_pred, y_anch, _ = model(bx, bu, bdt)
            diff = y_pred - by
            w = torch.where(diff > 0, 2.0, 1.0)
            loss = torch.mean(w * (diff ** 2))
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
    print(f"Training completed in {time.time() - t0:.1f}s.")
    
    # Evaluate on OOD 10x-20x
    model.eval()
    with torch.no_grad():
        vdt_zero = torch.full_like(vdt, 0.25)
        _, p_anch_norm, _ = model(vx, vu, vdt_zero)
        p_anch = torch.clamp(p_anch_norm[:, 0] * stats1['tgt_std'] + stats1['tgt_mean'], 0.0, 125.0).numpy()
        
        p_tot_norm, _, _ = model(vx, vu, vdt)
        p_tot = torch.clamp(p_tot_norm[:, 0] * stats1['tgt_std'] + stats1['tgt_mean'], 0.0, 125.0).numpy()
        
    anch_rmse = float(np.sqrt(np.mean((p_anch - y_curr)**2)))
    tot_rmse = float(np.sqrt(np.mean((p_tot - y_true)**2)))
    tot_mae = float(np.mean(np.abs(p_tot - y_true)))
    nasa = nasa_score(p_tot, y_true)
    
    true_slope = (y_true - y_curr) / np.maximum(1.0, dt_cyc)
    mask_act = (true_slope <= -0.90)
    signed_bias = float(np.mean(p_tot[mask_act] - y_true[mask_act])) if np.sum(mask_act) > 0 else 0.0
    
    print("\n--- OOD 10x-20x Evaluation Results (Seed 42) ---")
    print(f"Anchor RMSE: {anch_rmse:.2f} cycles")
    print(f"Total RMSE:  {tot_rmse:.2f} cycles")
    print(f"Total MAE:   {tot_mae:.2f} cycles")
    print(f"Active Bias: {signed_bias:+.2f} cycles")
    print(f"NASA Score:  {nasa:.1f}")

if __name__ == '__main__':
    test_prototype()
