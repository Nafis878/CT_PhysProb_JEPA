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

class TwoStatePhysicsJEPA(nn.Module):
    """
    Two-State Physical Mixture JEPA:
    Explicitly models the two thermodynamic phases of a machine:
    Phase 0: Healthy Plateau (RUL = 125.0, wear rate = 0 until knee)
    Phase 1: Active Degradation (RUL < 125.0, wear rate = 1.0 cyc/cyc)
    """
    def __init__(self, in_features=42, u_dim=3, latent_dim=16, alpha=0.10, gamma0=0.15):
        super().__init__()
        self.d = latent_dim // 2
        self.alpha = alpha
        self.gamma0 = gamma0
        
        self.encoder = GRUD_TemporalEncoder(in_features, u_dim, latent_dim, hidden_dim=48)
        
        # Phase classifier: probability that engine is on healthy plateau at t_k
        self.p_healthy_net = nn.Sequential(
            nn.Linear(latent_dim + u_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Degraded anchor RUL in [0, 125]
        self.deg_anchor_net = nn.Sequential(
            nn.Linear(latent_dim + u_dim, 48),
            nn.Tanh(),
            nn.Linear(48, 1),
            nn.Sigmoid()
        )
        
        # Time-to-knee estimate (in cycles) for healthy engines
        self.tau_knee_net = nn.Sequential(
            nn.Linear(latent_dim + u_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Softplus()
        )
        
        # Latent Hamiltonian potential & dynamics
        self.W_pot = nn.Parameter(torch.randn(self.d, self.d) * 0.25)
        self.b_pot = nn.Parameter(torch.zeros(self.d))
        self.mode_signs = nn.Parameter(torch.tensor([1.0 if j % 3 != 0 else -0.65 for j in range(self.d)]), requires_grad=False)
        self.force_net = nn.Sequential(nn.Linear(latent_dim + u_dim + 1, 32), nn.Tanh(), nn.Linear(32, self.d))
        self.dyn_probe = nn.Sequential(nn.Linear(latent_dim + u_dim + 1, 32), nn.Tanh(), nn.Linear(32, 1))
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
        dt_cyc = dt_eff * 4.0
        
        z0, _ = self.encoder(x, u, dt_eff)
        q0 = z0[:, :self.d]
        p0 = z0[:, self.d:]
        
        feat0 = torch.cat([z0, u], dim=-1)
        p_healthy = self.p_healthy_net(feat0)
        y_deg = 125.0 * self.deg_anchor_net(feat0)
        tau_knee = self.tau_knee_net(feat0) * 10.0  # expected time to knee in cycles
        
        # Initial anchor expectation
        y_anch_raw = p_healthy * 125.0 + (1.0 - p_healthy) * y_deg
        
        # Physical wear across gap dt:
        # If healthy: wear accumulates only after tau_knee: relu(dt_cyc - tau_knee)
        # If degraded: wear accumulates as dt_cyc
        wear_healthy = torch.relu(dt_cyc - tau_knee)
        wear_degraded = dt_cyc
        
        exp_wear = p_healthy * wear_healthy + (1.0 - p_healthy) * wear_degraded
        
        # Hamiltonian ODE residual correction
        gamma = self.gamma0 * (1.0 + torch.relu(self.decay_w * dt_eff))
        max_dt = float(dt_eff.max().item())
        m_steps = max(4, int(np.ceil(max_dt / 0.125)))
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
        ode_res = 5.0 * self.dyn_probe(torch.cat([z_pred - z0, u, dt_eff], dim=-1))
        
        # Predicted RUL
        y_pred_raw = torch.clamp(y_anch_raw - exp_wear + ode_res, 0.0, 125.0)
        
        y_pred_norm = (y_pred_raw - 75.0) / 41.6
        y_anch_norm = (y_anch_raw - 75.0) / 41.6
        return y_pred_norm, y_anch_norm, z0

def run_test():
    df_tr, df_va, _, stats1 = load_cmapss_subset('FD001')
    seeds = [42, 100]
    
    for s in seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        
        tr_short = build_seq_data(df_tr, stats1, drop_prob=0.25, seed=s)
        tr_long  = build_seq_data(df_tr, stats1, fixed_stride_range=(3, 15), seed=s + 1000)
        tr_mixed = {k: np.concatenate([tr_short[k], tr_long[k]], axis=0) for k in ['obs_t', 'ctrl_t', 'dt', 'obs_next', 'y_norm', 'y_raw', 'mem_fast', 'mem_slow']}
        x_tr, u_tr, dt_tr, y_tr = to_tensor_data(tr_mixed)
        
        model = TwoStatePhysicsJEPA()
        opt = torch.optim.AdamW(model.parameters(), lr=3.5e-3, weight_decay=1e-4)
        
        N = x_tr.size(0)
        batch_size = 512
        epochs = 8
        
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
                
        print(f"Seed {s} trained in {time.time() - t0:.1f}s.")
        
        # Evaluate on OOD 10x-20x
        val_ood = build_seq_data(df_va, stats1, fixed_stride_range=(10, 20), seed=s + 4000)
        vx, vu, vdt, _ = to_tensor_data(val_ood)
        y_true = val_ood['y_raw']
        dt_cyc = val_ood['dt'][:, 0] * 4.0
        y_curr = np.minimum(125.0, y_true + dt_cyc)
        
        model.eval()
        with torch.no_grad():
            vdt_zero = torch.full_like(vdt, 0.25)
            _, p_anch_norm, _ = model(vx, vu, vdt_zero)
            p_anch = np.clip(p_anch_norm[:, 0].numpy() * stats1['tgt_std'] + stats1['tgt_mean'], 0.0, 125.0)
            
            p_tot_norm, _, _ = model(vx, vu, vdt)
            p_tot = np.clip(p_tot_norm[:, 0].numpy() * stats1['tgt_std'] + stats1['tgt_mean'], 0.0, 125.0)
            
        anch_rmse = float(np.sqrt(np.mean((p_anch - y_curr)**2)))
        tot_rmse = float(np.sqrt(np.mean((p_tot - y_true)**2)))
        tot_mae = float(np.mean(np.abs(p_tot - y_true)))
        nasa = nasa_score(p_tot, y_true)
        
        true_slope = (y_true - y_curr) / np.maximum(1.0, dt_cyc)
        mask_act = (true_slope <= -0.90)
        signed_bias = float(np.mean(p_tot[mask_act] - y_true[mask_act])) if np.sum(mask_act) > 0 else 0.0
        
        print(f"Seed {s} OOD 10x-20x: Anchor RMSE={anch_rmse:.2f}, Total RMSE={tot_rmse:.2f}, MAE={tot_mae:.2f}, Bias={signed_bias:+.2f}, NASA={nasa:.1f}")

if __name__ == '__main__':
    run_test()
