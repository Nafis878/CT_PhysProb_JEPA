import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.run_ood_and_small_data import load_cmapss_subset, build_seq_data, nasa_score
from scripts.run_step1_diagnosis import to_tensor_data, REGIMES
from scripts.run_step2_development import GRUD_TemporalEncoder

class PortHamiltonian_JEPA(nn.Module):
    """
    Port-Hamiltonian Joint-Embedding Predictive Architecture (pH-JEPA):
    - Context Encoder E_c(x(t_k)) -> z0 = [q0, p0]
    - Target Encoder E_t(x(t_{k+1})) -> z_tgt (updated via EMA / shared)
    - Port-Hamiltonian Integrator:
        dq/dt = p
        dp/dt = -grad_q V(q) - R(q,p)*p + g(u)
        where R(q,p) >= 0 guarantees physical dissipation (dH/dt <= p^T g(u))
    - Symplectic Strang splitting preserves phase space over long horizons dt
    - Self-supervised JEPA loss: ||z_pred - stop_grad(z_tgt)||^2
    - RUL probe decodes RUL directly from the evolved physical state z_pred.
    """
    def __init__(self, in_features=42, u_dim=3, latent_dim=16, alpha=0.10):
        super().__init__()
        self.d = latent_dim // 2
        self.alpha = alpha
        
        # Context & Target Encoders
        self.encoder = GRUD_TemporalEncoder(in_features, u_dim, latent_dim, hidden_dim=48)
        self.target_encoder = GRUD_TemporalEncoder(in_features, u_dim, latent_dim, hidden_dim=48)
        # Initialize target encoder weights to match context encoder
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        for param in self.target_encoder.parameters():
            param.requires_grad = False
            
        # Port-Hamiltonian parameters:
        # Potential V(q) MLP
        self.V_net = nn.Sequential(
            nn.Linear(self.d, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        # Harmonizer potential weights
        self.omega = nn.Parameter(torch.ones(self.d) * 0.5)
        
        # Dissipation Matrix R(q, p) >= 0 (strictly positive semi-definite)
        self.damping_net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.Tanh(),
            nn.Linear(32, self.d),
            nn.Softplus()  # strictly positive damping
        )
        
        # Port input g(u): external forces / operating conditions
        self.port_net = nn.Sequential(
            nn.Linear(u_dim, 32),
            nn.Tanh(),
            nn.Linear(32, self.d)
        )
        
        # RUL Probe decoding RUL from physical latent state
        self.rul_probe = nn.Sequential(
            nn.Linear(latent_dim + u_dim, 48),
            nn.Tanh(),
            nn.Linear(48, 1)
        )
        self.ema_decay = 0.99

    def grad_V(self, q):
        # Analytical gradient of potential V(q)
        with torch.enable_grad():
            q_var = q.clone().detach().requires_grad_(True)
            v_mlp = self.V_net(q_var)
            v_quad = 0.5 * torch.sum((self.omega ** 2) * (q_var ** 2), dim=-1, keepdim=True)
            v_tot = v_mlp + v_quad
            grad = torch.autograd.grad(v_tot.sum(), q_var, create_graph=self.training)[0]
        return grad

    def update_target_encoder(self):
        with torch.no_grad():
            for param_c, param_t in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                param_t.data.mul_(self.ema_decay).add_(param_c.data, alpha=1.0 - self.ema_decay)

    def integrate_ph(self, q0, p0, u, dt_eff):
        # Symplectic Strang splitting with positive dissipation R(q, p)
        g_ext = 0.25 * self.port_net(u)
        max_dt = float(dt_eff.max().item())
        m_steps = max(4, int(np.ceil(max_dt / 0.125)))
        sub_dt = dt_eff / float(m_steps)
        
        q, p = q0.clone(), p0.clone()
        for _ in range(m_steps):
            # 1. Half-step dissipation damping: p = exp(-0.5 * R * dt) * p
            z_curr = torch.cat([q, p], dim=-1)
            R_diag = self.damping_net(z_curr) + 1e-4
            damp_half = torch.exp(-0.5 * R_diag * sub_dt)
            p = damp_half * p
            
            # 2. Half-step force kick: p = p + 0.5 * sub_dt * (-grad_V + g_ext)
            force = -self.grad_V(q) + g_ext
            p = p + 0.5 * sub_dt * force
            
            # 3. Full-step position drift: q = q + sub_dt * p
            q = q + sub_dt * p
            
            # 4. Half-step force kick
            force = -self.grad_V(q) + g_ext
            p = p + 0.5 * sub_dt * force
            
            # 5. Half-step dissipation damping
            z_curr = torch.cat([q, p], dim=-1)
            R_diag = self.damping_net(z_curr) + 1e-4
            damp_half = torch.exp(-0.5 * R_diag * sub_dt)
            p = damp_half * p
            
        z_pred = torch.cat([q, p], dim=-1)
        return z_pred

    def forward(self, x_t, u_t, dt, x_next=None):
        dt_eff = torch.clamp(dt, 0.01, 10.0)
        z0, _ = self.encoder(x_t, u_t, dt_eff)
        q0 = z0[:, :self.d]
        p0 = z0[:, self.d:]
        
        # Integrate Port-Hamiltonian dynamics forward across dt
        z_pred = self.integrate_ph(q0, p0, u_t, dt_eff)
        
        # Decode RUL prediction at t + dt
        y_pred = self.rul_probe(torch.cat([z_pred, u_t], dim=-1))
        
        # If target observations provided (during training), compute target embedding
        z_tgt = None
        if x_next is not None:
            with torch.no_grad():
                z_tgt, _ = self.target_encoder(x_next, u_t, torch.full_like(dt_eff, 0.25))
                
        return y_pred, z0, z_pred, z_tgt

def test_ph_jepa():
    print("=" * 70)
    print("TESTING PORT-HAMILTONIAN JEPA (pH-JEPA) ON SEED 42")
    print("=" * 70)
    
    df_tr, df_va, _, stats1 = load_cmapss_subset('FD001')
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    tr_short = build_seq_data(df_tr, stats1, drop_prob=0.25, seed=seed)
    tr_long  = build_seq_data(df_tr, stats1, fixed_stride_range=(3, 15), seed=seed + 1000)
    
    # Notice: tr_mixed contains both obs_t (current) and obs_next (true future observation)!
    tr_mixed = {
        k: np.concatenate([tr_short[k], tr_long[k]], axis=0)
        for k in ['obs_t', 'ctrl_t', 'dt', 'obs_next', 'y_norm', 'y_raw', 'mem_fast', 'mem_slow']
    }
    
    x_tr = torch.tensor(np.concatenate([tr_mixed['obs_t'], tr_mixed['mem_fast'], tr_mixed['mem_slow']], axis=-1), dtype=torch.float32)
    x_next_tr = torch.tensor(np.concatenate([tr_mixed['obs_next'], tr_mixed['mem_fast'], tr_mixed['mem_slow']], axis=-1), dtype=torch.float32)
    u_tr = torch.tensor(tr_mixed['ctrl_t'], dtype=torch.float32)
    dt_tr = torch.tensor(tr_mixed['dt'], dtype=torch.float32)
    y_tr = torch.tensor(tr_mixed['y_norm'], dtype=torch.float32)
    
    model = PortHamiltonian_JEPA()
    opt = torch.optim.AdamW(model.parameters(), lr=3.5e-3, weight_decay=1e-4)
    
    N = x_tr.size(0)
    batch_size = 512
    epochs = 8
    lambda_jepa = 0.5  # Weight for self-supervised JEPA predictive loss
    
    print("Training pH-JEPA with joint RUL + JEPA representation loss...")
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(N)
        ep_loss = 0.0
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            bx, bx_next, bu, bdt, by = x_tr[idx], x_next_tr[idx], u_tr[idx], dt_tr[idx], y_tr[idx]
            
            y_pred, z0, z_pred, z_tgt = model(bx, bu, bdt, x_next=bx_next)
            
            # Supervised Asymmetric RUL loss
            diff = y_pred - by
            w = torch.where(diff > 0, 2.0, 1.0)
            loss_rul = torch.mean(w * (diff ** 2))
            
            # Self-Supervised Port-Hamiltonian JEPA Predictive loss:
            # Predictor must match the future target embedding
            loss_jepa = F.mse_loss(z_pred, z_tgt)
            
            # Total multi-task loss
            loss = loss_rul + lambda_jepa * loss_jepa
            
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()
            
            # Update target encoder EMA
            model.update_target_encoder()
            ep_loss += loss.item()
            
        print(f"Epoch {ep+1}/{epochs} loss: {ep_loss / (N / batch_size):.4f}")
        
    print(f"Training completed in {time.time() - t0:.1f}s.")
    
    # Evaluate on OOD 10x-20x validation
    val_ood = build_seq_data(df_va, stats1, fixed_stride_range=(10, 20), seed=seed + 4000)
    vx = torch.tensor(np.concatenate([val_ood['obs_t'], val_ood['mem_fast'], val_ood['mem_slow']], axis=-1), dtype=torch.float32)
    vu = torch.tensor(val_ood['ctrl_t'], dtype=torch.float32)
    vdt = torch.tensor(val_ood['dt'], dtype=torch.float32)
    y_true = val_ood['y_raw']
    dt_cyc = val_ood['dt'][:, 0] * 4.0
    y_curr = np.minimum(125.0, y_true + dt_cyc)
    
    model.eval()
    with torch.no_grad():
        vdt_zero = torch.full_like(vdt, 0.25)
        p_anch_norm, _, _, _ = model(vx, vu, vdt_zero)
        p_anch = np.clip(p_anch_norm[:, 0].numpy() * stats1['tgt_std'] + stats1['tgt_mean'], 0.0, 125.0)
        
        p_tot_norm, _, _, _ = model(vx, vu, vdt)
        p_tot = np.clip(p_tot_norm[:, 0].numpy() * stats1['tgt_std'] + stats1['tgt_mean'], 0.0, 125.0)
        
    anch_rmse = float(np.sqrt(np.mean((p_anch - y_curr)**2)))
    tot_rmse = float(np.sqrt(np.mean((p_tot - y_true)**2)))
    tot_mae = float(np.mean(np.abs(p_tot - y_true)))
    nasa = nasa_score(p_tot, y_true)
    
    true_slope = (y_true - y_curr) / np.maximum(1.0, dt_cyc)
    mask_act = (true_slope <= -0.90)
    signed_bias = float(np.mean(p_tot[mask_act] - y_true[mask_act])) if np.sum(mask_act) > 0 else 0.0
    
    print("\n--- pH-JEPA OOD 10x-20x Evaluation Results (Seed 42) ---")
    print(f"Anchor RMSE: {anch_rmse:.2f} cycles")
    print(f"Total RMSE:  {tot_rmse:.2f} cycles")
    print(f"Total MAE:   {tot_mae:.2f} cycles")
    print(f"Active Bias: {signed_bias:+.2f} cycles")
    print(f"NASA Score:  {nasa:.1f}")

if __name__ == '__main__':
    test_ph_jepa()
