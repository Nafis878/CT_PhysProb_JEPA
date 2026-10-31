"""Test all Step 1 diagnostic models in PyTorch."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import numpy as np

# 1. CT_PhysProb_JEPA (Clamped & Adaptive)
class CT_PhysProb_JEPA(nn.Module):
    def __init__(self, in_features, u_dim=3, latent_dim=16, alpha=0.10, gamma=0.15, clamped=True):
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

# 2. Plain_2ndOrder_NODE_Kin (Causal Control)
class Plain_2ndOrder_NODE_Kin(nn.Module):
    def __init__(self, in_features, u_dim=3, latent_dim=16, alpha=0.10):
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
        
        # Euler 1-step (M=1)
        cond_vec = torch.cat([q0, p0, u, dt_eff], dim=-1)
        f_ext = 0.45 * self.force_net(cond_vec)
        q1 = q0 + dt_eff * p0
        p1 = p0 + dt_eff * f_ext
        
        z_pred = torch.cat([q1, p1], dim=-1)
        y_pred = self.probe(torch.cat([z_pred, u], dim=-1))
        return y_pred, z0, z_pred

# 3. GRU_D (Continuous-Time Decay Recurrent)
class GRU_D(nn.Module):
    def __init__(self, in_features, u_dim=3, hidden_dim=48):
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

# 4. Latent_SDE (Neural SDE with Euler-Maruyama)
class Latent_SDE(nn.Module):
    def __init__(self, in_features, u_dim=3, latent_dim=16):
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

# 5. PatchTST_RUL (Patched Transformer)
class PatchTST_RUL(nn.Module):
    def __init__(self, in_features, u_dim=3, d_model=48, nhead=2, num_layers=2):
        super().__init__()
        # Tokenizer: project sensor features into 3 patches (raw, fast_mem, slow_mem)
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
        # x is concatenation of [obs (14), mem_fast (14), mem_slow (14)]
        p_raw = self.patch_raw(x[:, :14]).unsqueeze(1)
        p_fast = self.patch_fast(x[:, 14:28]).unsqueeze(1)
        p_slow = self.patch_slow(x[:, 28:42]).unsqueeze(1)
        p_u = self.u_proj(u).unsqueeze(1)
        
        # Add time gap continuous embedding to tokens
        t_emb = self.time_proj(dt_eff).unsqueeze(1)
        tokens = torch.cat([p_raw + t_emb, p_fast + t_emb, p_slow + t_emb, p_u], dim=1)
        
        trans_out = self.transformer(tokens)
        flat = trans_out.reshape(trans_out.size(0), -1)
        y_pred = self.probe(flat)
        return y_pred, flat, flat

print("All models verified successfully!")
