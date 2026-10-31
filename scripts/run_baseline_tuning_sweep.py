#!/usr/bin/env python3
"""
BASELINE HYPERPARAMETER TUNING SWEEP (VALIDATION ENGINES ONLY, EQUAL BUDGET)
- Evaluates GRU_D_Enhanced_SWA, GRU_D_Canonical_SWA, PatchTST_RUL_v2, Plain_NODE_Enhanced_SWA.
- Grid: lr in [1.5e-3, 3.5e-3, 5.0e-3] x hidden_size in [32, 64] (6 trials per model, 24 total).
- Development units only: FD001 training pool and FD001 validation engines.
- Primary selection criterion: OOD 10x-20x Total RMSE on FD001 validation engines.
- Logs all trials to results/raw/baseline_sweep_raw.csv.
- Saves best configuration per model to configs/baseline_tuned_configurations.json.
- Note: Proposed CT_PhysProb_JEPA_v2_SWA is NOT retuned.
"""

import os
import sys
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scripts.run_ood_and_small_data import load_cmapss_subset, build_seq_data
from scripts.run_mechanism_ablations import KneeHazardHead, GRUD_TemporalEncoder

# ------------------------------------------------------------------------------
# Model Architectures with Configurable Hyperparameters
# ------------------------------------------------------------------------------

class GRU_D_Enhanced_Configurable(nn.Module):
    def __init__(self, in_features=56, u_dim=3, hidden_dim=48):
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

class GRU_D_Canonical_Configurable(nn.Module):
    def __init__(self, in_features=56, u_dim=3, hidden_dim=48):
        super().__init__()
        self.decay_h = nn.Sequential(nn.Linear(1, hidden_dim), nn.Softplus())
        self.decay_x = nn.Sequential(nn.Linear(1, in_features), nn.Softplus())
        self.enc_h0 = nn.Sequential(nn.Linear(in_features + u_dim, hidden_dim), nn.Tanh())
        self.gru_cell = nn.GRUCell(in_features + u_dim, hidden_dim)
        self.probe = nn.Sequential(nn.Linear(hidden_dim + u_dim + 1, 48), nn.Tanh(), nn.Linear(48, 1))

    def forward(self, x, u, dt):
        dt_eff = torch.clamp(dt, 0.01, 10.0)
        h0 = self.enc_h0(torch.cat([x, u], dim=-1))
        gamma_h = torch.exp(-self.decay_h(dt_eff))
        gamma_x = torch.exp(-self.decay_x(dt_eff))
        h_decayed = gamma_h * h0
        x_imputed = gamma_x * x
        h_next = self.gru_cell(torch.cat([x_imputed, u], dim=-1), h_decayed)
        y_pred = self.probe(torch.cat([h_next, u, dt_eff], dim=-1))
        return y_pred, h0, h_next

class Plain_NODE_Enhanced_Configurable(nn.Module):
    def __init__(self, in_features=56, u_dim=3, latent_dim=16):
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

class PatchTST_RUL_Configurable(nn.Module):
    def __init__(self, in_features=56, u_dim=3, d_model=48, nhead=2, num_layers=2):
        super().__init__()
        self.patch_raw = nn.Linear(14, d_model)
        self.patch_fast = nn.Linear(14, d_model)
        self.patch_slow = nn.Linear(14, d_model)
        self.patch_mask = nn.Linear(14, d_model)
        self.u_proj = nn.Linear(u_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=64, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = KneeHazardHead(d_model * 5, u_dim)

    def forward(self, x, u, dt):
        dt_eff = torch.clamp(dt, 0.01, 10.0)
        p_raw = self.patch_raw(x[:, :14]).unsqueeze(1)
        p_fast = self.patch_fast(x[:, 14:28]).unsqueeze(1)
        p_slow = self.patch_slow(x[:, 28:42]).unsqueeze(1)
        p_mask = self.patch_mask(x[:, 42:56]).unsqueeze(1)
        u_t = self.u_proj(u).unsqueeze(1)
        tokens = torch.cat([p_raw, p_fast, p_slow, p_mask, u_t], dim=1)
        trans_out = self.transformer(tokens)
        flat = trans_out.reshape(trans_out.size(0), -1)
        
        y_anch_norm, count_cyc, _ = self.head(flat, u, dt_eff)
        count_norm = count_cyc / 41.6
        y_pred = y_anch_norm - count_norm
        return y_pred, flat, flat

# ------------------------------------------------------------------------------
# Training and Evaluation Function
# ------------------------------------------------------------------------------

def train_and_eval(model_cls, kwargs, lr, tr_data, va_data, stats_dict, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = model_cls(**kwargs)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    x_tr = tr_data['x']
    u_tr = tr_data['u']
    dt_tr = tr_data['dt']
    y_tr = tr_data['y']
    N_tr = x_tr.size(0)
    batch_size = 512
    epochs = 8
    
    swa_weights = None
    swa_count = 0
    
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(N_tr)
        for i in range(0, N_tr, batch_size):
            idx = perm[i:i+batch_size]
            bx, bu, bdt, by = x_tr[idx], u_tr[idx], dt_tr[idx], y_tr[idx]
            y_pred, _, _ = model(bx, bu, bdt)
            diff = y_pred - by
            w = torch.where(diff > 0, 2.0, 1.0)
            loss = torch.mean(w * (diff ** 2))
            opt.zero_grad()
            loss.backward()
            opt.step()
            
        if ep >= 4:  # Epochs 5, 6, 7 (0-indexed: 4, 5, 6, 7)
            state = {k: v.clone() for k, v in model.state_dict().items()}
            if swa_weights is None:
                swa_weights = state
            else:
                for k in swa_weights:
                    swa_weights[k] += state[k]
            swa_count += 1
            
    if swa_weights is not None and swa_count > 0:
        for k in swa_weights:
            swa_weights[k] /= float(swa_count)
        model.load_state_dict(swa_weights)
        
    # Evaluate on FD001 validation set under OOD 10x-20x
    model.eval()
    with torch.no_grad():
        y_pred_norm, _, _ = model(va_data['x'], va_data['u'], va_data['dt'])
        pred = np.clip(y_pred_norm[:, 0].numpy() * stats_dict['tgt_std'] + stats_dict['tgt_mean'], 0.0, 125.0)
        true = va_data['y_true']
        rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
        
        # Active bias
        act_mask = (true < 124.9)
        bias_act = float(np.mean(pred[act_mask] - true[act_mask]))
        
    return rmse, bias_act

def run_baseline_sweep():
    print("=" * 80)
    print("BASELINE HYPERPARAMETER TUNING SWEEP (FD001 VALIDATION ONLY)")
    print("=" * 80)
    
    df_tr, df_va, _, stats1 = load_cmapss_subset('FD001')
    
    # Build mixed-gap training data
    tr_short = build_seq_data(df_tr, stats1, drop_prob=0.25, seed=42)
    tr_long  = build_seq_data(df_tr, stats1, fixed_stride_range=(3, 15), seed=1042)
    tr_mixed = {
        k: np.concatenate([tr_short[k], tr_long[k]], axis=0)
        for k in ['obs_t', 'ctrl_t', 'dt', 'obs_next', 'y_norm', 'y_raw', 'mem_fast', 'mem_slow']
    }
    N_tr = tr_mixed['obs_t'].shape[0]
    mask_tr = np.ones((N_tr, 14), dtype=np.float32)
    tr_data = {
        'x': torch.tensor(np.concatenate([tr_mixed['obs_t'], tr_mixed['mem_fast'], tr_mixed['mem_slow'], mask_tr], axis=-1), dtype=torch.float32),
        'u': torch.tensor(tr_mixed['ctrl_t'], dtype=torch.float32),
        'dt': torch.tensor(tr_mixed['dt'], dtype=torch.float32),
        'y': torch.tensor(tr_mixed['y_norm'], dtype=torch.float32)
    }
    
    # Build validation OOD 10x-20x data
    val_sparse = build_seq_data(df_va, stats1, fixed_stride_range=(10, 20), seed=4042)
    N_va = val_sparse['obs_t'].shape[0]
    mask_va = np.ones((N_va, 14), dtype=np.float32)
    va_data = {
        'x': torch.tensor(np.concatenate([val_sparse['obs_t'], val_sparse['mem_fast'], val_sparse['mem_slow'], mask_va], axis=-1), dtype=torch.float32),
        'u': torch.tensor(val_sparse['ctrl_t'], dtype=torch.float32),
        'dt': torch.tensor(val_sparse['dt'], dtype=torch.float32),
        'y_true': val_sparse['y_raw']
    }
    
    models = {
        'GRU_D_Enhanced_SWA': {
            'class': GRU_D_Enhanced_Configurable,
            'param_name': 'hidden_dim',
            'sizes': [32, 64]
        },
        'GRU_D_Canonical_SWA': {
            'class': GRU_D_Canonical_Configurable,
            'param_name': 'hidden_dim',
            'sizes': [32, 64]
        },
        'Plain_NODE_Enhanced_SWA': {
            'class': Plain_NODE_Enhanced_Configurable,
            'param_name': 'latent_dim',
            'sizes': [12, 24]
        },
        'PatchTST_RUL_v2': {
            'class': PatchTST_RUL_Configurable,
            'param_name': 'd_model',
            'sizes': [32, 64]
        }
    }
    
    lrs = [1.5e-3, 3.5e-3, 5.0e-3]
    
    records = []
    best_configs = {}
    
    for m_name, m_spec in models.items():
        print(f"\n--- Sweeping {m_name} (Equal Budget: 6 trials) ---")
        best_rmse = float('inf')
        best_cfg = None
        
        for sz in m_spec['sizes']:
            for lr in lrs:
                kwargs = {'in_features': 56, 'u_dim': 3, m_spec['param_name']: sz}
                
                # Evaluate across 2 validation seeds for robust selection
                rmses, biases = [], []
                for s in [42, 100]:
                    r, b = train_and_eval(m_spec['class'], kwargs, lr, tr_data, va_data, stats1, seed=s)
                    rmses.append(r)
                    biases.append(b)
                mean_rmse = np.mean(rmses)
                mean_bias = np.mean(biases)
                
                print(f"  {m_spec['param_name']}={sz:2d}, lr={lr:.1e} => Val RMSE = {mean_rmse:.3f} cycles, Bias = {mean_bias:+.2f}")
                
                rec = {
                    'model': m_name,
                    'param_name': m_spec['param_name'],
                    'size': sz,
                    'lr': lr,
                    'val_rmse_mean': float(mean_rmse),
                    'val_bias_mean': float(mean_bias),
                    'seed_42_rmse': float(rmses[0]),
                    'seed_100_rmse': float(rmses[1])
                }
                records.append(rec)
                
                if mean_rmse < best_rmse:
                    best_rmse = mean_rmse
                    best_cfg = {
                        'model': m_name,
                        'best_param': m_spec['param_name'],
                        'best_size': sz,
                        'best_lr': lr,
                        'best_val_rmse': float(mean_rmse),
                        'best_val_bias': float(mean_bias)
                    }
        best_configs[m_name] = best_cfg
        print(f"  --> BEST for {m_name}: {best_cfg['best_param']}={best_cfg['best_size']}, lr={best_cfg['best_lr']:.1e} (Val RMSE = {best_cfg['best_val_rmse']:.3f} cycles)")
        
    df_records = pd.DataFrame(records)
    Path('results/raw').mkdir(parents=True, exist_ok=True)
    df_records.to_csv('results/raw/baseline_sweep_raw.csv', index=False)
    print(f"\nSaved all 24 trials to results/raw/baseline_sweep_raw.csv")
    
    Path('configs').mkdir(parents=True, exist_ok=True)
    with open('configs/baseline_tuned_configurations.json', 'w') as f:
        json.dump(best_configs, f, indent=2)
    print("Saved best configurations to configs/baseline_tuned_configurations.json")
    
    return best_configs

if __name__ == '__main__':
    run_baseline_sweep()
