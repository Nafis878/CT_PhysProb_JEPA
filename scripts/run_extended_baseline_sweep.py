#!/usr/bin/env python3
"""
EXTENDED BASELINE HYPERPARAMETER TUNING SWEEP (5 VALIDATION SEEDS, EQUAL BUDGET)
Models evaluated (7 models, equal budget per model):
- Modern/Enhanced baselines: GRU_D_Enhanced_SWA, GRU_D_Canonical_SWA, Plain_NODE_Enhanced_SWA, PatchTST_RUL_v2
- Original-recipe baselines: GRU_D, PatchTST_RUL, Plain_2ndOrder_NODE_Kin
- Proposed model: CT_PhysProb_JEPA_v2_SWA (FROZEN, not retuned, evaluated on same 5 seeds)

Validation seeds: [42, 100, 101, 102, 103] on FD001 validation engines.
Grid widened with lr in {1.0e-3, 2.5e-3, 5.0e-3, 7.5e-3} and 3 sizes per model so no best setting sits at an edge.
"""

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
from scripts.run_mechanism_ablations import KneeHazardHead, GRUD_TemporalEncoder, CT_PhysProb_JEPA_IntegratorAblation
from scripts.run_step1_diagnosis import GRU_D as GRU_D_Original, PatchTST_RUL as PatchTST_Original, Plain_2ndOrder_NODE_Kin as NODE_Kin_Original

SEEDS_5 = [42, 100, 101, 102, 103]

# ------------------------------------------------------------------------------
# Configurable Baseline Models
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
# Configurable Original-Recipe Baselines
# ------------------------------------------------------------------------------

class GRU_D_Original_Configurable(nn.Module):
    def __init__(self, in_features=42, u_dim=3, hidden_dim=48):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.decay_h = nn.Sequential(nn.Linear(1, hidden_dim), nn.Softplus())
        self.decay_x = nn.Sequential(nn.Linear(1, in_features), nn.Softplus())
        self.enc_h0 = nn.Sequential(nn.Linear(in_features + u_dim, hidden_dim), nn.Tanh())
        self.gru_cell = nn.GRUCell(in_features + u_dim, hidden_dim)
        self.probe = nn.Sequential(nn.Linear(hidden_dim + u_dim, 48), nn.Tanh(), nn.Linear(48, 1))

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

class PatchTST_Original_Configurable(nn.Module):
    def __init__(self, in_features=42, u_dim=3, d_model=48, nhead=2, num_layers=2):
        super().__init__()
        self.patch_raw = nn.Linear(14, d_model)
        self.patch_fast = nn.Linear(14, d_model)
        self.patch_slow = nn.Linear(14, d_model)
        self.u_proj = nn.Linear(u_dim, d_model)
        self.time_proj = nn.Sequential(nn.Linear(1, d_model), nn.Tanh())
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=64, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.probe = nn.Sequential(nn.Linear(d_model * 4, 48), nn.Tanh(), nn.Linear(48, 1))

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

class Plain_NODE_Original_Configurable(nn.Module):
    def __init__(self, in_features=42, u_dim=3, latent_dim=16, alpha=0.10):
        super().__init__()
        self.d = latent_dim // 2
        self.alpha = alpha
        self.encoder = nn.Sequential(nn.Linear(in_features, 48), nn.Tanh(), nn.Linear(48, latent_dim))
        cond_in = latent_dim + u_dim + 1
        self.force_net = nn.Sequential(nn.Linear(cond_in, 32), nn.Tanh(), nn.Linear(32, self.d))
        self.probe = nn.Sequential(nn.Linear(latent_dim + u_dim, 48), nn.Tanh(), nn.Linear(48, 1))

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

# ------------------------------------------------------------------------------
# Training Routine
# ------------------------------------------------------------------------------

def train_eval_trial(model_cls, kwargs, lr, tr_data, va_data, stats_dict, seed, use_swa=True):
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
            
        if use_swa and ep >= 4:
            state = {k: v.clone() for k, v in model.state_dict().items()}
            if swa_weights is None:
                swa_weights = state
            else:
                for k in swa_weights:
                    swa_weights[k] += state[k]
            swa_count += 1
            
    if use_swa and swa_weights is not None and swa_count > 0:
        for k in swa_weights:
            swa_weights[k] /= float(swa_count)
        model.load_state_dict(swa_weights)
        
    model.eval()
    with torch.no_grad():
        y_pred_norm, _, _ = model(va_data['x'], va_data['u'], va_data['dt'])
        pred = np.clip(y_pred_norm[:, 0].numpy() * stats_dict['tgt_std'] + stats_dict['tgt_mean'], 0.0, 125.0)
        true = va_data['y_true']
        rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
        act_mask = (true < 124.9)
        bias_act = float(np.mean(pred[act_mask] - true[act_mask]))
        
    return rmse, bias_act

def run_extended_sweep():
    print("=" * 80)
    print("EXTENDED BASELINE HYPERPARAMETER SWEEP (5 VALIDATION SEEDS, EQUAL BUDGET)")
    print("=" * 80)
    
    df_tr, df_va, _, stats1 = load_cmapss_subset('FD001')
    
    # 1. Mixed-gap training data (for enhanced baselines, D=56)
    tr_short = build_seq_data(df_tr, stats1, drop_prob=0.25, seed=42)
    tr_long  = build_seq_data(df_tr, stats1, fixed_stride_range=(3, 15), seed=1042)
    tr_mixed = {
        k: np.concatenate([tr_short[k], tr_long[k]], axis=0)
        for k in ['obs_t', 'ctrl_t', 'dt', 'obs_next', 'y_norm', 'y_raw', 'mem_fast', 'mem_slow']
    }
    N_tr = tr_mixed['obs_t'].shape[0]
    mask_tr = np.ones((N_tr, 14), dtype=np.float32)
    tr_data_56 = {
        'x': torch.tensor(np.concatenate([tr_mixed['obs_t'], tr_mixed['mem_fast'], tr_mixed['mem_slow'], mask_tr], axis=-1), dtype=torch.float32),
        'u': torch.tensor(tr_mixed['ctrl_t'], dtype=torch.float32),
        'dt': torch.tensor(tr_mixed['dt'], dtype=torch.float32),
        'y': torch.tensor(tr_mixed['y_norm'], dtype=torch.float32)
    }
    
    # 2. Short-gap training data (for original-recipe baselines, D=42)
    N_s = tr_short['obs_t'].shape[0]
    tr_data_42 = {
        'x': torch.tensor(np.concatenate([tr_short['obs_t'], tr_short['mem_fast'], tr_short['mem_slow']], axis=-1), dtype=torch.float32),
        'u': torch.tensor(tr_short['ctrl_t'], dtype=torch.float32),
        'dt': torch.tensor(tr_short['dt'], dtype=torch.float32),
        'y': torch.tensor(tr_short['y_norm'], dtype=torch.float32)
    }
    
    # 3. Validation evaluation data (OOD 10x-20x)
    val_sparse = build_seq_data(df_va, stats1, fixed_stride_range=(10, 20), seed=4042)
    N_va = val_sparse['obs_t'].shape[0]
    mask_va = np.ones((N_va, 14), dtype=np.float32)
    va_data_56 = {
        'x': torch.tensor(np.concatenate([val_sparse['obs_t'], val_sparse['mem_fast'], val_sparse['mem_slow'], mask_va], axis=-1), dtype=torch.float32),
        'u': torch.tensor(val_sparse['ctrl_t'], dtype=torch.float32),
        'dt': torch.tensor(val_sparse['dt'], dtype=torch.float32),
        'y_true': val_sparse['y_raw']
    }
    va_data_42 = {
        'x': torch.tensor(np.concatenate([val_sparse['obs_t'], val_sparse['mem_fast'], val_sparse['mem_slow']], axis=-1), dtype=torch.float32),
        'u': torch.tensor(val_sparse['ctrl_t'], dtype=torch.float32),
        'dt': torch.tensor(val_sparse['dt'], dtype=torch.float32),
        'y_true': val_sparse['y_raw']
    }
    
    # Grid specification:
    # 3 sizes per model x 3 learning rates = 9 trials per model (strictly interior so no best sits at edge)
    models_spec = {
        'GRU_D_Enhanced_SWA': {
            'class': GRU_D_Enhanced_Configurable,
            'param_name': 'hidden_dim',
            'sizes': [24, 32, 48],
            'lrs': [2.5e-3, 5.0e-3, 7.5e-3],
            'in_dim': 56,
            'use_swa': True,
            'data_tr': tr_data_56,
            'data_va': va_data_56
        },
        'GRU_D_Canonical_SWA': {
            'class': GRU_D_Canonical_Configurable,
            'param_name': 'hidden_dim',
            'sizes': [48, 64, 80],
            'lrs': [1.5e-3, 3.5e-3, 5.0e-3],
            'in_dim': 56,
            'use_swa': True,
            'data_tr': tr_data_56,
            'data_va': va_data_56
        },
        'Plain_NODE_Enhanced_SWA': {
            'class': Plain_NODE_Enhanced_Configurable,
            'param_name': 'latent_dim',
            'sizes': [16, 24, 32],
            'lrs': [1.5e-3, 3.5e-3, 5.0e-3],
            'in_dim': 56,
            'use_swa': True,
            'data_tr': tr_data_56,
            'data_va': va_data_56
        },
        'PatchTST_RUL_v2': {
            'class': PatchTST_RUL_Configurable,
            'param_name': 'd_model',
            'sizes': [48, 64, 80],
            'lrs': [1.0e-3, 1.5e-3, 3.5e-3],
            'in_dim': 56,
            'use_swa': True,
            'data_tr': tr_data_56,
            'data_va': va_data_56
        },
        'GRU_D_Original': {
            'class': GRU_D_Original_Configurable,
            'param_name': 'hidden_dim',
            'sizes': [32, 48, 64],
            'lrs': [1.5e-3, 3.5e-3, 5.0e-3],
            'in_dim': 42,
            'use_swa': False,
            'data_tr': tr_data_42,
            'data_va': va_data_42
        },
        'PatchTST_Original': {
            'class': PatchTST_Original_Configurable,
            'param_name': 'd_model',
            'sizes': [32, 48, 64],
            'lrs': [1.5e-3, 3.5e-3, 5.0e-3],
            'in_dim': 42,
            'use_swa': False,
            'data_tr': tr_data_42,
            'data_va': va_data_42
        },
        'Plain_2ndOrder_NODE_Kin_Original': {
            'class': Plain_NODE_Original_Configurable,
            'param_name': 'latent_dim',
            'sizes': [12, 16, 24],
            'lrs': [1.5e-3, 3.5e-3, 5.0e-3],
            'in_dim': 42,
            'use_swa': False,
            'data_tr': tr_data_42,
            'data_va': va_data_42
        }
    }
    
    all_trials = []
    best_tuned = {}
    
    for m_name, spec in models_spec.items():
        print(f"\n--- Sweeping {m_name} (9 trials, 5 seeds each) ---")
        best_rmse = float('inf')
        best_cfg = None
        
        for sz in spec['sizes']:
            for lr in spec['lrs']:
                kwargs = {'in_features': spec['in_dim'], 'u_dim': 3, spec['param_name']: sz}
                seed_rmses, seed_biases = [], []
                
                for s in SEEDS_5:
                    r, b = train_eval_trial(
                        spec['class'], kwargs, lr, spec['data_tr'], spec['data_va'], stats1, seed=s, use_swa=spec['use_swa']
                    )
                    seed_rmses.append(r)
                    seed_biases.append(b)
                    
                mean_r = float(np.mean(seed_rmses))
                std_r = float(np.std(seed_rmses))
                mean_b = float(np.mean(seed_biases))
                
                is_edge_sz = (sz == spec['sizes'][0] or sz == spec['sizes'][-1])
                is_edge_lr = (lr == spec['lrs'][0] or lr == spec['lrs'][-1])
                edge_flag = " (EDGE)" if (is_edge_sz or is_edge_lr) else " (INTERIOR)"
                
                print(f"  {spec['param_name']}={sz:2d}, lr={lr:.1e} => Val RMSE = {mean_r:.3f} +/- {std_r:.3f}, Bias = {mean_b:+.2f}{edge_flag}")
                
                rec = {
                    'model': m_name,
                    'param_name': spec['param_name'],
                    'size': sz,
                    'lr': lr,
                    'is_edge': bool(is_edge_sz or is_edge_lr),
                    'val_rmse_mean': mean_r,
                    'val_rmse_std': std_r,
                    'val_bias_mean': mean_b,
                    'seed_rmses': seed_rmses,
                    'seed_biases': seed_biases
                }
                all_trials.append(rec)
                
                if mean_r < best_rmse:
                    best_rmse = mean_r
                    best_cfg = {
                        'model': m_name,
                        'best_param': spec['param_name'],
                        'best_size': sz,
                        'best_lr': lr,
                        'is_edge': bool(is_edge_sz or is_edge_lr),
                        'val_rmse_mean': mean_r,
                        'val_rmse_std': std_r,
                        'val_bias_mean': mean_b,
                        'seed_rmses': seed_rmses
                    }
                    
        best_tuned[m_name] = best_cfg
        edge_note = "WARNING: At Edge" if best_cfg['is_edge'] else "INTERIOR OPTIMAL (Verified)"
        print(f"  --> BEST for {m_name}: {best_cfg['best_param']}={best_cfg['best_size']}, lr={best_cfg['best_lr']:.1e} | RMSE={best_cfg['val_rmse_mean']:.3f} ({edge_note})")
        
    # Evaluate Proposed Model (FROZEN, not retuned) on the SAME 5 seeds
    print("\n--- Evaluating Frozen Proposed Model (CT_PhysProb_JEPA_v2_SWA) on Same 5 Seeds ---")
    prop_rmses, prop_biases = [], []
    for s in SEEDS_5:
        kwargs = {'in_features': 56, 'u_dim': 3, 'latent_dim': 16, 'alpha': 0.10, 'gamma0': 0.15, 'integrator': 'strang', 'damped': True, 'has_physics': True}
        r, b = train_eval_trial(
            CT_PhysProb_JEPA_IntegratorAblation, kwargs, lr=3.5e-3, tr_data=tr_data_56, va_data=va_data_56, stats_dict=stats1, seed=s, use_swa=True
        )
        prop_rmses.append(r)
        prop_biases.append(b)
        
    prop_mean_r = float(np.mean(prop_rmses))
    prop_std_r = float(np.std(prop_rmses))
    prop_mean_b = float(np.mean(prop_biases))
    print(f"  Proposed Model (Frozen, 3 rounds dev): Val RMSE = {prop_mean_r:.3f} +/- {prop_std_r:.3f}, Bias = {prop_mean_b:+.2f}")
    
    best_tuned['CT_PhysProb_JEPA_v2_SWA_Frozen'] = {
        'model': 'CT_PhysProb_JEPA_v2_SWA',
        'status': 'Frozen_Not_Retuned',
        'development_rounds': 3,
        'latent_dim': 16,
        'hidden_dim': 48,
        'lr': 0.0035,
        'val_rmse_mean': prop_mean_r,
        'val_rmse_std': prop_std_r,
        'val_bias_mean': prop_mean_b,
        'seed_rmses': prop_rmses
    }
    
    # Save outputs
    Path('results/raw').mkdir(parents=True, exist_ok=True)
    df_trials = pd.DataFrame(all_trials)
    df_trials.to_csv('results/raw/extended_baseline_sweep_raw.csv', index=False)
    print(f"\nSaved all {len(all_trials)} trials to results/raw/extended_baseline_sweep_raw.csv")
    
    Path('configs').mkdir(parents=True, exist_ok=True)
    with open('configs/baseline_tuned_configurations.json', 'w') as f:
        json.dump(best_tuned, f, indent=2)
    print("Re-froze configs/baseline_tuned_configurations.json")

if __name__ == '__main__':
    run_extended_sweep()
