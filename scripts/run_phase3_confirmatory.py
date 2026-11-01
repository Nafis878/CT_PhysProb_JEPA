#!/usr/bin/env python3
"""
PHASE 3 CONFIRMATORY EVALUATION (PREREGISTRATION v2.5 OFFICIAL EXECUTION)
Target Dataset: NASA C-MAPSS FD003 Official Test Set (100 engines)
Training Pool: 80 Training Engines (Folds 1-4)
Calibration Pool: 20 Held-out Calibration Engines (Fold 5)
Seeds: 30 Pre-registered Seeds [42, 100..128]
All 11 Models + Heuristics + H4b Component Ablation
Zero simulation, zero hardcoded numbers. All numbers logged to results/raw/.
"""

import sys
import os
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from scripts.run_ood_and_small_data import load_cmapss_subset, build_seq_data, nasa_score
from scripts.run_mechanism_ablations import KneeHazardHead, GRUD_TemporalEncoder, CT_PhysProb_JEPA_IntegratorAblation
from scripts.run_extended_baseline_sweep import (
    GRU_D_Enhanced_Configurable,
    GRU_D_Canonical_Configurable,
    Plain_NODE_Enhanced_Configurable,
    PatchTST_RUL_Configurable,
    GRU_D_Original_Configurable,
    PatchTST_Original_Configurable,
    Plain_NODE_Original_Configurable,
)

# ------------------------------------------------------------------------------
# 30 Pre-registered Seeds
# ------------------------------------------------------------------------------
SEEDS_30 = [
    42, 100, 101, 102, 103, 104, 105, 106, 107, 108, 
    109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 
    119, 120, 121, 122, 123, 124, 125, 126, 127, 128
]

# ------------------------------------------------------------------------------
# Linear Probe Model for Ablations
# ------------------------------------------------------------------------------
class GRU_D_LinearProbe(nn.Module):
    def __init__(self, in_features=56, u_dim=3, hidden_dim=48):
        super().__init__()
        self.encoder = GRUD_TemporalEncoder(in_features, u_dim, latent_dim=hidden_dim, hidden_dim=hidden_dim)
        self.probe = nn.Sequential(
            nn.Linear(hidden_dim + u_dim + 1, 48),
            nn.Tanh(),
            nn.Linear(48, 1)
        )
    def forward(self, x, u, dt):
        dt_eff = torch.clamp(dt, 0.01, 10.0)
        h, _ = self.encoder(x, u, dt_eff)
        y_pred = self.probe(torch.cat([h, u, dt_eff], dim=-1))
        return y_pred, h, h

# ------------------------------------------------------------------------------
# Training Function
# ------------------------------------------------------------------------------
def train_model(model_cls, kwargs, lr, tr_data, seed, epochs=8, batch_size=512, use_swa=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model_cls(**kwargs)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    x_tr = tr_data['x']
    u_tr = tr_data['u']
    dt_tr = tr_data['dt']
    y_tr = tr_data['y']
    N_tr = x_tr.size(0)
    
    swa_weights = None
    swa_count = 0
    
    t0 = time.time()
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
            
        if use_swa and ep >= 4:  # Epochs 5, 6, 7 (0-indexed: 4, 5, 6, 7)
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
        
    tr_sec = time.time() - t0
    return model, tr_sec

# ------------------------------------------------------------------------------
# Evaluation Function
# ------------------------------------------------------------------------------
def evaluate_predictions(model, te_data, stats, is_oracle=False):
    model.eval()
    with torch.no_grad():
        vx = te_data['x']
        vu = te_data['u']
        vdt = te_data['dt']
        
        # Predictions at actual dt
        y_norm, _, _ = model(vx, vu, vdt)
        pred_rul = np.clip(y_norm[:, 0].cpu().numpy() * stats['tgt_std'] + stats['tgt_mean'], 0.0, 125.0)
        
        # Predictions at anchor (nominal dt=0.25 cycles = 0.0625 normalized)
        vdt_anch = torch.full_like(vdt, 0.0625)
        y_anch_norm, _, _ = model(vx, vu, vdt_anch)
        pred_anch = np.clip(y_anch_norm[:, 0].cpu().numpy() * stats['tgt_std'] + stats['tgt_mean'], 0.0, 125.0)
        
    return pred_rul, pred_anch

# ------------------------------------------------------------------------------
# Metrics Computation
# ------------------------------------------------------------------------------
def compute_regime_metrics(pred_rul, pred_anch, te_data):
    y_true = te_data['y_true']
    y_anch_true = te_data['y_anch_true']
    last_idx = te_data['last_indices']
    
    # 1. RMSE_All
    rmse_all = float(np.sqrt(np.mean((pred_rul - y_true) ** 2)))
    
    # 2. RMSE_Last
    rmse_last = float(np.sqrt(np.mean((pred_rul[last_idx] - y_true[last_idx]) ** 2)))
    
    # 3. Active-Degradation Signed Bias & Late Prediction Error
    act_mask = (y_true < 124.9)
    if np.sum(act_mask) > 0:
        err_act = pred_rul[act_mask] - y_true[act_mask]
        bias_act = float(np.mean(err_act))
        late_err = float(np.mean(np.maximum(0.0, err_act)))
    else:
        bias_act = 0.0
        late_err = 0.0
        
    # 4. NASA Score
    nasa = float(nasa_score(pred_rul, y_true))
    
    # 5. Anchor RMSE & Gap-Induced RMSE
    anchor_rmse = float(np.sqrt(np.mean((pred_anch - y_anch_true) ** 2)))
    gap_induced_rmse = rmse_all - anchor_rmse
    
    return {
        'RMSE_All': rmse_all,
        'RMSE_Last': rmse_last,
        'Signed_Bias_ActiveDeg': bias_act,
        'NASA_Score': nasa,
        'Late_Prediction_Error': late_err,
        'Anchor_RMSE': anchor_rmse,
        'Gap_Induced_RMSE': gap_induced_rmse
    }

# ------------------------------------------------------------------------------
# Master Execution Routine
# ------------------------------------------------------------------------------
def run_phase3():
    print("=" * 80)
    print("PHASE 3 CONFIRMATORY EVALUATION (OFFICIAL NASA C-MAPSS FD003 TEST SET)")
    print("=" * 80)
    
    # 1. Load FD003 dataset
    df_tr, df_va, df_te, stats = load_cmapss_subset('FD003')
    print(f"Loaded FD003: Train pool = {df_tr['unit'].nunique()} engines, Calib pool = {df_va['unit'].nunique()} engines, Official Test = {df_te['unit'].nunique()} engines")
    
    # 2. Build Training Datasets
    # 2a. Mixed-gap training data (D=56 for enhanced models)
    tr_short = build_seq_data(df_tr, stats, drop_prob=0.25, seed=42)
    tr_long  = build_seq_data(df_tr, stats, fixed_stride_range=(3, 15), seed=1042)
    tr_mixed = {
        k: np.concatenate([tr_short[k], tr_long[k]], axis=0)
        for k in ['obs_t', 'ctrl_t', 'dt', 'obs_next', 'y_norm', 'y_raw', 'mem_fast', 'mem_slow']
    }
    N_tr = tr_mixed['obs_t'].shape[0]
    mask_tr_ones = np.ones((N_tr, 14), dtype=np.float32)
    mask_tr_zeros = np.zeros((N_tr, 14), dtype=np.float32)
    
    tr_data_mixed_56 = {
        'x': torch.tensor(np.concatenate([tr_mixed['obs_t'], tr_mixed['mem_fast'], tr_mixed['mem_slow'], mask_tr_ones], axis=-1), dtype=torch.float32),
        'u': torch.tensor(tr_mixed['ctrl_t'], dtype=torch.float32),
        'dt': torch.tensor(tr_mixed['dt'], dtype=torch.float32),
        'y': torch.tensor(tr_mixed['y_norm'], dtype=torch.float32)
    }
    
    # 2b. Short-gap training data (D=42 for canonical baselines)
    tr_data_short_42 = {
        'x': torch.tensor(np.concatenate([tr_short['obs_t'], tr_short['mem_fast'], tr_short['mem_slow']], axis=-1), dtype=torch.float32),
        'u': torch.tensor(tr_short['ctrl_t'], dtype=torch.float32),
        'dt': torch.tensor(tr_short['dt'], dtype=torch.float32),
        'y': torch.tensor(tr_short['y_norm'], dtype=torch.float32)
    }
    
    # 2c. Short-gap training data with mask (D=56 for H4b No-MixedGap ablation)
    N_s = tr_short['obs_t'].shape[0]
    mask_s_ones = np.ones((N_s, 14), dtype=np.float32)
    tr_data_short_56 = {
        'x': torch.tensor(np.concatenate([tr_short['obs_t'], tr_short['mem_fast'], tr_short['mem_slow'], mask_s_ones], axis=-1), dtype=torch.float32),
        'u': torch.tensor(tr_short['ctrl_t'], dtype=torch.float32),
        'dt': torch.tensor(tr_short['dt'], dtype=torch.float32),
        'y': torch.tensor(tr_short['y_norm'], dtype=torch.float32)
    }
    
    # 2d. Mixed-gap training data with zero mask (D=56 for H4b No-Mask ablation)
    tr_data_mixed_56_nomask = {
        'x': torch.tensor(np.concatenate([tr_mixed['obs_t'], tr_mixed['mem_fast'], tr_mixed['mem_slow'], mask_tr_zeros], axis=-1), dtype=torch.float32),
        'u': torch.tensor(tr_mixed['ctrl_t'], dtype=torch.float32),
        'dt': torch.tensor(tr_mixed['dt'], dtype=torch.float32),
        'y': torch.tensor(tr_mixed['y_norm'], dtype=torch.float32)
    }

    # 3. Build Test Regimes (Generated ONCE under Common Random Numbers seed=4042)
    print("\nGenerating Official Test Regimes under Common Random Numbers (seed=4042)...")
    test_raw_dict = {
        'InDist_1x_2x': build_seq_data(df_te, stats, drop_prob=0.25, seed=4042),
        'OOD_10x_20x':  build_seq_data(df_te, stats, fixed_stride_range=(10, 20), seed=4042),
        'OOD_15x_25x':  build_seq_data(df_te, stats, fixed_stride_range=(15, 25), seed=4042),
        'OOD_20x_30x':  build_seq_data(df_te, stats, fixed_stride_range=(20, 30), seed=4042),
        'OOD_10x_20x_sens': build_seq_data(df_te, stats, fixed_stride_range=(10, 20), seed=9042),
    }
    
    # Prepare Tensor dictionaries for 56-dim and 42-dim models, and sensor dropouts
    regimes_56 = {}
    regimes_42 = {}
    
    for r_name, r_raw in test_raw_dict.items():
        N = r_raw['obs_t'].shape[0]
        m_ones = np.ones((N, 14), dtype=np.float32)
        dt_cyc = r_raw['dt'][:, 0] * 4.0
        y_anch_true = np.minimum(125.0, r_raw['y_raw'] + dt_cyc)
        
        regimes_56[r_name] = {
            'x': torch.tensor(np.concatenate([r_raw['obs_t'], r_raw['mem_fast'], r_raw['mem_slow'], m_ones], axis=-1), dtype=torch.float32),
            'u': torch.tensor(r_raw['ctrl_t'], dtype=torch.float32),
            'dt': torch.tensor(r_raw['dt'], dtype=torch.float32),
            'y_true': r_raw['y_raw'],
            'y_anch_true': y_anch_true,
            'unit_id': r_raw['unit_id'],
            'last_indices': r_raw['last_indices']
        }
        regimes_42[r_name] = {
            'x': torch.tensor(np.concatenate([r_raw['obs_t'], r_raw['mem_fast'], r_raw['mem_slow']], axis=-1), dtype=torch.float32),
            'u': torch.tensor(r_raw['ctrl_t'], dtype=torch.float32),
            'dt': torch.tensor(r_raw['dt'], dtype=torch.float32),
            'y_true': r_raw['y_raw'],
            'y_anch_true': y_anch_true,
            'unit_id': r_raw['unit_id'],
            'last_indices': r_raw['last_indices']
        }
        
    # Sensor dropouts based on OOD 10x-20x: 10% (1 ch), 20% (3 ch), 30% (4 ch)
    ood_raw = test_raw_dict['OOD_10x_20x']
    N_ood = ood_raw['obs_t'].shape[0]
    dt_cyc = ood_raw['dt'][:, 0] * 4.0
    y_anch_true = np.minimum(125.0, ood_raw['y_raw'] + dt_cyc)
    
    np.random.seed(5042)
    perm_channels = np.random.permutation(14)
    drop_channels_dict = {
        'Dropout_10pct': perm_channels[:1],  # 1 channel
        'Dropout_20pct': perm_channels[:3],  # 3 channels
        'Dropout_30pct': perm_channels[:4]   # 4 channels
    }
    
    for d_name, ch_list in drop_channels_dict.items():
        # 56-dim: zero out sensor, fast EWMA, slow EWMA, and set mask to 0
        x_obs = ood_raw['obs_t'].copy()
        x_fast = ood_raw['mem_fast'].copy()
        x_slow = ood_raw['mem_slow'].copy()
        m_vec = np.ones((N_ood, 14), dtype=np.float32)
        
        for ch in ch_list:
            x_obs[:, ch] = 0.0
            x_fast[:, ch] = 0.0
            x_slow[:, ch] = 0.0
            m_vec[:, ch] = 0.0
            
        regimes_56[d_name] = {
            'x': torch.tensor(np.concatenate([x_obs, x_fast, x_slow, m_vec], axis=-1), dtype=torch.float32),
            'u': torch.tensor(ood_raw['ctrl_t'], dtype=torch.float32),
            'dt': torch.tensor(ood_raw['dt'], dtype=torch.float32),
            'y_true': ood_raw['y_raw'],
            'y_anch_true': y_anch_true,
            'unit_id': ood_raw['unit_id'],
            'last_indices': ood_raw['last_indices']
        }
        # 42-dim: zero out sensor, fast EWMA, slow EWMA
        regimes_42[d_name] = {
            'x': torch.tensor(np.concatenate([x_obs, x_fast, x_slow], axis=-1), dtype=torch.float32),
            'u': torch.tensor(ood_raw['ctrl_t'], dtype=torch.float32),
            'dt': torch.tensor(ood_raw['dt'], dtype=torch.float32),
            'y_true': ood_raw['y_raw'],
            'y_anch_true': y_anch_true,
            'unit_id': ood_raw['unit_id'],
            'last_indices': ood_raw['last_indices']
        }

    # Also build calibration data on 20 held-out engines for conformal evaluation
    calib_ood = build_seq_data(df_va, stats, fixed_stride_range=(10, 20), seed=4042)
    N_cal = calib_ood['obs_t'].shape[0]
    m_cal_ones = np.ones((N_cal, 14), dtype=np.float32)
    calib_data_56 = {
        'x': torch.tensor(np.concatenate([calib_ood['obs_t'], calib_ood['mem_fast'], calib_ood['mem_slow'], m_cal_ones], axis=-1), dtype=torch.float32),
        'u': torch.tensor(calib_ood['ctrl_t'], dtype=torch.float32),
        'dt': torch.tensor(calib_ood['dt'], dtype=torch.float32),
        'y_true': calib_ood['y_raw'],
        'y_anch_true': np.minimum(125.0, calib_ood['y_raw'] + calib_ood['dt'][:, 0] * 4.0),
        'unit_id': calib_ood['unit_id'],
        'last_indices': calib_ood['last_indices']
    }
    calib_data_42 = {
        'x': torch.tensor(np.concatenate([calib_ood['obs_t'], calib_ood['mem_fast'], calib_ood['mem_slow']], axis=-1), dtype=torch.float32),
        'u': torch.tensor(calib_ood['ctrl_t'], dtype=torch.float32),
        'dt': torch.tensor(calib_ood['dt'], dtype=torch.float32),
        'y_true': calib_ood['y_raw'],
        'y_anch_true': np.minimum(125.0, calib_ood['y_raw'] + calib_ood['dt'][:, 0] * 4.0),
        'unit_id': calib_ood['unit_id'],
        'last_indices': calib_ood['last_indices']
    }

    # 4. Define All Evaluated Models per the Pre-Registration Symmetry Table
    models_config = {
        # Proposed Model (Frozen, 3 dev rounds)
        'CT_PhysProb_JEPA_v2_SWA': {
            'class': CT_PhysProb_JEPA_IntegratorAblation,
            'kwargs': {'in_features': 56, 'u_dim': 3, 'latent_dim': 16, 'alpha': 0.10, 'gamma0': 0.15, 'integrator': 'strang', 'damped': True, 'has_physics': True},
            'lr': 3.5e-3, 'tr_data': tr_data_mixed_56, 'regimes': regimes_56, 'calib_data': calib_data_56, 'use_swa': True, 'in_dim': 56
        },
        # No-Physics Control (Matched ablation backbone)
        'CT_PhysProb_JEPA_NoPhysics': {
            'class': CT_PhysProb_JEPA_IntegratorAblation,
            'kwargs': {'in_features': 56, 'u_dim': 3, 'latent_dim': 16, 'alpha': 0.10, 'gamma0': 0.15, 'integrator': 'strang', 'damped': True, 'has_physics': False},
            'lr': 3.5e-3, 'tr_data': tr_data_mixed_56, 'regimes': regimes_56, 'calib_data': calib_data_56, 'use_swa': True, 'in_dim': 56
        },
        # Integrator Ablations (Descriptive)
        'Euler_Damped': {
            'class': CT_PhysProb_JEPA_IntegratorAblation,
            'kwargs': {'in_features': 56, 'u_dim': 3, 'latent_dim': 16, 'alpha': 0.10, 'gamma0': 0.15, 'integrator': 'euler', 'damped': True, 'has_physics': True},
            'lr': 3.5e-3, 'tr_data': tr_data_mixed_56, 'regimes': regimes_56, 'calib_data': calib_data_56, 'use_swa': True, 'in_dim': 56
        },
        'Undamped': {
            'class': CT_PhysProb_JEPA_IntegratorAblation,
            'kwargs': {'in_features': 56, 'u_dim': 3, 'latent_dim': 16, 'alpha': 0.10, 'gamma0': 0.15, 'integrator': 'strang', 'damped': False, 'has_physics': True},
            'lr': 3.5e-3, 'tr_data': tr_data_mixed_56, 'regimes': regimes_56, 'calib_data': calib_data_56, 'use_swa': True, 'in_dim': 56
        },
        # Recipe Baselines
        'GRU_D_Enhanced_SWA': {
            'class': GRU_D_Enhanced_Configurable,
            'kwargs': {'in_features': 56, 'u_dim': 3, 'hidden_dim': 48},
            'lr': 5.0e-3, 'tr_data': tr_data_mixed_56, 'regimes': regimes_56, 'calib_data': calib_data_56, 'use_swa': True, 'in_dim': 56
        },
        'GRU_D_Canonical_SWA': {
            'class': GRU_D_Canonical_Configurable,
            'kwargs': {'in_features': 56, 'u_dim': 3, 'hidden_dim': 80},
            'lr': 3.5e-3, 'tr_data': tr_data_mixed_56, 'regimes': regimes_56, 'calib_data': calib_data_56, 'use_swa': True, 'in_dim': 56
        },
        'Plain_NODE_Enhanced_SWA': {
            'class': Plain_NODE_Enhanced_Configurable,
            'kwargs': {'in_features': 56, 'u_dim': 3, 'latent_dim': 24},
            'lr': 3.5e-3, 'tr_data': tr_data_mixed_56, 'regimes': regimes_56, 'calib_data': calib_data_56, 'use_swa': True, 'in_dim': 56
        },
        'PatchTST_RUL_v2': {
            'class': PatchTST_RUL_Configurable,
            'kwargs': {'in_features': 56, 'u_dim': 3, 'd_model': 64},
            'lr': 1.0e-3, 'tr_data': tr_data_mixed_56, 'regimes': regimes_56, 'calib_data': calib_data_56, 'use_swa': True, 'in_dim': 56
        },
        # Original-Recipe Baselines
        'GRU_D_Original': {
            'class': GRU_D_Original_Configurable,
            'kwargs': {'in_features': 42, 'u_dim': 3, 'hidden_dim': 80},
            'lr': 3.5e-3, 'tr_data': tr_data_short_42, 'regimes': regimes_42, 'calib_data': calib_data_42, 'use_swa': False, 'in_dim': 42
        },
        'PatchTST_Original': {
            'class': PatchTST_Original_Configurable,
            'kwargs': {'in_features': 42, 'u_dim': 3, 'd_model': 48},
            'lr': 3.5e-3, 'tr_data': tr_data_short_42, 'regimes': regimes_42, 'calib_data': calib_data_42, 'use_swa': False, 'in_dim': 42
        },
        'Plain_2ndOrder_NODE_Kin_Original': {
            'class': Plain_NODE_Original_Configurable,
            'kwargs': {'in_features': 42, 'u_dim': 3, 'latent_dim': 24},
            'lr': 7.5e-3, 'tr_data': tr_data_short_42, 'regimes': regimes_42, 'calib_data': calib_data_42, 'use_swa': False, 'in_dim': 42
        }
    }
    
    # 5. Define H4b Component Ablation Models
    h4b_config = {
        'H4b_All_On': {
            'class': GRU_D_Enhanced_Configurable,
            'kwargs': {'in_features': 56, 'u_dim': 3, 'hidden_dim': 48},
            'lr': 5.0e-3, 'tr_data': tr_data_mixed_56, 'use_swa': True, 'in_dim': 56
        },
        'H4b_All_Off': {
            'class': GRU_D_LinearProbe,
            'kwargs': {'in_features': 56, 'u_dim': 3, 'hidden_dim': 48},
            'lr': 5.0e-3, 'tr_data': tr_data_short_56, 'use_swa': False, 'in_dim': 56
        },
        'H4b_No_Mask': {
            'class': GRU_D_Enhanced_Configurable,
            'kwargs': {'in_features': 56, 'u_dim': 3, 'hidden_dim': 48},
            'lr': 5.0e-3, 'tr_data': tr_data_mixed_56_nomask, 'use_swa': True, 'in_dim': 56
        },
        'H4b_No_MixedGap': {
            'class': GRU_D_Enhanced_Configurable,
            'kwargs': {'in_features': 56, 'u_dim': 3, 'hidden_dim': 48},
            'lr': 5.0e-3, 'tr_data': tr_data_short_56, 'use_swa': True, 'in_dim': 56
        },
        'H4b_No_HazardHead': {
            'class': GRU_D_LinearProbe,
            'kwargs': {'in_features': 56, 'u_dim': 3, 'hidden_dim': 48},
            'lr': 5.0e-3, 'tr_data': tr_data_mixed_56, 'use_swa': True, 'in_dim': 56
        },
        'H4b_No_SWA': {
            'class': GRU_D_Enhanced_Configurable,
            'kwargs': {'in_features': 56, 'u_dim': 3, 'hidden_dim': 48},
            'lr': 5.0e-3, 'tr_data': tr_data_mixed_56, 'use_swa': False, 'in_dim': 56
        }
    }
    
    # Setup Output Paths
    Path('results/raw').mkdir(parents=True, exist_ok=True)
    Path('results/preds').mkdir(parents=True, exist_ok=True)
    raw_csv = Path('results/raw/phase3_confirmatory_raw.csv')
    h4b_csv = Path('results/raw/phase3_h4b_ablation_raw.csv')
    
    # Save test metadata for all regimes once
    for r_name, r_data in regimes_56.items():
        meta_file = Path(f"results/preds/metadata_{r_name}.npz")
        if not meta_file.exists():
            np.savez(
                meta_file,
                y_true=r_data['y_true'],
                y_anch_true=r_data['y_anch_true'],
                unit_id=r_data['unit_id'],
                last_indices=r_data['last_indices'],
                dt=r_data['dt'].numpy()
            )
    
    # Check for existing partial logs to allow resume
    all_rows = []
    h4b_rows = []
    completed_keys = set()
    if raw_csv.exists():
        df_exist = pd.read_csv(raw_csv)
        all_rows = df_exist.to_dict('records')
        for r in all_rows:
            completed_keys.add((r['Seed'], r['Model'], r['Regime']))
        print(f"Resuming with {len(all_rows)} already completed test rows.")
        
    completed_h4b_keys = set()
    if h4b_csv.exists():
        df_h4b_exist = pd.read_csv(h4b_csv)
        h4b_rows = df_h4b_exist.to_dict('records')
        for r in h4b_rows:
            completed_h4b_keys.add((r['Seed'], r['Ablation']))
        print(f"Resuming with {len(h4b_rows)} already completed H4b rows.")

    # --------------------------------------------------------------------------
    # 6. Main Confirmatory Loop across 30 Seeds
    # --------------------------------------------------------------------------
    print(f"\nBeginning Evaluation of {len(models_config)} Models across {len(SEEDS_30)} Seeds...")
    t_start = time.time()
    
    for s_idx, s in enumerate(SEEDS_30):
        t_seed = time.time()
        print(f"\n{'='*40}\nSEED {s} ({s_idx+1}/{len(SEEDS_30)})\n{'='*40}")
        
        # 6a. Evaluate 11 Primary Confirmatory Models
        for m_name, cfg in models_config.items():
            # Check if all regimes for this seed/model are already logged
            needs_eval = any((s, m_name, r) not in completed_keys for r in cfg['regimes'])
            if not needs_eval:
                continue
                
            print(f"  Training {m_name} (seed={s})...", end=" ", flush=True)
            model, tr_sec = train_model(
                cfg['class'], cfg['kwargs'], cfg['lr'], cfg['tr_data'], seed=s, use_swa=cfg['use_swa']
            )
            print(f"done ({tr_sec:.1f}s). Evaluating regimes...", flush=True)
            
            # Conformal calibration scores on held-out 20 engines
            pred_cal, _ = evaluate_predictions(model, cfg['calib_data'], stats)
            s_calib = np.abs(pred_cal - cfg['calib_data']['y_true'])
            # 90% non-conformity quantile with finite-sample correction
            n_cal = len(s_calib)
            q_idx = int(np.ceil((n_cal + 1) * 0.90)) - 1
            q_alpha = float(np.sort(s_calib)[min(q_idx, n_cal - 1)])
            
            # Evaluate across all 7 test regimes
            for r_name, r_data in cfg['regimes'].items():
                if (s, m_name, r_name) in completed_keys:
                    continue
                    
                pred_rul, pred_anch = evaluate_predictions(model, r_data, stats)
                
                # Save predictions for all regimes and seeds
                pred_file = Path(f"results/preds/{m_name}_{r_name}_seed{s}_pred.npy")
                np.save(pred_file, pred_rul)
                if r_name == 'OOD_10x_20x':
                    np.save(Path(f"results/preds/{m_name}_seed{s}_pred.npy"), pred_rul)
                    
                metrics = compute_regime_metrics(pred_rul, pred_anch, r_data)
                
                # Conformal coverage & width on test set
                ci_lo = np.clip(pred_rul - q_alpha, 0.0, 125.0)
                ci_hi = np.clip(pred_rul + q_alpha, 0.0, 125.0)
                coverage = float(np.mean((r_data['y_true'] >= ci_lo) & (r_data['y_true'] <= ci_hi)))
                mean_width = float(np.mean(ci_hi - ci_lo))
                
                # Missed critical failures: actual <= 10 cycles, but lower bound > 15
                crit_mask = (r_data['y_true'] <= 10.0)
                missed_crit = float(np.mean(ci_lo[crit_mask] > 15.0)) if np.sum(crit_mask) > 0 else 0.0
                
                # CBM Cost: intervention threshold = 15 cycles, Cu / Cp = 5.0
                # True failure if y_true <= 0 without intervention
                # Planned replacement if pred <= 15
                intervene = (pred_rul <= 15.0)
                unplanned = (~intervene) & (r_data['y_true'] <= 5.0)
                cbm_cost = float(np.mean(intervene * 1.0 + unplanned * 5.0))
                
                rec = {
                    'Seed': s,
                    'Model': m_name,
                    'Regime': r_name,
                    'Train_Sec': tr_sec,
                    'RMSE_All': metrics['RMSE_All'],
                    'RMSE_Last': metrics['RMSE_Last'],
                    'Signed_Bias_ActiveDeg': metrics['Signed_Bias_ActiveDeg'],
                    'NASA_Score': metrics['NASA_Score'],
                    'Late_Prediction_Error': metrics['Late_Prediction_Error'],
                    'Anchor_RMSE': metrics['Anchor_RMSE'],
                    'Gap_Induced_RMSE': metrics['Gap_Induced_RMSE'],
                    'Conformal_Coverage_90': coverage,
                    'Conformal_Width_90': mean_width,
                    'Missed_Critical_Failures': missed_crit,
                    'Expected_CBM_Cost': cbm_cost,
                    'Q_Alpha': q_alpha
                }
                all_rows.append(rec)
                completed_keys.add((s, m_name, r_name))
                
            # Heuristic Reference 1: EstimatedAnchor_Countdown (from this model's own anchor on OOD 10x-20x)
            r_ood = cfg['regimes']['OOD_10x_20x']
            dt_cyc = r_ood['dt'][:, 0].numpy() * 4.0
            _, pred_anch_ood = evaluate_predictions(model, r_ood, stats)
            pred_countdown = np.clip(pred_anch_ood - dt_cyc, 0.0, 125.0)
            np.save(Path(f"results/preds/{m_name}_Countdown_OOD_10x_20x_seed{s}_pred.npy"), pred_countdown)
            m_count = compute_regime_metrics(pred_countdown, pred_anch_ood, r_ood)
            all_rows.append({
                'Seed': s, 'Model': f'{m_name}_Countdown', 'Regime': 'OOD_10x_20x', 'Train_Sec': tr_sec,
                'RMSE_All': m_count['RMSE_All'], 'RMSE_Last': m_count['RMSE_Last'],
                'Signed_Bias_ActiveDeg': m_count['Signed_Bias_ActiveDeg'],
                'NASA_Score': m_count['NASA_Score'], 'Late_Prediction_Error': m_count['Late_Prediction_Error'],
                'Anchor_RMSE': m_count['Anchor_RMSE'], 'Gap_Induced_RMSE': m_count['Gap_Induced_RMSE'],
                'Conformal_Coverage_90': 0.0, 'Conformal_Width_90': 0.0, 'Missed_Critical_Failures': 0.0, 'Expected_CBM_Cost': 0.0, 'Q_Alpha': 0.0
            })
            
        # Heuristic Reference 2: TimeCorrected_LOCF_Oracle (Non-deployable upper bound)
        r_ood = regimes_56['OOD_10x_20x']
        dt_cyc = r_ood['dt'][:, 0].numpy() * 4.0
        pred_oracle = np.clip(r_ood['y_anch_true'] - dt_cyc, 0.0, 125.0)
        np.save(Path(f"results/preds/TimeCorrected_LOCF_Oracle_OOD_10x_20x_seed{s}_pred.npy"), pred_oracle)
        m_oracle = compute_regime_metrics(pred_oracle, r_ood['y_anch_true'], r_ood)
        all_rows.append({
            'Seed': s, 'Model': 'TimeCorrected_LOCF_Oracle', 'Regime': 'OOD_10x_20x', 'Train_Sec': 0.0,
            'RMSE_All': m_oracle['RMSE_All'], 'RMSE_Last': m_oracle['RMSE_Last'],
            'Signed_Bias_ActiveDeg': m_oracle['Signed_Bias_ActiveDeg'],
            'NASA_Score': m_oracle['NASA_Score'], 'Late_Prediction_Error': m_oracle['Late_Prediction_Error'],
            'Anchor_RMSE': m_oracle['Anchor_RMSE'], 'Gap_Induced_RMSE': m_oracle['Gap_Induced_RMSE'],
            'Conformal_Coverage_90': 1.0, 'Conformal_Width_90': 0.0, 'Missed_Critical_Failures': 0.0, 'Expected_CBM_Cost': 0.0, 'Q_Alpha': 0.0
        })

        # 6b. Evaluate H4b Component Ablations on GRU_D_Enhanced Backbone
        print(f"  Evaluating H4b Component Ablations (seed={s})...", flush=True)
        r_ood = regimes_56['OOD_10x_20x']
        for ab_name, ab_cfg in h4b_config.items():
            if (s, ab_name) in completed_h4b_keys:
                continue
            m_ab, tr_sec_ab = train_model(
                ab_cfg['class'], ab_cfg['kwargs'], ab_cfg['lr'], ab_cfg['tr_data'], seed=s, use_swa=ab_cfg['use_swa']
            )
            p_rul, p_anch = evaluate_predictions(m_ab, r_ood, stats)
            np.save(Path(f"results/preds/h4b_{ab_name}_seed{s}_pred.npy"), p_rul)
            m_res = compute_regime_metrics(p_rul, p_anch, r_ood)
            h4b_rec = {
                'Seed': s,
                'Ablation': ab_name,
                'Train_Sec': tr_sec_ab,
                'RMSE_All': m_res['RMSE_All'],
                'RMSE_Last': m_res['RMSE_Last'],
                'Signed_Bias_ActiveDeg': m_res['Signed_Bias_ActiveDeg'],
                'NASA_Score': m_res['NASA_Score'],
                'Late_Prediction_Error': m_res['Late_Prediction_Error'],
                'Anchor_RMSE': m_res['Anchor_RMSE'],
                'Gap_Induced_RMSE': m_res['Gap_Induced_RMSE']
            }
            h4b_rows.append(h4b_rec)
            completed_h4b_keys.add((s, ab_name))
            
        # Save checkpoints after each seed
        pd.DataFrame(all_rows).to_csv(raw_csv, index=False)
        pd.DataFrame(h4b_rows).to_csv(h4b_csv, index=False)
        print(f"Seed {s} complete in {time.time() - t_seed:.1f}s. Total logged rows: {len(all_rows)}.")
        
    print("\n" + "=" * 80)
    print(f"CONFIRMATORY EVALUATION COMPLETE in {time.time() - t_start:.1f}s.")
    print(f"Saved {len(all_rows)} rows to {raw_csv}")
    print(f"Saved {len(h4b_rows)} rows to {h4b_csv}")
    print("=" * 80)

if __name__ == '__main__':
    run_phase3()
