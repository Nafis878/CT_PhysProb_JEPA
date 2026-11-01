#!/usr/bin/env python3
"""
Evaluate grid boundary extensions for the 5 baselines whose best setting was at an edge.
Runs on the same 5 validation seeds [42, 100, 101, 102, 103].
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json
import numpy as np
import pandas as pd
import torch

from scripts.run_extended_baseline_sweep import (
    train_eval_trial,
    GRU_D_Enhanced_Configurable,
    GRU_D_Canonical_Configurable,
    PatchTST_RUL_Configurable,
    GRU_D_Original_Configurable,
    Plain_NODE_Original_Configurable,
    load_cmapss_subset,
    build_seq_data,
    SEEDS_5
)

def main():
    df_tr, df_va, _, stats1 = load_cmapss_subset('FD001')
    
    # 1. Mixed-gap training data (D=56)
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
    
    # 2. Short-gap training data (D=42)
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
    
    # Edge extension trials to run
    edge_extensions = [
        # Model 1: GRU_D_Enhanced_SWA was best at size=48, lr=5.0e-3. Extend size to 64.
        {
            'model': 'GRU_D_Enhanced_SWA',
            'class': GRU_D_Enhanced_Configurable,
            'param_name': 'hidden_dim',
            'size': 64,
            'lr': 5.0e-3,
            'in_dim': 56,
            'use_swa': True,
            'data_tr': tr_data_56,
            'data_va': va_data_56
        },
        # Model 2: GRU_D_Canonical_SWA was best at size=80, lr=3.5e-3. Extend size to 96.
        {
            'model': 'GRU_D_Canonical_SWA',
            'class': GRU_D_Canonical_Configurable,
            'param_name': 'hidden_dim',
            'size': 96,
            'lr': 3.5e-3,
            'in_dim': 56,
            'use_swa': True,
            'data_tr': tr_data_56,
            'data_va': va_data_56
        },
        # Model 3: PatchTST_RUL_v2 was best at size=64, lr=1.0e-3. Extend lr to 5.0e-4.
        {
            'model': 'PatchTST_RUL_v2',
            'class': PatchTST_RUL_Configurable,
            'param_name': 'd_model',
            'size': 64,
            'lr': 5.0e-4,
            'in_dim': 56,
            'use_swa': True,
            'data_tr': tr_data_56,
            'data_va': va_data_56
        },
        # Model 4: GRU_D_Original was best at size=64, lr=3.5e-3. Extend size to 80.
        {
            'model': 'GRU_D_Original',
            'class': GRU_D_Original_Configurable,
            'param_name': 'hidden_dim',
            'size': 80,
            'lr': 3.5e-3,
            'in_dim': 42,
            'use_swa': False,
            'data_tr': tr_data_42,
            'data_va': va_data_42
        },
        # Model 5: Plain_2ndOrder_NODE_Kin_Original was best at size=24, lr=5.0e-3. Extend size to 32 and lr to 7.5e-3.
        {
            'model': 'Plain_2ndOrder_NODE_Kin_Original',
            'class': Plain_NODE_Original_Configurable,
            'param_name': 'latent_dim',
            'size': 32,
            'lr': 5.0e-3,
            'in_dim': 42,
            'use_swa': False,
            'data_tr': tr_data_42,
            'data_va': va_data_42
        },
        {
            'model': 'Plain_2ndOrder_NODE_Kin_Original',
            'class': Plain_NODE_Original_Configurable,
            'param_name': 'latent_dim',
            'size': 24,
            'lr': 7.5e-3,
            'in_dim': 42,
            'use_swa': False,
            'data_tr': tr_data_42,
            'data_va': va_data_42
        }
    ]
    
    results = []
    print("Running Edge Extensions across 5 Seeds...")
    for ext in edge_extensions:
        kwargs = {'in_features': ext['in_dim'], 'u_dim': 3, ext['param_name']: ext['size']}
        rmses, biases = [], []
        for s in SEEDS_5:
            r, b = train_eval_trial(
                ext['class'], kwargs, ext['lr'], ext['data_tr'], ext['data_va'], stats1, seed=s, use_swa=ext['use_swa']
            )
            rmses.append(r)
            biases.append(b)
        mean_r = float(np.mean(rmses))
        std_r = float(np.std(rmses))
        mean_b = float(np.mean(biases))
        print(f"  {ext['model']} [{ext['param_name']}={ext['size']}, lr={ext['lr']:.1e}]: RMSE = {mean_r:.3f} +/- {std_r:.3f}, Bias = {mean_b:+.2f}")
        results.append({
            'model': ext['model'],
            'param_name': ext['param_name'],
            'size': ext['size'],
            'lr': ext['lr'],
            'val_rmse_mean': mean_r,
            'val_rmse_std': std_r,
            'val_bias_mean': mean_b,
            'seed_rmses': rmses,
            'seed_biases': biases
        })
        
    df_ext = pd.DataFrame(results)
    df_ext.to_csv(ROOT / "results" / "raw" / "baseline_edge_extensions_raw.csv", index=False)
    print("\nSaved edge extension trials to results/raw/baseline_edge_extensions_raw.csv")

if __name__ == '__main__':
    main()
