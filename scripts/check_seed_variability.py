#!/usr/bin/env python3
"""
Diagnostic script for Seed Variability Check (Point 8)
Compares standard deviations between:
1. Original 10 seeds in step2_development_raw.csv (where seed controlled model init, batch shuffling, and window sampling)
2. Extended baseline sweep 5 seeds (where validation windows were held fixed, seed controlled model init and batch permute)
3. Direct evaluation of proposed model across both protocols.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json
import numpy as np
import pandas as pd
import torch

from scripts.run_extended_baseline_sweep import train_eval_trial, CT_PhysProb_JEPA_IntegratorAblation, load_cmapss_subset, build_seq_data

SEEDS_10 = [42, 100, 2024, 777, 999, 101, 202, 303, 404, 505]
SEEDS_5 = [42, 100, 101, 102, 103]

def main():
    print("=" * 80)
    print("POINT 8: SEED VARIABILITY CHECK & SOURCE OF SD DIFFERENCES")
    print("=" * 80)
    
    # 1. Historical 10 seeds from step2_development_raw.csv
    df_dev = pd.read_csv(ROOT / "results" / "raw" / "step2_development_raw.csv")
    prop_10_hist = df_dev[(df_dev['Regime'] == 'OOD_10x_20x') & (df_dev['Model'] == 'CT_PhysProb_JEPA_v2_SWA')]
    
    # 2. Historical 5 seeds from configs/baseline_tuned_configurations.json
    with open(ROOT / "configs" / "baseline_tuned_configurations.json") as f:
        cfg = json.load(f)
    prop_5_sweep = cfg['CT_PhysProb_JEPA_v2_SWA_Frozen']
    
    print("\n[Protocol A: Historical Step 2 (Varying Window Sampling + Model Init + Batch Shuffling)]")
    print(prop_10_hist[['Seed', 'Total_RMSE']].to_string(index=False))
    sd_10_hist = float(prop_10_hist['Total_RMSE'].std())
    mean_10_hist = float(prop_10_hist['Total_RMSE'].mean())
    print(f"--> 10-Seed Historical SD: {sd_10_hist:.4f} cycles (Mean = {mean_10_hist:.3f})")
    
    print("\n[Protocol B: Extended Baseline Sweep (Fixed Validation Windows, Varying Model Init + Batch Shuffling)]")
    for s, r in zip(SEEDS_5, prop_5_sweep['seed_rmses']):
        print(f"Seed {s:4d}: {r:.4f}")
    sd_5_sweep = float(prop_5_sweep['val_rmse_std'])
    mean_5_sweep = float(prop_5_sweep['val_rmse_mean'])
    print(f"--> 5-Seed Sweep SD: {sd_5_sweep:.4f} cycles (Mean = {mean_5_sweep:.3f})")

    # 3. What happens if we evaluate the original 10 seeds under Protocol B (Fixed validation windows)?
    df_tr, df_va, _, stats1 = load_cmapss_subset('FD001')
    tr_short = build_seq_data(df_tr, stats1, drop_prob=0.25, seed=42)
    tr_long  = build_seq_data(df_tr, stats1, fixed_stride_range=(3, 15), seed=1042)
    tr_mixed = {
        k: np.concatenate([tr_short[k], tr_long[k]], axis=0)
        for k in ['obs_t', 'ctrl_t', 'dt', 'obs_next', 'y_norm', 'y_raw', 'mem_fast', 'mem_slow']
    }
    mask_tr = np.ones((tr_mixed['obs_t'].shape[0], 14), dtype=np.float32)
    tr_data_56 = {
        'x': torch.tensor(np.concatenate([tr_mixed['obs_t'], tr_mixed['mem_fast'], tr_mixed['mem_slow'], mask_tr], axis=-1), dtype=torch.float32),
        'u': torch.tensor(tr_mixed['ctrl_t'], dtype=torch.float32),
        'dt': torch.tensor(tr_mixed['dt'], dtype=torch.float32),
        'y': torch.tensor(tr_mixed['y_norm'], dtype=torch.float32)
    }
    val_sparse = build_seq_data(df_va, stats1, fixed_stride_range=(10, 20), seed=4042)
    mask_va = np.ones((val_sparse['obs_t'].shape[0], 14), dtype=np.float32)
    va_data_56 = {
        'x': torch.tensor(np.concatenate([val_sparse['obs_t'], val_sparse['mem_fast'], val_sparse['mem_slow'], mask_va], axis=-1), dtype=torch.float32),
        'u': torch.tensor(val_sparse['ctrl_t'], dtype=torch.float32),
        'dt': torch.tensor(val_sparse['dt'], dtype=torch.float32),
        'y_true': val_sparse['y_raw']
    }
    
    kwargs = {'in_features': 56, 'u_dim': 3, 'latent_dim': 16, 'alpha': 0.10, 'gamma0': 0.15, 'integrator': 'strang', 'damped': True, 'has_physics': True}
    
    print("\n[Protocol B across ALL 10 Original Seeds (Fixed Validation Windows)]")
    rmses_10_fixed = []
    for s in SEEDS_10:
        r, _ = train_eval_trial(
            CT_PhysProb_JEPA_IntegratorAblation, kwargs, lr=3.5e-3, tr_data=tr_data_56, va_data=va_data_56, stats_dict=stats1, seed=s, use_swa=True
        )
        rmses_10_fixed.append(r)
        print(f"Seed {s:4d}: {r:.4f}")
    
    sd_10_fixed = float(np.std(rmses_10_fixed))
    mean_10_fixed = float(np.mean(rmses_10_fixed))
    print(f"--> 10-Seed Fixed-Window SD: {sd_10_fixed:.4f} cycles (Mean = {mean_10_fixed:.3f})")
    
    print("\n" + "=" * 80)
    print("SEED CONTROL MECHANISMS:")
    print("1. Initialization (torch.manual_seed(seed)): Sets all initial layer weights and biases.")
    print("2. Batch Shuffling (torch.randperm(N)): Sets minibatch sequence order per epoch.")
    print("3. Data Shuffling / Window Masking: In Protocol A, seed controlled build_seq_data stride onset (seed+4000). In Protocol B, validation windows are held fixed across all seeds (seed=4042), so SD reflects purely training stochasticity.")
    print("=" * 80)

if __name__ == '__main__':
    main()
