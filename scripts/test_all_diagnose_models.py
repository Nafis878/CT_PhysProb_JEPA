"""Test all Step 1 diagnostic models on seed 42 to verify implementations."""
import sys, os, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from scripts.run_ood_and_small_data import load_cmapss_subset, build_seq_data, nasa_score, AdamW, sigmoid_np
from scripts.run_ress_must_haves_suite import train_and_eval_competitor

def test_models():
    print("Testing Step 1 models on Seed 42...")
    df_tr, df_va, df_te, stats1 = load_cmapss_subset('FD001')
    
    # 1. Prepare data
    tr_d = build_seq_data(df_tr, stats1, drop_prob=0.25, seed=42)
    va_in = build_seq_data(df_va, stats1, drop_prob=0.25, seed=1042)
    va_5x = build_seq_data(df_va, stats1, fixed_stride_range=(5, 10), seed=2042)
    va_10x = build_seq_data(df_va, stats1, fixed_stride_range=(10, 20), seed=3042)
    va_15x = build_seq_data(df_va, stats1, fixed_stride_range=(15, 25), seed=4042)
    
    val_dict = {
        'InDist_1x_2x': va_in,
        'OOD_5x_10x': va_5x,
        'OOD_10x_20x': va_10x,
        'OOD_15x_25x': va_15x,
    }
    for r_name, vd in val_dict.items():
        dt_cyc = vd['dt'][:, 0] * 4.0
        vd['y_curr'] = np.minimum(125.0, vd['y_raw'] + dt_cyc)

    # Test 1: Clamped CT-PhysProb-JEPA
    t0 = time.time()
    _, _, _, _, eval_clamped = train_and_eval_competitor(
        arch_name='CT_PhysProb_JEPA', tr_data=tr_d, va_data=va_in, te_dict=val_dict, stats_dict=stats1,
        seed=42, epochs=8, alpha=0.10, gamma=0.15, n_substeps=4, integrator='strang_leapfrog', bound_forces=True
    )
    print(f"Clamped done in {time.time() - t0:.2f}s")
    for r in val_dict:
        rmse = np.sqrt(np.mean((eval_clamped[r][0] - val_dict[r]['y_raw'])**2))
        print(f"  Clamped {r} RMSE: {rmse:.2f}")

if __name__ == '__main__':
    test_models()
