#!/usr/bin/env python3
"""
STEP 1: ERROR DECOMPOSITION ON VALIDATION UNITS (10 REAL SEEDS)
Analyzes where CT_PhysProb_JEPA falls short by decomposing total long-gap error into:
  (a) Anchor Error: Model's in-distribution error at the blackout start (t_k).
  (b) Gap-Induced Dynamic Error: Long-gap prediction error minus anchor error.
  (c) EstimatedAnchor_Countdown: Deployable countdown clip(model_anchor_pred - dt, 0, 125).
  (d) Oracle_Countdown: Upper non-deployable reference clip(true_y_curr - dt, 0, 125).

STRICT PROTOCOL:
- 10 real seeds: [42, 100, 2024, 777, 999, 101, 202, 303, 404, 505]
- Evaluated strictly on 20 validation units (FD001 test engines untouched).
- All numbers logged directly to results/raw/step1_error_decomposition_raw.csv.
"""

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn as nn

from scripts.run_ood_and_small_data import load_cmapss_subset, build_seq_data, nasa_score
from scripts.run_step1_diagnosis import (
    to_tensor_data,
    CT_PhysProb_JEPA,
    Plain_2ndOrder_NODE_Kin,
    GRU_D,
    PatchTST_RUL,
    REGIMES,
    SEEDS_10
)
from scripts.run_step2_improvements import GRU_D_v2, PatchTST_RUL_v2, train_model_step2
from scripts.run_hybrid_ablation_10seeds import CT_PhysProb_JEPA_KinGate


def run_error_decomposition():
    print("=" * 80)
    print("STEP 1: ERROR DECOMPOSITION & ESTIMATED ANCHOR COUNTDOWN (10 SEEDS, VAL ONLY)")
    print("=" * 80)

    os.makedirs(ROOT / "results" / "raw", exist_ok=True)
    raw_csv = ROOT / "results" / "raw" / "step1_error_decomposition_raw.csv"
    summary_csv = ROOT / "results" / "step1_error_decomposition_summary.csv"

    # Load official NASA C-MAPSS FD001 (Train: 80, Val: 20)
    df_tr, df_va, _, stats1 = load_cmapss_subset('FD001')
    print(f"Loaded FD001: Train units={df_tr['unit'].nunique()}, Val units={df_va['unit'].nunique()} (Test units untouched)")

    models_to_evaluate = [
        ('CT_PhysProb_JEPA_KinGate', CT_PhysProb_JEPA_KinGate, dict(), True, 'asym', 1.0),
        ('GRU_D_v2',                 GRU_D_v2,                 dict(), False, 'asym', 1.0),
        ('PatchTST_RUL_v2',          PatchTST_RUL_v2,          dict(), False, 'asym', 1.0),
        ('Plain_2ndOrder_NODE_Kin',  Plain_2ndOrder_NODE_Kin,  dict(latent_dim=16), True, 'mse', 0.0),
        ('CT_PhysProb_JEPA_Adaptive',CT_PhysProb_JEPA,         dict(latent_dim=16, clamped=False), True, 'asym', 1.0),
        ('CT_PhysProb_JEPA_Clamped', CT_PhysProb_JEPA,         dict(latent_dim=16, clamped=True),  True, 'asym', 1.0),
        ('GRU_D',                    GRU_D,                    dict(hidden_dim=48), False, 'mse', 0.0),
        ('PatchTST_RUL',             PatchTST_RUL,             dict(d_model=48), False, 'mse', 0.0),
    ]

    records = []

    for s_idx, s in enumerate(SEEDS_10):
        t0_s = time.time()
        print(f"\n--- Seed {s} ({s_idx + 1}/10) ---", flush=True)

        # 1. Training data (mixed gaps: 50% short, 50% stride 3-15)
        tr_short = build_seq_data(df_tr, stats1, drop_prob=0.25, seed=s)
        tr_long  = build_seq_data(df_tr, stats1, fixed_stride_range=(3, 15), seed=s + 1000)
        tr_mixed = {
            k: np.concatenate([tr_short[k], tr_long[k]], axis=0)
            for k in ['obs_t', 'ctrl_t', 'dt', 'obs_next', 'y_norm', 'y_raw', 'mem_fast', 'mem_slow']
        }
        x_tr, u_tr, dt_tr, y_tr = to_tensor_data(tr_mixed)

        # Unaugmented training data for standard baselines
        x_tr_std, u_tr_std, dt_tr_std, y_tr_std = to_tensor_data(tr_short)

        # 2. Validation regimes (20 validation units ONLY)
        val_regimes = {
            'InDist_1x_2x': build_seq_data(df_va, stats1, drop_prob=0.25, seed=s + 2000),
            'OOD_5x_10x':   build_seq_data(df_va, stats1, fixed_stride_range=(5, 10), seed=s + 3000),
            'OOD_10x_20x':  build_seq_data(df_va, stats1, fixed_stride_range=(10, 20), seed=s + 4000),
            'OOD_15x_25x':  build_seq_data(df_va, stats1, fixed_stride_range=(15, 25), seed=s + 5000),
        }
        for vd in val_regimes.values():
            dt_cyc = vd['dt'][:, 0] * 4.0
            vd['y_curr'] = np.minimum(125.0, vd['y_raw'] + dt_cyc)

        # 3. Reference: TimeCorrected_LOCF_Oracle (Non-deployable ceiling)
        for r_name in REGIMES:
            vd = val_regimes[r_name]
            dt_cyc = vd['dt'][:, 0] * 4.0
            y_pred_oracle = np.clip(vd['y_curr'] - dt_cyc, 0.0, 125.0)
            y_true = vd['y_raw']
            
            true_slope = (y_true - vd['y_curr']) / np.maximum(1.0, dt_cyc)
            mask_act = (true_slope <= -0.90)

            records.append({
                'Seed': s, 'Model': 'TimeCorrected_LOCF_Oracle', 'Regime': r_name,
                'Anchor_RMSE': 0.0,
                'Anchor_MAE': 0.0,
                'Total_RMSE': float(np.sqrt(np.mean((y_pred_oracle - y_true)**2))),
                'Total_MAE': float(np.mean(np.abs(y_pred_oracle - y_true))),
                'Gap_Induced_RMSE': float(np.sqrt(np.mean((y_pred_oracle - y_true)**2))),
                'EstAnchor_Countdown_RMSE': float(np.sqrt(np.mean((y_pred_oracle - y_true)**2))),
                'Signed_Bias_ActiveDeg': float(np.mean(y_pred_oracle[mask_act] - y_true[mask_act])) if np.sum(mask_act) > 0 else 0.0,
                'NASA_Score': nasa_score(y_pred_oracle, y_true),
                'Is_Deployable': False
            })

        # 4. Train each learned model
        for m_name, m_cls, m_kw, has_kin, l_type, l_opt in models_to_evaluate:
            use_mixed = ('v2' in m_name or 'KinGate' in m_name or 'Adaptive' in m_name or 'Clamped' in m_name or 'Plain' in m_name)
            curr_xtr = x_tr if use_mixed else x_tr_std
            curr_utr = u_tr if use_mixed else u_tr_std
            curr_dttr = dt_tr if use_mixed else dt_tr_std
            curr_ytr = y_tr if use_mixed else y_tr_std

            model, tr_sec = train_model_step2(
                m_cls, m_kw, curr_xtr, curr_utr, curr_dttr, curr_ytr,
                seed=s, epochs=8, has_kin=has_kin, loss_type=l_type, lambda_opt=l_opt
            )

            # Evaluate on each regime
            for r_name in REGIMES:
                vd = val_regimes[r_name]
                vx, vu, vdt, _ = to_tensor_data(vd)
                y_true = vd['y_raw']
                y_curr = vd['y_curr']
                dt_cyc = vd['dt'][:, 0] * 4.0
                
                # A. Anchor Prediction: evaluated at dt_in = 0.25 (nominal 1 cycle / gap=0)
                vdt_zero = torch.full_like(vdt, 0.25)
                with torch.no_grad():
                    y_anch_norm, _, _ = model(vx, vu, vdt_zero)
                    y_anch_pred = torch.clamp(y_anch_norm[:, 0] * stats1['tgt_std'] + stats1['tgt_mean'], 0.0, 125.0).numpy()
                
                anchor_rmse = float(np.sqrt(np.mean((y_anch_pred - y_curr)**2)))
                anchor_mae  = float(np.mean(np.abs(y_anch_pred - y_curr)))

                # B. Integrated Prediction: evaluated across the actual blackout duration
                with torch.no_grad():
                    y_pred_norm, _, _ = model(vx, vu, vdt)
                    y_pred_integ = torch.clamp(y_pred_norm[:, 0] * stats1['tgt_std'] + stats1['tgt_mean'], 0.0, 125.0).numpy()

                total_rmse = float(np.sqrt(np.mean((y_pred_integ - y_true)**2)))
                total_mae  = float(np.mean(np.abs(y_pred_integ - y_true)))
                gap_induced_rmse = total_rmse - anchor_rmse

                # C. Deployable Reference: EstimatedAnchor_Countdown
                # Uses model's OWN anchor prediction minus elapsed cycles
                y_est_countdown = np.clip(y_anch_pred - dt_cyc, 0.0, 125.0)
                est_cd_rmse = float(np.sqrt(np.mean((y_est_countdown - y_true)**2)))

                # D. Active degradation bias & NASA score
                true_slope = (y_true - y_curr) / np.maximum(1.0, dt_cyc)
                mask_act = (true_slope <= -0.90)

                records.append({
                    'Seed': s, 'Model': m_name, 'Regime': r_name,
                    'Anchor_RMSE': anchor_rmse,
                    'Anchor_MAE': anchor_mae,
                    'Total_RMSE': total_rmse,
                    'Total_MAE': total_mae,
                    'Gap_Induced_RMSE': gap_induced_rmse,
                    'EstAnchor_Countdown_RMSE': est_cd_rmse,
                    'Signed_Bias_ActiveDeg': float(np.mean(y_pred_integ[mask_act] - y_true[mask_act])) if np.sum(mask_act) > 0 else 0.0,
                    'NASA_Score': nasa_score(y_pred_integ, y_true),
                    'Is_Deployable': True
                })

        print(f"Seed {s} finished in {time.time() - t0_s:.1f}s.", flush=True)
        pd.DataFrame(records).to_csv(raw_csv, index=False)

    df_res = pd.DataFrame(records)
    df_res.to_csv(raw_csv, index=False)
    print(f"\nRaw decomposition saved to {raw_csv}")

    # Summary table
    summary = df_res.groupby(['Regime', 'Model']).agg({
        'Anchor_RMSE': ['mean', 'std'],
        'Total_RMSE': ['mean', 'std'],
        'Gap_Induced_RMSE': ['mean', 'std'],
        'EstAnchor_Countdown_RMSE': ['mean', 'std'],
        'Signed_Bias_ActiveDeg': ['mean', 'std'],
        'NASA_Score': ['mean', 'std']
    }).round(2)
    summary.to_csv(summary_csv)
    print(f"Summary decomposition saved to {summary_csv}")

    return df_res, summary


if __name__ == '__main__':
    run_error_decomposition()
