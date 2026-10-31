#!/usr/bin/env python3
"""
STEP 4: FINAL TEST EVALUATION (30 FIXED SEEDS)
Unblinds official NASA C-MAPSS FD001 test engines (100 units) for final evaluation.
Evaluates the frozen proposed model and all pre-registered competitors across 30 real seeds.

STRICT PROTOCOL:
- 30 fixed seeds from PREREGISTRATION.md
- Test set touched only for final evaluation.
- All numbers logged directly to results/raw/step4_final_30seeds_raw.csv.
- Conformal coverage and CBM costs computed from saved predictions only,
  calibrated strictly on validation units (20 engines) and evaluated on test units (100 engines).
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
    REGIMES
)
from scripts.run_step2_improvements import GRU_D_v2, PatchTST_RUL_v2, train_model_step2
from scripts.run_hybrid_ablation_10seeds import CT_PhysProb_JEPA_KinGate

SEEDS_30 = [
    42,  100, 2024,  777,  999,  101,  202,  303,  404,  505,
   606,  707,  808,  909,  111,  222,  333,  444,  555,  666,
   123,  234,  345,  456,  567,  678,  789,  890,  901, 1010
]


def run_final_evaluation():
    print("=" * 80)
    print("STEP 4: FINAL TEST SET EVALUATION ACROSS 30 REAL SEEDS (FD001)")
    print("=" * 80)
    
    os.makedirs(ROOT / "results" / "raw", exist_ok=True)
    raw_csv_path = ROOT / "results" / "raw" / "step4_final_30seeds_raw.csv"
    conf_cbm_csv_path = ROOT / "results" / "raw" / "step4_conformal_cbm_raw.csv"
    summary_csv_path = ROOT / "results" / "step4_final_30seeds_summary.csv"
    
    # Load official C-MAPSS FD001 data
    df_tr, df_va, df_te, stats1 = load_cmapss_subset('FD001')
    print(f"Loaded FD001: Train units={df_tr['unit'].nunique()}, Val units={df_va['unit'].nunique()}, Test units={df_te['unit'].nunique()}")
    
    # Models to evaluate
    learned_models = [
        ('CT_PhysProb_JEPA_KinGate', CT_PhysProb_JEPA_KinGate, dict(), True, 'asym', 1.0),
        ('GRU_D_v2',                 GRU_D_v2,                 dict(), False, 'asym', 1.0),
        ('PatchTST_RUL_v2',          PatchTST_RUL_v2,          dict(), False, 'asym', 1.0),
        ('Plain_2ndOrder_NODE_Kin',  Plain_2ndOrder_NODE_Kin,  dict(latent_dim=16), True, 'mse', 0.0),
        ('CT_PhysProb_JEPA_Adaptive',CT_PhysProb_JEPA,         dict(latent_dim=16, clamped=False), True, 'asym', 1.0),
        ('CT_PhysProb_JEPA_Clamped', CT_PhysProb_JEPA,         dict(latent_dim=16, clamped=True),  True, 'asym', 1.0),
        ('GRU_D',                    GRU_D,                    dict(hidden_dim=48), False, 'mse', 0.0),
        ('PatchTST_RUL',             PatchTST_RUL,             dict(d_model=48), False, 'mse', 0.0),
    ]
    
    results_records = []
    conformal_cbm_records = []
    
    start_total = time.time()
    
    for s_idx, s in enumerate(SEEDS_30):
        t0_seed = time.time()
        print(f"\n>>> Running Seed {s} ({s_idx + 1}/30) <<<", flush=True)
        
        # 1. Build training data (mixed gaps: 50% short, 50% stride 3-15)
        tr_short = build_seq_data(df_tr, stats1, drop_prob=0.25, seed=s)
        tr_long  = build_seq_data(df_tr, stats1, fixed_stride_range=(3, 15), seed=s + 1000)
        tr_mixed = {
            k: np.concatenate([tr_short[k], tr_long[k]], axis=0)
            for k in ['obs_t', 'ctrl_t', 'dt', 'obs_next', 'y_norm', 'y_raw', 'mem_fast', 'mem_slow']
        }
        x_tr, u_tr, dt_tr, y_tr = to_tensor_data(tr_mixed)
        
        # Standard unaugmented training data for standard baselines
        tr_std = tr_short
        x_tr_std, u_tr_std, dt_tr_std, y_tr_std = to_tensor_data(tr_std)
        
        # 2. Build validation data (conformal calibration)
        val_regimes = {
            'InDist_1x_2x': build_seq_data(df_va, stats1, drop_prob=0.25, seed=s + 2000),
            'OOD_5x_10x':   build_seq_data(df_va, stats1, fixed_stride_range=(5, 10), seed=s + 3000),
            'OOD_10x_20x':  build_seq_data(df_va, stats1, fixed_stride_range=(10, 20), seed=s + 4000),
            'OOD_15x_25x':  build_seq_data(df_va, stats1, fixed_stride_range=(15, 25), seed=s + 5000),
        }
        for vd in val_regimes.values():
            dt_cyc = vd['dt'][:, 0] * 4.0
            vd['y_curr'] = np.minimum(125.0, vd['y_raw'] + dt_cyc)
            
        # 3. Build test data (unblinded official test engines)
        test_regimes = {
            'InDist_1x_2x': build_seq_data(df_te, stats1, drop_prob=0.25, seed=s + 6000),
            'OOD_5x_10x':   build_seq_data(df_te, stats1, fixed_stride_range=(5, 10), seed=s + 7000),
            'OOD_10x_20x':  build_seq_data(df_te, stats1, fixed_stride_range=(10, 20), seed=s + 8000),
            'OOD_15x_25x':  build_seq_data(df_te, stats1, fixed_stride_range=(15, 25), seed=s + 9000),
        }
        for td in test_regimes.values():
            dt_cyc = td['dt'][:, 0] * 4.0
            td['y_curr'] = np.minimum(125.0, td['y_raw'] + dt_cyc)

        # 4. Evaluate Reference Baseline: TimeCorrected_LOCF_Oracle
        for r_name in REGIMES:
            td = test_regimes[r_name]
            dt_cyc = td['dt'][:, 0] * 4.0
            y_pred_locf = np.clip(td['y_curr'] - dt_cyc, 0.0, 125.0)
            y_true = td['y_raw']
            
            true_slope = (y_true - td['y_curr']) / np.maximum(1.0, dt_cyc)
            mask_act = (true_slope <= -0.90)
            l_idx = td['last_indices']
            
            results_records.append({
                'Seed': s, 'Model': 'TimeCorrected_LOCF_Oracle', 'Regime': r_name,
                'RMSE_All': float(np.sqrt(np.mean((y_pred_locf - y_true)**2))),
                'MAE_All': float(np.mean(np.abs(y_pred_locf - y_true))),
                'RMSE_Last': float(np.sqrt(np.mean((y_pred_locf[l_idx] - y_true[l_idx])**2))),
                'MAE_Last': float(np.mean(np.abs(y_pred_locf[l_idx] - y_true[l_idx]))),
                'Signed_Bias_ActiveDeg': float(np.mean(y_pred_locf[mask_act] - y_true[mask_act])) if np.sum(mask_act) > 0 else 0.0,
                'NASA_Score': nasa_score(y_pred_locf[l_idx], y_true[l_idx]),
                'Train_Sec': 0.0
            })

        # 5. Train and Evaluate each learned model
        for m_name, m_cls, m_kw, has_kin, l_type, l_opt in learned_models:
            use_mixed = ('v2' in m_name or 'KinGate' in m_name or 'Adaptive' in m_name or 'Clamped' in m_name or 'Plain' in m_name)
            curr_xtr = x_tr if use_mixed else x_tr_std
            curr_utr = u_tr if use_mixed else u_tr_std
            curr_dttr = dt_tr if use_mixed else dt_tr_std
            curr_ytr = y_tr if use_mixed else y_tr_std
            
            model, tr_sec = train_model_step2(
                m_cls, m_kw, curr_xtr, curr_utr, curr_dttr, curr_ytr,
                seed=s, epochs=8, has_kin=has_kin, loss_type=l_type, lambda_opt=l_opt
            )
            
            # Predict across all regimes
            val_preds_dict = {}
            test_preds_dict = {}
            
            for r_name in REGIMES:
                # Validation predictions (calibration set)
                vd = val_regimes[r_name]
                vx, vu, vdt, _ = to_tensor_data(vd)
                with torch.no_grad():
                    v_yp_norm, _, _ = model(vx, vu, vdt)
                    v_yp = torch.clamp(v_yp_norm[:, 0] * stats1['tgt_std'] + stats1['tgt_mean'], 0.0, 125.0).numpy()
                val_preds_dict[r_name] = v_yp
                
                # Test predictions
                td = test_regimes[r_name]
                tx, tu, tdt, _ = to_tensor_data(td)
                with torch.no_grad():
                    t_yp_norm, _, _ = model(tx, tu, tdt)
                    t_yp = torch.clamp(t_yp_norm[:, 0] * stats1['tgt_std'] + stats1['tgt_mean'], 0.0, 125.0).numpy()
                test_preds_dict[r_name] = t_yp
                
                y_true = td['y_raw']
                dt_cyc = td['dt'][:, 0] * 4.0
                true_slope = (y_true - td['y_curr']) / np.maximum(1.0, dt_cyc)
                mask_act = (true_slope <= -0.90)
                l_idx = td['last_indices']
                
                results_records.append({
                    'Seed': s, 'Model': m_name, 'Regime': r_name,
                    'RMSE_All': float(np.sqrt(np.mean((t_yp - y_true)**2))),
                    'MAE_All': float(np.mean(np.abs(t_yp - y_true))),
                    'RMSE_Last': float(np.sqrt(np.mean((t_yp[l_idx] - y_true[l_idx])**2))),
                    'MAE_Last': float(np.mean(np.abs(t_yp[l_idx] - y_true[l_idx]))),
                    'Signed_Bias_ActiveDeg': float(np.mean(t_yp[mask_act] - y_true[mask_act])) if np.sum(mask_act) > 0 else 0.0,
                    'NASA_Score': nasa_score(t_yp[l_idx], y_true[l_idx]),
                    'Train_Sec': tr_sec
                })

            # 6. Conformal Calibration & CBM Maintenance Cost on Primary Endpoint (OOD_10x_20x)
            r_cbm = 'OOD_10x_20x'
            v_yp = val_preds_dict[r_cbm]
            v_yt = val_regimes[r_cbm]['y_raw']
            t_yp = test_preds_dict[r_cbm]
            t_yt = test_regimes[r_cbm]['y_raw']
            t_last_idx = test_regimes[r_cbm]['last_indices']
            
            # Calibration quantile threshold chosen strictly on validation units
            cal_res = np.abs(v_yt - v_yp)
            n_cal = len(cal_res)
            q_level = min(1.0, np.ceil((n_cal + 1) * 0.95) / n_cal)
            q95 = float(np.quantile(cal_res, q_level))
            
            # Test conformal coverage
            lo_b = np.maximum(0.0, t_yp - q95)
            hi_b = np.minimum(125.0, t_yp + q95)
            covered = (t_yt >= lo_b) & (t_yt <= hi_b)
            picp95 = float(np.mean(covered) * 100.0)
            width95 = float(np.mean(hi_b - lo_b))
            
            # CBM Cost model on test engines (100 test engines)
            y_last = t_yt[t_last_idx]
            mu_last = t_yp[t_last_idx]
            lcb_last = lo_b[t_last_idx]
            n_eng = float(len(y_last))
            
            # Point forecast policy
            trig_pt = (mu_last <= 25.0)
            crit_missed_pt = float(np.sum((y_last <= 25.0) & (mu_last > 26.0)))
            fa_pt = float(np.sum((y_last > 55.0) & (mu_last <= 20.0)))
            wasted_pt = float(np.sum(np.maximum(0.0, y_last[trig_pt] - 15.0)))
            cost_pt = (crit_missed_pt * 100000.0 + fa_pt * 12000.0 + wasted_pt * 450.0) / n_eng
            
            conformal_cbm_records.append({
                'Seed': s, 'Model': m_name, 'Policy': 'Point_Forecast',
                'PICP95': picp95, 'Width95': width95,
                'Crit_Missed_Failures': crit_missed_pt,
                'False_Alarms': fa_pt,
                'Wasted_Life_Cycles': wasted_pt / n_eng,
                'Expected_Cost_USD': cost_pt
            })
            
            # Conformal LCB policy
            trig_lcb = (lcb_last <= 25.0)
            crit_missed_lcb = float(np.sum((y_last <= 25.0) & (lcb_last > 26.0)))
            fa_lcb = float(np.sum((y_last > 55.0) & (lcb_last <= 20.0)))
            wasted_lcb = float(np.sum(np.maximum(0.0, y_last[trig_lcb] - 15.0)))
            cost_lcb = (crit_missed_lcb * 100000.0 + fa_lcb * 12000.0 + wasted_lcb * 450.0) / n_eng
            
            conformal_cbm_records.append({
                'Seed': s, 'Model': m_name, 'Policy': 'Split_Conformal_LCB',
                'PICP95': picp95, 'Width95': width95,
                'Crit_Missed_Failures': crit_missed_lcb,
                'False_Alarms': fa_lcb,
                'Wasted_Life_Cycles': wasted_lcb / n_eng,
                'Expected_Cost_USD': cost_lcb
            })

        print(f"Seed {s} finished in {time.time() - t0_seed:.1f}s.", flush=True)
        
        # Incremental save
        pd.DataFrame(results_records).to_csv(raw_csv_path, index=False)
        pd.DataFrame(conformal_cbm_records).to_csv(conf_cbm_csv_path, index=False)

    total_time = time.time() - start_total
    print(f"\nAll 30 seeds completed in {total_time:.1f}s ({total_time / 60:.2f} mins).")
    
    # Save final summaries
    df_raw = pd.DataFrame(results_records)
    df_raw.to_csv(raw_csv_path, index=False)
    
    df_conf = pd.DataFrame(conformal_cbm_records)
    df_conf.to_csv(conf_cbm_csv_path, index=False)
    
    summary = df_raw.groupby(['Regime', 'Model']).agg({
        'RMSE_All': ['mean', 'std'],
        'MAE_All': ['mean', 'std'],
        'RMSE_Last': ['mean', 'std'],
        'NASA_Score': ['mean', 'std'],
        'Signed_Bias_ActiveDeg': ['mean', 'std']
    }).reset_index()
    summary.to_csv(summary_csv_path, index=False)
    print(f"Summaries saved to {summary_csv_path}")
    
    return df_raw, df_conf, summary


if __name__ == '__main__':
    run_final_evaluation()
