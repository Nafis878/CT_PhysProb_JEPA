"""Runs 10-Seed Systematic Ablations and Genuine Continuous-Time Flight-Hours Benchmark
(Benchmark2_CMAPSS_ContinuousTime_RUL_Degradation.csv & Official FD001).
"""
import os
import time
import json
import numpy as np
import pandas as pd
from scipy import stats

from execute_all_benchmarks_and_ablations import (
    SEEDS, df_train, df_val, df_test, TRAIN_STATS, build_seq_data, run_model_architecture, nasa_score
)

t0 = time.time()

# -------------------------------------------------------------------------
# 1. 10-Seed Systematic Ablations on FD001 (MCAR_70 & Burst_70)
# -------------------------------------------------------------------------
ABLATION_CONFIGS = [
    ('Full_CT_PhysProb_JEPA',  dict(alpha=0.10, gamma=0.15, n_substeps=4, integrator='strang_leapfrog', bound_forces=True,  time_scaling=True,  use_vicreg=True,  ema_tau=0.985, coupled_probe=False, latent_dim=16)),
    ('No_Confining_Well_a0',   dict(alpha=0.00, gamma=0.15, n_substeps=4, integrator='strang_leapfrog', bound_forces=True,  time_scaling=True,  use_vicreg=True,  ema_tau=0.985, coupled_probe=False, latent_dim=16)),
    ('No_Rayleigh_Damping_g0', dict(alpha=0.10, gamma=0.00, n_substeps=4, integrator='strang_leapfrog', bound_forces=True,  time_scaling=True,  use_vicreg=True,  ema_tau=0.985, coupled_probe=False, latent_dim=16)),
    ('Unbounded_Forces_noTanh',dict(alpha=0.10, gamma=0.15, n_substeps=4, integrator='strang_leapfrog', bound_forces=False, time_scaling=True,  use_vicreg=True,  ema_tau=0.985, coupled_probe=False, latent_dim=16)),
    ('Euler_Integrator_M1',    dict(alpha=0.10, gamma=0.15, n_substeps=1, integrator='euler',           bound_forces=True,  time_scaling=True,  use_vicreg=True,  ema_tau=0.985, coupled_probe=False, latent_dim=16)),
    ('RK4_Integrator_M4',      dict(alpha=0.10, gamma=0.15, n_substeps=4, integrator='rk4',             bound_forces=True,  time_scaling=True,  use_vicreg=True,  ema_tau=0.985, coupled_probe=False, latent_dim=16)),
    ('Leapfrog_M1_Substep',    dict(alpha=0.10, gamma=0.15, n_substeps=1, integrator='strang_leapfrog', bound_forces=True,  time_scaling=True,  use_vicreg=True,  ema_tau=0.985, coupled_probe=False, latent_dim=16)),
    ('Leapfrog_M8_Substeps',   dict(alpha=0.10, gamma=0.15, n_substeps=8, integrator='strang_leapfrog', bound_forces=True,  time_scaling=True,  use_vicreg=True,  ema_tau=0.985, coupled_probe=False, latent_dim=16)),
    ('No_Time_Scaling_rawDt',  dict(alpha=0.10, gamma=0.15, n_substeps=4, integrator='strang_leapfrog', bound_forces=True,  time_scaling=False, use_vicreg=True,  ema_tau=0.985, coupled_probe=False, latent_dim=16)),
    ('No_VICReg_AntiCollapse', dict(alpha=0.10, gamma=0.15, n_substeps=4, integrator='strang_leapfrog', bound_forces=True,  time_scaling=True,  use_vicreg=False, ema_tau=0.985, coupled_probe=False, latent_dim=16)),
    ('No_EMA_Target_tau0',     dict(alpha=0.10, gamma=0.15, n_substeps=4, integrator='strang_leapfrog', bound_forces=True,  time_scaling=True,  use_vicreg=True,  ema_tau=0.000, coupled_probe=False, latent_dim=16)),
    ('EndToEnd_Coupled_Probe', dict(alpha=0.10, gamma=0.15, n_substeps=4, integrator='strang_leapfrog', bound_forces=True,  time_scaling=True,  use_vicreg=True,  ema_tau=0.985, coupled_probe=True,  latent_dim=16)),
]

ablation_logs = []
for reg_label, pattern, drop_prob in [('MCAR_70', 'random', 0.7), ('Burst_70', 'burst', 0.7)]:
    for s in SEEDS:
        tr_d = build_seq_data(df_train, TRAIN_STATS, drop_prob, pattern, seed=s)
        va_d = build_seq_data(df_val,   TRAIN_STATS, drop_prob, pattern, seed=s + 1000)
        te_d = build_seq_data(df_test,  TRAIN_STATS, drop_prob, pattern, seed=s + 2000)
        y_te = te_d['y_raw']
        last_idx = te_d['last_indices']

        for ab_name, kw in ABLATION_CONFIGS:
            v_mu, v_sig, t_mu, t_sig, erank, lnorm = run_model_architecture(
                'CT_PhysProb_JEPA', tr_d, va_d, te_d, TRAIN_STATS, seed=s, epochs=10, **kw
            )
            rmse_all  = float(np.sqrt(np.mean((t_mu - y_te) ** 2)))
            rmse_last = float(np.sqrt(np.mean((t_mu[last_idx] - y_te[last_idx]) ** 2)))
            ablation_logs.append({
                'Seed': s,
                'Regime': reg_label,
                'Variant': ab_name,
                'RMSE_All': round(rmse_all, 4),
                'RMSE_Last': round(rmse_last, 4),
                'Eff_Rank': round(erank, 3),
                'Latent_Norm': round(lnorm, 3)
            })

df_ab = pd.DataFrame(ablation_logs)
df_ab.to_csv('results/ress_revision/10Seed_FD001_Ablations.csv', index=False)
print("=== Ablation Summary (Mean +- Std over 10 Seeds) ===")
print(df_ab.groupby(['Regime', 'Variant'])[['RMSE_All', 'RMSE_Last', 'Eff_Rank', 'Latent_Norm']].agg(['mean', 'std']).to_string())
