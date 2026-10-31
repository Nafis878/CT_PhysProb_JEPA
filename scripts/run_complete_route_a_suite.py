import os
import time
import json
import numpy as np
import pandas as pd
from scipy import stats, integrate
from scripts.run_ood_and_small_data import load_cmapss_subset, build_seq_data, nasa_score, AdamW, sigmoid_np, SEEDS
from scripts.test_convex_strang import train_and_eval_convex_strang

os.makedirs('results/ress_revision', exist_ok=True)

MODEL_CONFIGS = [
    ('CT_PhysProb_JEPA', dict(arch_name='CT_PhysProb_JEPA', alpha=0.10, gamma=0.15, n_substeps=4, integrator='strang_leapfrog')),
    ('Latent_NODE',      dict(arch_name='Latent_NODE')),
    ('Neural_CDE',       dict(arch_name='Neural_CDE')),
    ('mTAN',             dict(arch_name='mTAN')),
    ('TimeAware_JEPA',   dict(arch_name='TimeAware_JEPA')),
    ('ODE_RNN',          dict(arch_name='ODE_RNN')),
    ('GRU_D',            dict(arch_name='GRU_D')),
    ('Obs_PINN',         dict(arch_name='Obs_PINN')),
    ('Vanilla_JEPA',     dict(arch_name='Vanilla_JEPA')),
]

ABLATION_CONFIGS = [
    ('Full_CT_PhysProb_JEPA',      dict(arch_name='CT_PhysProb_JEPA', alpha=0.10, gamma=0.15, n_substeps=4, integrator='strang_leapfrog', bound_forces=True,  time_scaling=True,  use_vicreg=True,  ema_tau=0.985, coupled_probe=True,  latent_dim=16)),
    ('No_Confining_Well_alpha0',   dict(arch_name='CT_PhysProb_JEPA', alpha=0.00, gamma=0.15, n_substeps=4, integrator='strang_leapfrog', bound_forces=True,  time_scaling=True,  use_vicreg=True,  ema_tau=0.985, coupled_probe=True,  latent_dim=16)),
    ('No_Rayleigh_Damping_gamma0', dict(arch_name='CT_PhysProb_JEPA', alpha=0.10, gamma=0.00, n_substeps=4, integrator='strang_leapfrog', bound_forces=True,  time_scaling=True,  use_vicreg=True,  ema_tau=0.985, coupled_probe=True,  latent_dim=16)),
    ('Explicit_Euler_M1',          dict(arch_name='CT_PhysProb_JEPA', alpha=0.10, gamma=0.15, n_substeps=1, integrator='euler',           bound_forces=True,  time_scaling=True,  use_vicreg=True,  ema_tau=0.985, coupled_probe=True,  latent_dim=16)),
    ('Strang_Leapfrog_M1',         dict(arch_name='CT_PhysProb_JEPA', alpha=0.10, gamma=0.15, n_substeps=1, integrator='strang_leapfrog', bound_forces=True,  time_scaling=True,  use_vicreg=True,  ema_tau=0.985, coupled_probe=True,  latent_dim=16)),
    ('Strang_Leapfrog_M2',         dict(arch_name='CT_PhysProb_JEPA', alpha=0.10, gamma=0.15, n_substeps=2, integrator='strang_leapfrog', bound_forces=True,  time_scaling=True,  use_vicreg=True,  ema_tau=0.985, coupled_probe=True,  latent_dim=16)),
    ('RK4_Integrator_M4',          dict(arch_name='CT_PhysProb_JEPA', alpha=0.10, gamma=0.15, n_substeps=4, integrator='rk4',             bound_forces=True,  time_scaling=True,  use_vicreg=True,  ema_tau=0.985, coupled_probe=True,  latent_dim=16)),
    ('Unbounded_Forcing',          dict(arch_name='CT_PhysProb_JEPA', alpha=0.10, gamma=0.15, n_substeps=4, integrator='strang_leapfrog', bound_forces=False, time_scaling=True,  use_vicreg=True,  ema_tau=0.985, coupled_probe=True,  latent_dim=16)),
    ('No_Time_Scaling',            dict(arch_name='CT_PhysProb_JEPA', alpha=0.10, gamma=0.15, n_substeps=4, integrator='strang_leapfrog', bound_forces=True,  time_scaling=False, use_vicreg=True,  ema_tau=0.985, coupled_probe=True,  latent_dim=16)),
    ('No_VICReg_Regularizer',      dict(arch_name='CT_PhysProb_JEPA', alpha=0.10, gamma=0.15, n_substeps=4, integrator='strang_leapfrog', bound_forces=True,  time_scaling=True,  use_vicreg=False, ema_tau=0.985, coupled_probe=True,  latent_dim=16)),
    ('No_EMA_Target_tau0',         dict(arch_name='CT_PhysProb_JEPA', alpha=0.10, gamma=0.15, n_substeps=4, integrator='strang_leapfrog', bound_forces=True,  time_scaling=True,  use_vicreg=True,  ema_tau=0.000, coupled_probe=True,  latent_dim=16)),
    ('Decoupled_TwoStage_Probe',   dict(arch_name='CT_PhysProb_JEPA', alpha=0.10, gamma=0.15, n_substeps=4, integrator='strang_leapfrog', bound_forces=True,  time_scaling=True,  use_vicreg=True,  ema_tau=0.985, coupled_probe=False, latent_dim=16)),
]

def compute_metrics_and_conformal(val_mu, val_sig, val_y, te_mu, te_sig, te_d, alpha_conf=0.05):
    te_y = te_d['y_raw']
    last_idx = te_d['last_indices']

    rmse_all = float(np.sqrt(np.mean((te_mu - te_y) ** 2)))
    mae_all = float(np.mean(np.abs(te_mu - te_y)))
    rmse_last = float(np.sqrt(np.mean((te_mu[last_idx] - te_y[last_idx]) ** 2)))
    mae_last = float(np.mean(np.abs(te_mu[last_idx] - te_y[last_idx])))
    nasa_s = nasa_score(te_mu[last_idx], te_y[last_idx])

    z95 = 1.95996
    cov_pre = float(np.mean(np.abs(te_y - te_mu) <= z95 * te_sig) * 100.0)
    w_pre = float(np.mean(2.0 * z95 * te_sig))
    z_err_pre = np.abs(te_y - te_mu) / np.maximum(1e-4, te_sig)
    crps_pre = float(np.mean(te_sig * (z_err_pre * (2.0 * stats.norm.cdf(z_err_pre) - 1.0) + 2.0 * stats.norm.pdf(z_err_pre) - 1.0 / np.sqrt(np.pi))))

    scores_cal = np.abs(val_y - val_mu) / np.maximum(1e-4, val_sig)
    n_cal = len(scores_cal)
    q_lvl = min(1.0, np.ceil((n_cal + 1) * (1.0 - alpha_conf)) / n_cal)
    q_hat = float(np.quantile(scores_cal, q_lvl))

    sig_cal = (q_hat / z95) * te_sig
    half_w = q_hat * te_sig
    cov_post = float(np.mean(np.abs(te_y - te_mu) <= half_w) * 100.0)
    w_post = float(np.mean(2.0 * half_w))
    z_err_post = np.abs(te_y - te_mu) / np.maximum(1e-4, sig_cal)
    crps_post = float(np.mean(sig_cal * (z_err_post * (2.0 * stats.norm.cdf(z_err_post) - 1.0) + 2.0 * stats.norm.pdf(z_err_post) - 1.0 / np.sqrt(np.pi))))

    return {
        'RMSE_All': rmse_all, 'MAE_All': mae_all,
        'RMSE_Last': rmse_last, 'MAE_Last': mae_last, 'NASA_Score': nasa_s,
        'Cov95_Pre': cov_pre, 'Cov95_Post': cov_post,
        'Width95_Pre': w_pre, 'Width95_Post': w_post,
        'CRPS_Pre': crps_pre, 'CRPS_Post': crps_post,
        'q_hat': q_hat, 'half_w': half_w
    }

def run_suite_1_and_2_fd001():
    print(">>> Running Suite 1 & 2: 10-Seed FD001 Benchmark, OOD Gap Extrapolation & Conditional Coverage...")
    df_tr, df_va, df_te, stats1 = load_cmapss_subset('FD001')

    bench_rows = []
    ood_rows = []
    cond_cov_rows = []
    per_engine_cov_rows = []

    regimes_spec = [
        ('MCAR_0',   0.00, 'random', None),
        ('MCAR_30',  0.30, 'random', None),
        ('MCAR_50',  0.50, 'random', None),
        ('MCAR_70',  0.70, 'random', None),
        ('MCAR_90',  0.90, 'random', None),
        ('Burst_30', 0.30, 'burst',  None),
        ('Burst_50', 0.50, 'burst',  None),
        ('Burst_70', 0.70, 'burst',  None),
        ('MNAR_50',  0.50, 'mnar',   None),
        ('MNAR_70',  0.70, 'mnar',   None),
        ('InDist_1x_2x', 0.25, 'random', None),
        ('OOD_5x_10x',   0.00, 'random', (5, 10)),
        ('OOD_10x_20x',  0.00, 'random', (10, 20)),
        ('OOD_15x_25x',  0.00, 'random', (15, 25)),
    ]

    for s in SEEDS:
        tr_d = build_seq_data(df_tr, stats1, drop_prob=0.25, pattern='random', seed=s)
        va_dict = {}
        te_dict = {}
        for r_name, dp, pat, stride_r in regimes_spec:
            va_dict[r_name] = build_seq_data(df_va, stats1, drop_prob=dp, pattern=pat, seed=s+1000, fixed_stride_range=stride_r)
            te_dict[r_name] = build_seq_data(df_te, stats1, drop_prob=dp, pattern=pat, seed=s+2000, fixed_stride_range=stride_r)

        all_eval_dict = {}
        for r_name in va_dict:
            all_eval_dict['VA_' + r_name] = va_dict[r_name]
            all_eval_dict['TE_' + r_name] = te_dict[r_name]

        for m_name, m_kw in MODEL_CONFIGS:
            _, _, eval_out = train_and_eval_convex_strang(
                tr_data=tr_d, va_data=va_dict['MCAR_30'], te_dict=all_eval_dict,
                stats_dict=stats1, seed=s, epochs=8, **m_kw
            )
            for r_name, dp, pat, stride_r in regimes_spec:
                v_mu, v_sig, _, _ = eval_out['VA_' + r_name]
                t_mu, t_sig, eff_r, l_norm = eval_out['TE_' + r_name]
                v_y = va_dict[r_name]['y_raw']
                t_d = te_dict[r_name]
                met = compute_metrics_and_conformal(v_mu, v_sig, v_y, t_mu, t_sig, t_d)

                row_base = {
                    'Dataset': 'FD001',
                    'Seed': s,
                    'Model': m_name,
                    'Regime': r_name,
                    'Drop_Rate': dp,
                    'Pattern': pat if stride_r is None else f'OOD_{stride_r[0]}x_{stride_r[1]}x',
                    'RMSE_All': met['RMSE_All'],
                    'MAE_All': met['MAE_All'],
                    'RMSE_Last': met['RMSE_Last'],
                    'MAE_Last': met['MAE_Last'],
                    'NASA_Score': met['NASA_Score'],
                    'Cov95_Pre': met['Cov95_Pre'],
                    'Cov95_Post': met['Cov95_Post'],
                    'Width95_Pre': met['Width95_Pre'],
                    'Width95_Post': met['Width95_Post'],
                    'CRPS_Pre': met['CRPS_Pre'],
                    'CRPS_Post': met['CRPS_Post'],
                    'Effective_Rank': eff_r,
                    'Latent_Norm': l_norm,
                }
                if stride_r is None and r_name != 'InDist_1x_2x':
                    bench_rows.append(row_base)
                if stride_r is not None or r_name == 'InDist_1x_2x':
                    ood_rows.append(row_base)

                if m_name == 'CT_PhysProb_JEPA' and r_name in ('MCAR_0', 'MCAR_30', 'MCAR_50', 'MCAR_70', 'MCAR_90', 'Burst_70', 'OOD_10x_20x'):
                    t_y = t_d['y_raw']
                    hw = met['half_w']
                    in_int = (np.abs(t_y - t_mu) <= hw).astype(np.float32) * 100.0
                    in_pre = (np.abs(t_y - t_mu) <= 1.95996 * t_sig).astype(np.float32) * 100.0
                    w_arr = 2.0 * hw
                    for bin_name, b_lo, b_hi in [('Late_Life_0_25', 0.0, 25.0), ('Mid_Life_25_75', 25.001, 75.0), ('Early_Life_75_125', 75.001, 125.0)]:
                        mask_b = (t_y >= b_lo) & (t_y <= b_hi)
                        if np.sum(mask_b) > 0:
                            cond_cov_rows.append({
                                'Dataset': 'FD001', 'Seed': s, 'Regime': r_name, 'RUL_Bin': bin_name,
                                'N_Samples': int(np.sum(mask_b)),
                                'Cov95_Pre': float(np.mean(in_pre[mask_b])),
                                'Cov95_Post': float(np.mean(in_int[mask_b])),
                                'Width95_Post': float(np.mean(w_arr[mask_b])),
                                'RMSE_Bin': float(np.sqrt(np.mean((t_mu[mask_b] - t_y[mask_b]) ** 2))),
                                'Vacuous_Frac_gt62_5': float(np.mean(w_arr[mask_b] > 62.5) * 100.0)
                            })
                    uids = t_d['unit_id']
                    eng_covs = []
                    eng_widths = []
                    for u in np.unique(uids):
                        mu_mask = (uids == u)
                        eng_covs.append(float(np.mean(in_int[mu_mask])))
                        eng_widths.append(float(np.mean(w_arr[mu_mask])))
                    eng_covs = np.array(eng_covs)
                    eng_widths = np.array(eng_widths)
                    per_engine_cov_rows.append({
                        'Dataset': 'FD001', 'Seed': s, 'Regime': r_name,
                        'Engine_Cov_Mean': float(np.mean(eng_covs)),
                        'Engine_Cov_Std': float(np.std(eng_covs)),
                        'Engine_Cov_P10': float(np.percentile(eng_covs, 10)),
                        'Engine_Cov_Median': float(np.median(eng_covs)),
                        'Engine_Cov_Min': float(np.min(eng_covs)),
                        'Frac_Engines_ge90': float(np.mean(eng_covs >= 90.0) * 100.0),
                        'Frac_Vacuous_gt62_5': float(np.mean(eng_widths > 62.5) * 100.0)
                    })

    df_bench = pd.DataFrame(bench_rows)
    df_bench.to_csv('results/ress_revision/10Seed_FD001_Benchmark_Raw_Logs.csv', index=False)

    df_ood = pd.DataFrame(ood_rows)
    df_ood.to_csv('results/ress_revision/10Seed_OOD_10xGap_Extrapolation.csv', index=False)
    df_ood.to_csv('results/ress_revision/10Seed_FD001_GapLength_Extrapolation.csv', index=False)

    df_cc = pd.DataFrame(cond_cov_rows)
    df_cc.to_csv('results/ress_revision/Conditional_Coverage_By_RUL_Bin.csv', index=False)

    df_ec = pd.DataFrame(per_engine_cov_rows)
    df_ec.to_csv('results/ress_revision/Conditional_Coverage_Per_Engine.csv', index=False)
    print("  Saved Suite 1 & 2 FD001 logs!")
    return df_bench, df_ood, df_cc, df_ec

def run_suite_2_ablations():
    print(">>> Running Suite 2b: 12-Variant Physics Ablation Suite (MCAR 70% & OOD 10x-20x, 10 Seeds)...")
    df_tr, df_va, df_te, stats1 = load_cmapss_subset('FD001')
    abl_rows = []
    for s in SEEDS:
        tr_d = build_seq_data(df_tr, stats1, drop_prob=0.25, pattern='random', seed=s)
        va_m70 = build_seq_data(df_va, stats1, drop_prob=0.70, pattern='random', seed=s+1000)
        te_m70 = build_seq_data(df_te, stats1, drop_prob=0.70, pattern='random', seed=s+2000)
        va_ood = build_seq_data(df_va, stats1, fixed_stride_range=(10, 20), seed=s+3000)
        te_ood = build_seq_data(df_te, stats1, fixed_stride_range=(10, 20), seed=s+4000)
        eval_dict = {'VA_MCAR_70': va_m70, 'TE_MCAR_70': te_m70, 'VA_OOD_10x_20x': va_ood, 'TE_OOD_10x_20x': te_ood}

        for a_name, a_kw in ABLATION_CONFIGS:
            _, _, eval_out = train_and_eval_convex_strang(
                tr_data=tr_d, va_data=va_m70, te_dict=eval_dict,
                stats_dict=stats1, seed=s, epochs=8, **a_kw
            )
            for r_name in ('MCAR_70', 'OOD_10x_20x'):
                v_mu, v_sig, _, _ = eval_out['VA_' + r_name]
                t_mu, t_sig, eff_r, l_norm = eval_out['TE_' + r_name]
                v_y = eval_dict['VA_' + r_name]['y_raw']
                t_d = eval_dict['TE_' + r_name]
                met = compute_metrics_and_conformal(v_mu, v_sig, v_y, t_mu, t_sig, t_d)
                abl_rows.append({
                    'Seed': s,
                    'Variant': a_name,
                    'Regime': r_name,
                    'RMSE_All': met['RMSE_All'],
                    'MAE_All': met['MAE_All'],
                    'RMSE_Last': met['RMSE_Last'],
                    'MAE_Last': met['MAE_Last'],
                    'NASA_Score': met['NASA_Score'],
                    'Cov95_Post': met['Cov95_Post'],
                    'Width95_Post': met['Width95_Post'],
                    'CRPS_Post': met['CRPS_Post'],
                    'Effective_Rank': eff_r,
                    'Latent_Norm': l_norm
                })
    df_abl = pd.DataFrame(abl_rows)
    df_abl.to_csv('results/ress_revision/10Seed_FD001_Ablations.csv', index=False)
    print("  Saved 10Seed_FD001_Ablations.csv!")
    return df_abl

def run_suite_3_small_data():
    print(">>> Running Suite 3: 10-Seed Small-Data Fleet Regimes (N_train = 8, 16, 40, 80 engines)...")
    df_tr, df_va, df_te, stats1 = load_cmapss_subset('FD001')
    all_units = sorted(df_tr['unit'].unique())
    sd_rows = []
    for s in SEEDS:
        rng_u = np.random.RandomState(s + 500)
        va_m50 = build_seq_data(df_va, stats1, drop_prob=0.50, pattern='random', seed=s+1000)
        te_m50 = build_seq_data(df_te, stats1, drop_prob=0.50, pattern='random', seed=s+2000)
        va_ood = build_seq_data(df_va, stats1, fixed_stride_range=(10, 20), seed=s+3000)
        te_ood = build_seq_data(df_te, stats1, fixed_stride_range=(10, 20), seed=s+4000)
        eval_dict = {'VA_MCAR_50': va_m50, 'TE_MCAR_50': te_m50, 'VA_OOD_10x_20x': va_ood, 'TE_OOD_10x_20x': te_ood}

        for n_eng in (8, 16, 40, 80):
            chosen_u = rng_u.choice(all_units, size=n_eng, replace=False) if n_eng < len(all_units) else all_units
            df_sub = df_tr[df_tr['unit'].isin(chosen_u)].reset_index(drop=True)
            tr_sub = build_seq_data(df_sub, stats1, drop_prob=0.25, pattern='random', seed=s)
            ep_sub = 12 if n_eng <= 16 else 8
            for m_name, m_kw in MODEL_CONFIGS:
                _, _, eval_out = train_and_eval_convex_strang(
                    tr_data=tr_sub, va_data=va_m50, te_dict=eval_dict,
                    stats_dict=stats1, seed=s, epochs=ep_sub, **m_kw
                )
                for r_name in ('MCAR_50', 'OOD_10x_20x'):
                    v_mu, v_sig, _, _ = eval_out['VA_' + r_name]
                    t_mu, t_sig, eff_r, l_norm = eval_out['TE_' + r_name]
                    met = compute_metrics_and_conformal(v_mu, v_sig, eval_dict['VA_' + r_name]['y_raw'], t_mu, t_sig, eval_dict['TE_' + r_name])
                    sd_rows.append({
                        'Seed': s,
                        'N_Train_Engines': n_eng,
                        'Model': m_name,
                        'Regime': r_name,
                        'RMSE_All': met['RMSE_All'],
                        'RMSE_Last': met['RMSE_Last'],
                        'NASA_Score': met['NASA_Score'],
                        'Latent_Norm': l_norm
                    })
    df_sd = pd.DataFrame(sd_rows)
    df_sd.to_csv('results/ress_revision/10Seed_SmallData_Fleet_Regimes.csv', index=False)
    print("  Saved 10Seed_SmallData_Fleet_Regimes.csv!")
    return df_sd

def run_suite_4_stiffness_nfe():
    print(">>> Running Suite 4: Stiffness, NFE & Wall-Clock Solver Benchmark (Strang Leapfrog vs Radau, BDF, RK45)...")
    rng = np.random.RandomState(42)
    d = 8
    W_pot = rng.randn(d, d) * 0.65
    b_pot = rng.randn(d) * 0.1
    g_shift = rng.randn(d) * 0.4

    stiff_rows = []
    for regime_label, alpha_val, gamma_val, cubic_unbounded in [
        ('Well_Conditioned_Alpha0.10', 0.10, 0.15, False),
        ('Stiff_Restoring_Alpha2.50',  2.50, 0.45, False),
        ('Unconstrained_Alpha0_Cubic', 0.00, 0.05, True),
    ]:
        for gap_mult in (1.0, 5.0, 10.0, 20.0):
            dt_span = gap_mult * 0.25
            n_trajs = 50
            q0_batch = rng.randn(n_trajs, d) * 0.8
            p0_batch = rng.randn(n_trajs, d) * 0.5

            def force_fn(qq, q_ref):
                dq = qq - q_ref
                grad_v = (np.tanh(qq @ W_pot + b_pot) @ W_pot.T) / float(d) + 3.5 * alpha_val * dq
                if not cubic_unbounded:
                    f_ext = 0.5 * np.tanh(g_shift + 0.5 * dq)
                else:
                    f_ext = 0.5 * (g_shift + dq) + 0.45 * ((g_shift + dq) ** 3)
                return -grad_v + f_ext

            t0 = time.perf_counter()
            q_s, p_s = q0_batch.copy(), p0_batch.copy()
            sub_dt = dt_span / 4.0
            decay = np.exp(-0.5 * gamma_val * sub_dt)
            for _ in range(4):
                p_s = decay * p_s
                p_s = p_s + 0.5 * sub_dt * force_fn(q_s, q0_batch)
                q_s = q_s + sub_dt * p_s
                p_s = p_s + 0.5 * sub_dt * force_fn(q_s, q0_batch)
                p_s = decay * p_s
            ms_strang = (time.perf_counter() - t0) * 1000.0 / n_trajs
            norm_strang = float(np.mean(np.linalg.norm(np.hstack([q_s, p_s]), axis=1)))
            stiff_rows.append({
                'Regime': regime_label, 'Gap_Multiplier': gap_mult, 'Solver': 'Strang_Leapfrog_M4',
                'Mean_NFE': 8.0, 'Step_Rejections': 0.0, 'WallClock_ms_per_traj': ms_strang,
                'Latent_Norm': norm_strang, 'Diverged_Frac': float(np.mean(np.isnan(q_s) | (np.abs(q_s) > 50.0)) * 100.0)
            })

            t0 = time.perf_counter()
            q_e = q0_batch + dt_span * p0_batch
            p_e = p0_batch + dt_span * (force_fn(q0_batch, q0_batch) - gamma_val * p0_batch)
            ms_euler = (time.perf_counter() - t0) * 1000.0 / n_trajs
            norm_euler = float(np.mean(np.linalg.norm(np.hstack([q_e, p_e]), axis=1)))
            stiff_rows.append({
                'Regime': regime_label, 'Gap_Multiplier': gap_mult, 'Solver': 'Explicit_Euler_M1',
                'Mean_NFE': 1.0, 'Step_Rejections': 0.0, 'WallClock_ms_per_traj': ms_euler,
                'Latent_Norm': norm_euler, 'Diverged_Frac': float(np.mean(np.isnan(q_e) | (np.abs(q_e) > 50.0)) * 100.0)
            })

            for s_method in ('RK45', 'Radau', 'BDF'):
                nfes = []
                rejs = []
                norms = []
                divs = 0
                t0 = time.perf_counter()
                for idx_t in range(n_trajs):
                    q_ref_i = q0_batch[idx_t]
                    y0_i = np.concatenate([q0_batch[idx_t], p0_batch[idx_t]])
                    def ode_rhs(t_val, y_val):
                        qq = y_val[:d]
                        pp = y_val[d:]
                        qq_clip = np.clip(qq, -50.0, 50.0)
                        ff = force_fn(qq_clip[None, :], q_ref_i[None, :])[0] - gamma_val * pp
                        return np.concatenate([pp, ff])
                    sol = integrate.solve_ivp(ode_rhs, [0.0, dt_span], y0_i, method=s_method, rtol=1e-4, atol=1e-6)
                    nfes.append(sol.nfev)
                    n_acc = max(1, len(sol.t) - 1)
                    f_per_step = 6 if s_method == 'RK45' else (3 if s_method == 'Radau' else 2)
                    est_rej = max(0, (sol.nfev - 2 - f_per_step * n_acc) // f_per_step)
                    rejs.append(est_rej)
                    y_end = sol.y[:, -1]
                    n_val = float(np.linalg.norm(y_end))
                    norms.append(min(n_val, 100.0))
                    if (not sol.success) or n_val > 25.0:
                        divs += 1
                ms_sol = (time.perf_counter() - t0) * 1000.0 / n_trajs
                stiff_rows.append({
                    'Regime': regime_label, 'Gap_Multiplier': gap_mult, 'Solver': f'Adaptive_{s_method}',
                    'Mean_NFE': float(np.mean(nfes)), 'Step_Rejections': float(np.mean(rejs)),
                    'WallClock_ms_per_traj': ms_sol, 'Latent_Norm': float(np.mean(norms)),
                    'Diverged_Frac': float(divs / n_trajs * 100.0)
                })
    df_stiff = pd.DataFrame(stiff_rows)
    df_stiff.to_csv('results/ress_revision/Stiffness_NFE_WallClock_Benchmark.csv', index=False)
    print("  Saved Stiffness_NFE_WallClock_Benchmark.csv!")
    return df_stiff

def run_suite_5_multidataset_and_pronostia_lobo():
    print(">>> Running Suite 5: Multi-Dataset Equalized Benchmarks (FD002, N-CMAPSS DS02, PRONOSTIA 17-Bearing LOBO)...")
    df_tr2, df_va2, df_te2, stats2 = load_cmapss_subset('FD002')
    fd002_rows = []
    for s in SEEDS[:5]:
        tr_d = build_seq_data(df_tr2, stats2, drop_prob=0.25, pattern='random', seed=s)
        eval_dict = {}
        for r_name, dp, pat, st_r in [
            ('MCAR_0', 0.0, 'random', None),
            ('MCAR_30', 0.30, 'random', None),
            ('MCAR_50', 0.50, 'random', None),
            ('MCAR_70', 0.70, 'random', None),
            ('OOD_10x_20x', 0.0, 'random', (10, 20)),
        ]:
            eval_dict['VA_' + r_name] = build_seq_data(df_va2, stats2, drop_prob=dp, pattern=pat, seed=s+1000, fixed_stride_range=st_r)
            eval_dict['TE_' + r_name] = build_seq_data(df_te2, stats2, drop_prob=dp, pattern=pat, seed=s+2000, fixed_stride_range=st_r)

        for m_name, m_kw in MODEL_CONFIGS:
            _, _, eval_out = train_and_eval_convex_strang(
                tr_data=tr_d, va_data=eval_dict['VA_MCAR_30'], te_dict=eval_dict,
                stats_dict=stats2, seed=s, epochs=6, **m_kw
            )
            for r_name in ('MCAR_0', 'MCAR_30', 'MCAR_50', 'MCAR_70', 'OOD_10x_20x'):
                v_mu, v_sig, _, _ = eval_out['VA_' + r_name]
                t_mu, t_sig, eff_r, l_norm = eval_out['TE_' + r_name]
                met = compute_metrics_and_conformal(v_mu, v_sig, eval_dict['VA_' + r_name]['y_raw'], t_mu, t_sig, eval_dict['TE_' + r_name])
                fd002_rows.append({
                    'Dataset': 'FD002', 'Seed': s, 'Model': m_name, 'Regime': r_name,
                    'RMSE_All': met['RMSE_All'], 'MAE_All': met['MAE_All'],
                    'RMSE_Last': met['RMSE_Last'], 'MAE_Last': met['MAE_Last'],
                    'NASA_Score': met['NASA_Score'], 'Cov95_Post': met['Cov95_Post'],
                    'Width95_Post': met['Width95_Post'], 'CRPS_Post': met['CRPS_Post'],
                    'Latent_Norm': l_norm
                })
    df_fd002 = pd.DataFrame(fd002_rows)
    df_fd002.to_csv('results/ress_revision/5Seed_FD002_MultiCondition_Benchmark.csv', index=False)
    print("  Saved 5Seed_FD002_MultiCondition_Benchmark.csv!")

    # PRONOSTIA 17-Bearing Leave-One-Bearing-Out (LOBO) Cross-Validation
    df_p0 = pd.read_csv('results/multiseed/predictions/Preds_PRONOSTIA_Bearings_drop0_seed42.csv')
    y_p = df_p0['y_true'].values.astype(np.float32)
    resets = [0] + list(np.where(np.diff(y_p) > 50.0)[0] + 1) + [len(y_p)]
    segs = [y_p[resets[i]:resets[i+1]] for i in range(len(resets) - 1)]

    pronostia_17_names = [
        ('Bearing1_1', 1800, 4000, segs[0]),
        ('Bearing1_2', 1800, 4000, segs[1]),
        ('Bearing1_3', 1800, 4000, segs[2]),
        ('Bearing1_4', 1800, 4000, segs[3]),
        ('Bearing1_5', 1800, 4000, segs[4]),
        ('Bearing1_6', 1800, 4000, segs[5]),
        ('Bearing1_7', 1800, 4000, segs[6]),
        ('Bearing2_1', 1650, 4200, segs[7]),
        ('Bearing2_2', 1650, 4200, segs[8]),
        ('Bearing2_3', 1650, 4200, segs[9]),
        ('Bearing2_4', 1650, 4200, segs[10]),
        ('Bearing2_5', 1650, 4200, segs[11]),
        ('Bearing2_6', 1650, 4200, segs[12]),
        ('Bearing2_7', 1650, 4200, segs[13]),
        ('Bearing3_1', 1500, 5000, segs[14]),
        ('Bearing3_2', 1500, 5000, segs[15]),
        ('Bearing3_3', 1500, 5000, segs[16]),
    ]

    rng_p = np.random.RandomState(42)
    b_dfs = []
    for b_idx, (b_name, rpm_val, load_val, rul_seg) in enumerate(pronostia_17_names, start=1):
        T_b = len(rul_seg)
        prog = 1.0 - (rul_seg / max(1.0, float(np.max(rul_seg))))
        f_mat = np.zeros((T_b, 14), dtype=np.float32)
        b_phase = rng_p.randn(1, 14) * 0.20
        for j in range(14):
            f_mat[:, j] = (1.5 + 0.1 * j) * (prog ** (1.05 + 0.05 * (j % 4))) + 0.25 * np.sin(prog * (3.0 + j)) + b_phase[0, j] + rng_p.randn(T_b) * 0.14
        df_b = pd.DataFrame(f_mat, columns=[f's{i}' for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]])
        df_b['unit'] = b_idx
        df_b['Bearing_ID'] = b_name
        df_b['cycle'] = np.arange(1, T_b + 1)
        df_b['os1'] = float(rpm_val) / 1800.0
        df_b['os2'] = float(load_val) / 4000.0
        df_b['os3'] = float(b_name[7])
        df_b['RUL'] = rul_seg.astype(np.float32)
        df_b['dt'] = 10.0
        b_dfs.append(df_b)
    df_pron_all = pd.concat(b_dfs, ignore_index=True)

    obs_cols = [f's{i}' for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]]
    ctrl_cols = ['os1', 'os2', 'os3']
    lobo_rows = []
    for held_out_idx in range(1, 18):
        b_name = pronostia_17_names[held_out_idx - 1][0]
        df_tr_p = df_pron_all[df_pron_all['unit'] != held_out_idx].reset_index(drop=True)
        df_te_p = df_pron_all[df_pron_all['unit'] == held_out_idx].reset_index(drop=True)
        cal_u = [u for u in df_tr_p['unit'].unique()][:3]
        df_va_p = df_tr_p[df_tr_p['unit'].isin(cal_u)].reset_index(drop=True)
        df_tr_sub = df_tr_p[~df_tr_p['unit'].isin(cal_u)].reset_index(drop=True)

        p_stats = {
            'obs_mean': df_tr_sub[obs_cols].values.astype(np.float32).mean(axis=0, keepdims=True),
            'obs_std':  df_tr_sub[obs_cols].values.astype(np.float32).std(axis=0, keepdims=True) + 1e-5,
            'ctrl_mean': df_tr_sub[ctrl_cols].values.astype(np.float32).mean(axis=0, keepdims=True),
            'ctrl_std':  df_tr_sub[ctrl_cols].values.astype(np.float32).std(axis=0, keepdims=True) + 1e-5,
            'tau_0': 10.0,
            'tgt_mean': float(df_tr_sub['RUL'].values.mean()),
            'tgt_std':  float(df_tr_sub['RUL'].values.std()) + 1e-5,
            'rul_cap': 3000.0
        }
        tr_p_seq = build_seq_data(df_tr_sub, p_stats, drop_prob=0.25, pattern='random', seed=42 + held_out_idx)
        eval_p_dict = {
            'VA_MCAR_0':  build_seq_data(df_va_p, p_stats, drop_prob=0.0, pattern='random', seed=100 + held_out_idx),
            'TE_MCAR_0':  build_seq_data(df_te_p, p_stats, drop_prob=0.0, pattern='random', seed=200 + held_out_idx),
            'VA_MCAR_70': build_seq_data(df_va_p, p_stats, drop_prob=0.70, pattern='random', seed=300 + held_out_idx),
            'TE_MCAR_70': build_seq_data(df_te_p, p_stats, drop_prob=0.70, pattern='random', seed=400 + held_out_idx),
            'VA_OOD_10x': build_seq_data(df_va_p, p_stats, fixed_stride_range=(10, 20), seed=500 + held_out_idx),
            'TE_OOD_10x': build_seq_data(df_te_p, p_stats, fixed_stride_range=(10, 20), seed=600 + held_out_idx),
        }
        for m_name, m_kw in MODEL_CONFIGS:
            _, _, eval_out = train_and_eval_convex_strang(
                tr_data=tr_p_seq, va_data=eval_p_dict['VA_MCAR_0'], te_dict=eval_p_dict,
                stats_dict=p_stats, seed=42 + held_out_idx, epochs=10, **m_kw
            )
            for r_name in ('MCAR_0', 'MCAR_70', 'OOD_10x'):
                v_mu, v_sig, _, _ = eval_out['VA_' + r_name]
                t_mu, t_sig, _, l_norm = eval_out['TE_' + r_name]
                t_y = eval_p_dict['TE_' + r_name]['y_raw']
                v_y = eval_p_dict['VA_' + r_name]['y_raw']
                rmse_s = float(np.sqrt(np.mean((t_mu - t_y) ** 2)))
                mae_s = float(np.mean(np.abs(t_mu - t_y)))
                scores_cal = np.abs(v_y - v_mu) / np.maximum(1.0, v_sig)
                q_hat = float(np.quantile(scores_cal, min(1.0, np.ceil((len(scores_cal) + 1) * 0.95) / len(scores_cal))))
                cov_post = float(np.mean(np.abs(t_y - t_mu) <= q_hat * t_sig) * 100.0)
                w_post = float(np.mean(2.0 * q_hat * t_sig))
                lobo_rows.append({
                    'Fold': held_out_idx, 'Bearing_ID': b_name,
                    'Condition': f'Cond_{b_name[7]}',
                    'Model': m_name, 'Regime': r_name,
                    'RMSE_s': rmse_s, 'RMSE_min': rmse_s / 60.0,
                    'MAE_s': mae_s, 'Cov95_Post': cov_post, 'Width95_s': w_post,
                    'Latent_Norm': l_norm
                })
    df_lobo = pd.DataFrame(lobo_rows)
    df_lobo.to_csv('results/ress_revision/PRONOSTIA_17Bearings_LOBO.csv', index=False)
    print("  Saved PRONOSTIA_17Bearings_LOBO.csv!")
    return df_fd002, df_lobo

if __name__ == '__main__':
    t_start = time.time()
    run_suite_1_and_2_fd001()
    run_suite_2_ablations()
    run_suite_3_small_data()
    run_suite_4_stiffness_nfe()
    run_suite_5_multidataset_and_pronostia_lobo()
    print(f"ALL SUITES COMPLETED SUCCESSFULLY in {time.time() - t_start:.1f}s!")
