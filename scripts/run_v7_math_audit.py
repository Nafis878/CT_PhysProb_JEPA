#!/usr/bin/env python3
"""
Rigorous Mathematical Audit Verification Script (v7 Revision):
1. Computes exact Young's inequality constants K_Young, nu_Young, K_contr,
   and global two-trajectory UUB bounds R_inf, R_inf_star, R_UUB = R_inf + R_inf_star
   across all 10 seeds for BOTH:
   - Panel A: Default Capped Setting (alpha_eff = 0.35, c_alpha = 3.5, h_max = 0.35, eps = 0.08)
   - Panel B: Uncapped Full-Horizon Setting (alpha_eff = 0.10, c_alpha = 1.0, h_max = inf, eps = 0.05)
2. Trains the Uncapped Full-Horizon configuration end-to-end across all 10 seeds (zero post-hoc approximations)
   to extract its exact trained spectral norms ||W_pot||_2, ||W_pot,-||_2, L_V_full, L_nl_full, L_V_neg, L_nl_neg,
   directional/spectral probe constants, initial errors e_q, e_v, model residual eps_model, and empirical OOD RMSE.
"""
import numpy as np
import pandas as pd
from scripts.run_ood_and_small_data import load_cmapss_subset, build_seq_data, AdamW, sigmoid_np, SEEDS

df_tr, df_va, df_te, stats1 = load_cmapss_subset('FD001')

def train_and_measure_seed_config(tr_data, te_in, te_10x, stats_dict, seed=42,
                                  alpha=0.10, c_alpha=3.5, h_max=0.35, gamma=0.15,
                                  n_substeps=4, use_kin=True, latent_dim=16, epochs=8):
    rng = np.random.RandomState(seed)
    D = tr_data['obs_t'].shape[1]
    U = tr_data['ctrl_t'].shape[1]
    d = latent_dim // 2
    H = 48
    rul_cap = float(stats_dict.get('rul_cap', 125.0))
    out_dim = 2 * d

    in_enc = 3 * D + U
    W_enc1 = (rng.randn(in_enc, H) * np.sqrt(2.0 / in_enc)).astype(np.float32)
    b_enc1 = np.zeros((1, H), dtype=np.float32)
    W_enc2 = (rng.randn(H, out_dim) * np.sqrt(2.0 / H)).astype(np.float32)
    b_enc2 = np.zeros((1, out_dim), dtype=np.float32)

    W_tgt1, b_tgt1 = W_enc1.copy(), b_enc1.copy()
    W_tgt2, b_tgt2 = W_enc2.copy(), b_enc2.copy()

    W_pot = (rng.randn(d, d) * 0.25).astype(np.float32)
    b_pot = np.zeros((1, d), dtype=np.float32)
    mode_signs = np.array([1.0 if j % 3 != 0 else -0.65 for j in range(d)], dtype=np.float32).reshape(1, d)

    W_diss = (rng.randn(out_dim + U + 1, d) * 0.15).astype(np.float32)
    b_diss = np.zeros((1, d), dtype=np.float32)
    W_aux = (rng.randn(D, out_dim) * 0.15).astype(np.float32)
    W_gate = (rng.randn(out_dim + U + 1, out_dim) * 0.15).astype(np.float32)
    b_gate = np.full((1, out_dim), -1.2, dtype=np.float32)

    in_pr = out_dim + U
    W_pr1 = (rng.randn(in_pr, H) * np.sqrt(2.0 / in_pr)).astype(np.float32)
    b_pr1 = np.zeros((1, H), dtype=np.float32)
    W_pr2 = (rng.randn(H, 2) * 0.1).astype(np.float32)
    b_pr2 = np.zeros((1, 2), dtype=np.float32)

    params = [W_enc1, b_enc1, W_enc2, b_enc2, W_pot, b_pot, W_diss, b_diss, W_aux, W_gate, b_gate, W_pr1, b_pr1, W_pr2, b_pr2]
    opt = AdamW(params, lr=3.5e-3, wd=1e-4)
    ema_tau = 0.985

    def encode(obs, mf, ms, ctrl, w1, b1, w2, b2):
        x = np.hstack([obs, mf, ms, ctrl])
        h = np.tanh(x @ w1 + b1)
        z = h @ w2 + b2
        return x, h, z

    def forward_dynamics(z_t, ctrl_t, dt_in):
        dt_eff = np.clip(dt_in, 0.01, 10.0)
        q0 = z_t[:, :d].copy()
        q = q0.copy()
        p = z_t[:, d:].copy()
        h_int = np.minimum(dt_eff, h_max)
        sub_dt = h_int / float(n_substeps)
        ctrl_cat = np.hstack([q0, p, ctrl_t, np.minimum(dt_eff, 1.0)])
        g_shift = ctrl_cat @ W_diss + b_diss

        def force_tot(qq):
            dq = qq - q0
            grad_v_mlp = ((np.tanh(qq @ W_pot + b_pot) * mode_signs) @ W_pot.T) / float(d)
            conf_force = c_alpha * alpha * dq
            grad_v = grad_v_mlp + conf_force
            f_ext = 0.45 * np.tanh(g_shift + 0.35 * dq)
            return -grad_v + f_ext

        for _ in range(n_substeps):
            decay = np.exp(-0.5 * gamma * sub_dt)
            p = decay * p
            p = p + 0.5 * sub_dt * force_tot(q)
            q = q + sub_dt * p
            p = p + 0.5 * sub_dt * force_tot(q)
            p = decay * p
        return np.hstack([q, p]), dt_eff, force_tot(q0)

    N = len(tr_data['obs_t'])
    batch_size = min(512, max(32, N // 4))
    for ep in range(epochs):
        perm = rng.permutation(N)
        for i in range(0, max(1, N - batch_size + 1), batch_size):
            idx = perm[i:i+batch_size]
            ot = tr_data['obs_t'][idx]
            ct = tr_data['ctrl_t'][idx]
            dt_b = tr_data['dt'][idx]
            on = tr_data['obs_next'][idx]
            yn = tr_data['y_norm'][idx]
            mf = tr_data['mem_fast'][idx]
            ms = tr_data['mem_slow'][idx]

            x_c, h_c, z_t = encode(ot, mf, ms, ct, W_enc1, b_enc1, W_enc2, b_enc2)
            dt_eff_b = np.clip(dt_b, 0.01, 10.0)
            af_n = np.exp(-0.60 * dt_eff_b)
            as_n = np.exp(-0.12 * dt_eff_b)
            mf_n = af_n * mf + (1.0 - af_n) * on
            ms_n = as_n * ms + (1.0 - as_n) * on
            _, _, z_tgt = encode(on, mf_n, ms_n, ct, W_tgt1, b_tgt1, W_tgt2, b_tgt2)

            z_pred, dt_eff, _ = forward_dynamics(z_t, ct, dt_b)
            pr_in = np.hstack([z_pred, ct])
            h_pr = np.tanh(pr_in @ W_pr1 + b_pr1)
            out_pr = h_pr @ W_pr2 + b_pr2
            y_pred = out_pr[:, 0:1]
            log_var = np.clip(out_pr[:, 1:2], -4.0, 2.0)
            inv_var = np.exp(-log_var)

            err_y = y_pred - yn
            dy_pred = (err_y + 0.25 * inv_var * err_y) / len(idx)
            dlog_var = 0.125 * (1.0 - inv_var * (err_y ** 2)) / len(idx)
            dout_pr = np.hstack([dy_pred, dlog_var])

            dW_pr2 = h_pr.T @ dout_pr
            db_pr2 = dout_pr.sum(axis=0, keepdims=True)
            dh_pr = (dout_pr @ W_pr2.T) * (1.0 - h_pr ** 2)
            dW_pr1 = pr_in.T @ dh_pr
            db_pr1 = dh_pr.sum(axis=0, keepdims=True)

            dpr_in = dh_pr @ W_pr1.T
            dz_probe = dpr_in[:, :out_dim]
            dz_jepa = 0.5 * (z_pred - z_tgt) / len(idx)

            if use_kin:
                q_t = z_t[:, :d]
                p_t = z_t[:, d:]
                q_tgt = z_tgt[:, :d]
                v_sec = (q_tgt - q_t) / np.maximum(1.0, 4.0 * dt_eff)
                dp_kin = 0.30 * (p_t - v_sec) / len(idx)
                dz_kin = np.hstack([np.zeros_like(q_t), dp_kin])
            else:
                dz_kin = 0.0

            dz_total = dz_jepa + dz_probe + dz_kin
            z_cat = np.hstack([z_t, ct, np.minimum(dt_eff, 1.0)])
            dq_tot = dz_total[:, :d]
            dp_tot = dz_total[:, d:] + 0.25 * dq_tot
            dW_diss = z_cat.T @ (dp_tot * 0.20)
            db_diss = (dp_tot * 0.20).sum(axis=0, keepdims=True)
            hq = z_t[:, :d] @ W_pot + b_pot
            aq = np.tanh(hq) * mode_signs
            dW_pot = -(dp_tot.T @ aq + z_t[:, :d].T @ ((dp_tot @ W_pot) * (1.0 - np.tanh(hq) ** 2) * mode_signs)) * (0.20 / float(d))
            db_pot = -((dp_tot @ W_pot) * (1.0 - np.tanh(hq) ** 2) * mode_signs).sum(axis=0, keepdims=True) * (0.20 / float(d))
            dW_gate = np.zeros_like(W_gate)
            db_gate = np.zeros_like(b_gate)
            dW_aux = ot.T @ (dz_total * 0.1)
            dW_enc2 = h_c.T @ dz_total
            db_enc2 = dz_total.sum(axis=0, keepdims=True)
            dh_c = (dz_total @ W_enc2.T) * (1.0 - h_c ** 2)
            dW_enc1 = x_c.T @ dh_c
            db_enc1 = dh_c.sum(axis=0, keepdims=True)

            opt.step([dW_enc1, db_enc1, dW_enc2, db_enc2, dW_pot, db_pot, dW_diss, db_diss, dW_aux, dW_gate, db_gate, dW_pr1, db_pr1, dW_pr2, db_pr2])
            W_tgt1 = ema_tau * W_tgt1 + (1.0 - ema_tau) * W_enc1
            b_tgt1 = ema_tau * b_tgt1 + (1.0 - ema_tau) * b_enc1
            W_tgt2 = ema_tau * W_tgt2 + (1.0 - ema_tau) * W_enc2
            b_tgt2 = ema_tau * b_tgt2 + (1.0 - ema_tau) * b_enc2

    sigma_Wpot = float(np.linalg.norm(W_pot, ord=2))
    L_V_full = float((sigma_Wpot ** 2) / float(d))
    L_V_bound = float(sigma_Wpot / np.sqrt(float(d))) # global bound ||nabla_q V_mlp(q)||_2 <= ||W_pot||_2 / sqrt(d)
    neg_cols = [j for j in range(d) if mode_signs[0, j] < 0]
    W_pot_neg = W_pot[:, neg_cols]
    sigma_Wpot_neg = float(np.linalg.norm(W_pot_neg, ord=2))
    L_V_neg = float(0.65 * (sigma_Wpot_neg ** 2) / float(d))

    c_f_beta_q = 0.45 * 0.35 # 0.1575
    L_nl_full = float(L_V_full + c_f_beta_q)
    L_nl_neg  = float(L_V_neg + c_f_beta_q)

    _, _, z_in = encode(te_in['obs_t'], te_in['mem_fast'], te_in['mem_slow'], te_in['ctrl_t'], W_enc1, b_enc1, W_enc2, b_enc2)
    z_pred_in, _, _ = forward_dynamics(z_in, te_in['ctrl_t'], te_in['dt'])
    h_pr_in = np.tanh(np.hstack([z_pred_in, te_in['ctrl_t']]) @ W_pr1 + b_pr1)
    raw_in = (h_pr_in @ W_pr2 + b_pr2)[:, 0] * stats_dict['tgt_std'] + stats_dict['tgt_mean']
    dt_in_ex = np.maximum(0.0, (np.clip(te_in['dt'][:, 0], 0.01, 10.0) - 0.25) * 4.0)
    deg_in = sigmoid_np((0.88 * rul_cap - raw_in) / (0.12 * rul_cap))
    mu_in = np.clip(raw_in - deg_in * dt_in_ex * 0.40, 0.0, rul_cap)
    eps_probe_in = float(np.sqrt(np.mean((mu_in - te_in['y_raw']) ** 2)))

    _, _, z_10 = encode(te_10x['obs_t'], te_10x['mem_fast'], te_10x['mem_slow'], te_10x['ctrl_t'], W_enc1, b_enc1, W_enc2, b_enc2)
    _, _, z_tgt_at_tk = encode(te_10x['obs_t'], te_10x['mem_fast'], te_10x['mem_slow'], te_10x['ctrl_t'], W_tgt1, b_tgt1, W_tgt2, b_tgt2)

    dt_eff_10 = np.clip(te_10x['dt'], 0.01, 10.0)
    af_10 = np.exp(-0.60 * dt_eff_10)
    as_10 = np.exp(-0.12 * dt_eff_10)
    mf_10 = af_10 * te_10x['mem_fast'] + (1.0 - af_10) * te_10x['obs_next']
    ms_10 = as_10 * te_10x['mem_slow'] + (1.0 - as_10) * te_10x['obs_next']
    _, _, z_tgt_10 = encode(te_10x['obs_next'], mf_10, ms_10, te_10x['ctrl_t'], W_tgt1, b_tgt1, W_tgt2, b_tgt2)

    z_pred_10, _, f_tot_10 = forward_dynamics(z_10, te_10x['ctrl_t'], te_10x['dt'])
    h_pr_10 = np.tanh(np.hstack([z_pred_10, te_10x['ctrl_t']]) @ W_pr1 + b_pr1)
    raw_10 = (h_pr_10 @ W_pr2 + b_pr2)[:, 0] * stats_dict['tgt_std'] + stats_dict['tgt_mean']
    dt_cycles_ex = np.maximum(0.0, (dt_eff_10[:, 0] - 0.25) * 4.0)
    deg_prob = sigmoid_np((0.88 * rul_cap - raw_10) / (0.12 * rul_cap))
    mu_10 = np.clip(raw_10 - deg_prob * dt_cycles_ex * 0.40, 0.0, rul_cap)
    emp_rmse_10 = float(np.sqrt(np.mean((mu_10 - te_10x['y_raw']) ** 2)))

    h_pr_tgt10 = np.tanh(np.hstack([z_tgt_10, te_10x['ctrl_t']]) @ W_pr1 + b_pr1)
    raw_tgt10 = (h_pr_tgt10 @ W_pr2 + b_pr2)[:, 0] * stats_dict['tgt_std'] + stats_dict['tgt_mean']
    mu_tgt10 = np.clip(raw_tgt10 - deg_prob * dt_cycles_ex * 0.40, 0.0, rul_cap)

    lat_err_rms = float(np.sqrt(np.mean(np.sum((z_pred_10 - z_tgt_10) ** 2, axis=1))))
    rul_diff_rms = float(np.sqrt(np.mean((mu_10 - mu_tgt10) ** 2)))
    L_g_dir = float(rul_diff_rms / max(1e-6, lat_err_rms))
    L_g_spec = float(stats_dict['tgt_std'] * np.linalg.norm(W_pr1[:out_dim, :], ord=2) * np.linalg.norm(W_pr2[:, 0], ord=2))

    q_10, p_10 = z_10[:, :d], z_10[:, d:]
    q_tgt_10, p_tgt_10 = z_tgt_10[:, :d], z_tgt_10[:, d:]
    v_sec_10 = (q_tgt_10 - q_10) / np.maximum(1.0, 4.0 * dt_eff_10)

    e_q_init = float(np.sqrt(np.mean(np.sum((q_10 - z_tgt_at_tk[:, :d]) ** 2, axis=1))))
    e_v_init = float(np.sqrt(np.mean(np.sum((p_10 - v_sec_10) ** 2, axis=1))))
    p_init_rms = float(np.sqrt(np.mean(np.sum(p_10 ** 2, axis=1))))
    a_sec_10 = (p_tgt_10 - p_10) / np.maximum(1.0, 4.0 * dt_eff_10)
    eps_model = float(np.sqrt(np.mean(np.sum((a_sec_10 - f_tot_10 + gamma * p_10) ** 2, axis=1))))
    mean_h_int = float(np.mean(np.minimum(dt_eff_10[:, 0], h_max)))

    return {
        'sigma_Wpot': sigma_Wpot,
        'sigma_Wpot_neg': sigma_Wpot_neg,
        'L_V_bound': L_V_bound,
        'L_V_full': L_V_full,
        'L_V_neg': L_V_neg,
        'L_nl_full': L_nl_full,
        'L_nl_neg': L_nl_neg,
        'L_g_dir': L_g_dir,
        'L_g_spec': L_g_spec,
        'e_q_init': e_q_init,
        'e_v_init': e_v_init,
        'p_init_rms': p_init_rms,
        'eps_model': eps_model,
        'eps_probe_in': eps_probe_in,
        'emp_rmse_10': emp_rmse_10,
        'mean_h_int': mean_h_int,
    }

def compute_young_and_uub(rk, rnk, alpha_eff, gamma, eps, h_k, C_symp_dt2, d=8):
    lam_min = 0.5 * ((alpha_eff + 1.0) - np.sqrt((alpha_eff - 1.0)**2 + 4.0 * eps**2))
    lam_max = 0.5 * ((alpha_eff + 1.0) + np.sqrt((alpha_eff - 1.0)**2 + 4.0 * eps**2))
    kappa_P = float(np.sqrt(lam_max / lam_min))

    # Theorem 2 UUB parameters:
    c_q = 0.5 * eps * alpha_eff
    c_p = 0.5 * (gamma - eps) - (eps * gamma**2) / alpha_eff
    mu_uub = float(2.0 * min(c_q, c_p) / lam_max)
    F_max = 0.45 * np.sqrt(float(d)) # 1.27279
    dist_coeff = (eps / alpha_eff) + 1.0 / (2.0 * (gamma - eps))
    C_F_mod = dist_coeff * ((rk['L_V_bound'] + F_max) ** 2)
    C_F_tgt = dist_coeff * ((rk['L_V_bound'] + F_max + rk['eps_model']) ** 2)
    R_inf = float(np.sqrt(2.0 * C_F_mod / (mu_uub * lam_min)))
    R_inf_star = float(np.sqrt(2.0 * C_F_tgt / (mu_uub * lam_min)))
    R_UUB_sum = R_inf + R_inf_star

    # Theorem 3 Young's inequality parameters on Eq. (36):
    # Cross-term: (eps*gamma + L_nl_full)*||dq||*||dp|| <= [(eps*gamma + L_nl_full)^2 / (4*(gamma - eps))] * ||dq||^2 + (gamma - eps)*||dp||^2
    K_Young = float(((eps * gamma + rk['L_nl_full']) ** 2) / (4.0 * (gamma - eps)) - eps * (alpha_eff - rk['L_nl_neg']))
    nu_Young = float(K_Young / (alpha_eff - eps**2))

    # Small-gain strict contraction check (requires (eps*gamma + L_nl_full)^2 < 2*eps*(gamma - eps)*(alpha_eff - L_nl_neg))
    strict_contractive = bool(((eps * gamma + rk['L_nl_full']) ** 2) < 2.0 * eps * (gamma - eps) * (alpha_eff - rk['L_nl_neg']))

    growth = float(np.exp(min(50.0, nu_Young * h_k)))
    int_fac = float((growth - 1.0) / nu_Young)

    lat_gron_kin = float(kappa_P * growth * np.sqrt(rk['e_q_init']**2 + (h_k * rk['e_v_init'])**2) + int_fac * (rk['eps_model'] + C_symp_dt2))
    lat_gron_nokin = float(kappa_P * growth * np.sqrt(rk['e_q_init']**2 + (h_k * rnk['e_v_init'])**2) + int_fac * (rk['eps_model'] + C_symp_dt2))

    lat_eff_kin = min(R_UUB_sum, lat_gron_kin)
    lat_eff_nokin = min(R_UUB_sum, lat_gron_nokin)

    b_dir_kin = float(rk['eps_probe_in'] + rk['L_g_dir'] * lat_eff_kin)
    b_dir_nokin = float(rk['eps_probe_in'] + rk['L_g_dir'] * lat_eff_nokin)
    b_spec_kin = float(rk['eps_probe_in'] + rk['L_g_spec'] * lat_eff_kin)

    return {
        'K_Young': K_Young,
        'nu_Young': nu_Young,
        'Strict_Contractive': strict_contractive,
        'R_inf': R_inf,
        'R_inf_star': R_inf_star,
        'R_UUB_sum': R_UUB_sum,
        'lat_gron_kin': lat_gron_kin,
        'lat_gron_nokin': lat_gron_nokin,
        'Dir_Bound_WithKin': b_dir_kin,
        'Dir_Bound_NoKin': b_dir_nokin,
        'Spec_Bound_WithKin': b_spec_kin,
    }

rows_default = []
rows_uncapped = []

for s in SEEDS:
    tr_d = build_seq_data(df_tr, stats1, drop_prob=0.25, pattern='random', seed=s)
    te_in = build_seq_data(df_te, stats1, drop_prob=0.25, pattern='random', seed=s+2000)
    te_10x = build_seq_data(df_te, stats1, fixed_stride_range=(10, 20), seed=s+4000)

    # 1. Default Capped Setting (alpha_eff = 0.35, c_alpha = 3.5, h_max = 0.35, eps = 0.08)
    rk_def = train_and_measure_seed_config(tr_d, te_in, te_10x, stats1, seed=s, c_alpha=3.5, h_max=0.35, use_kin=True)
    rnk_def = train_and_measure_seed_config(tr_d, te_in, te_10x, stats1, seed=s, c_alpha=3.5, h_max=0.35, use_kin=False)
    b_def = compute_young_and_uub(rk_def, rnk_def, alpha_eff=0.35, gamma=0.15, eps=0.08, h_k=0.35, C_symp_dt2=0.0140)

    rows_default.append({
        'Setting': 'Default_Capped_a035_h035',
        'Seed': s,
        'sigma_Wpot': round(rk_def['sigma_Wpot'], 4),
        'sigma_Wpot_neg': round(rk_def['sigma_Wpot_neg'], 4),
        'L_V_full': round(rk_def['L_V_full'], 4),
        'L_nl_full': round(rk_def['L_nl_full'], 4),
        'L_V_neg': round(rk_def['L_V_neg'], 4),
        'L_nl_neg': round(rk_def['L_nl_neg'], 4),
        'K_Young': round(b_def['K_Young'], 4),
        'nu_Young': round(b_def['nu_Young'], 4),
        'Strict_Contractive': b_def['Strict_Contractive'],
        'R_inf': round(b_def['R_inf'], 2),
        'R_inf_star': round(b_def['R_inf_star'], 2),
        'R_UUB_sum': round(b_def['R_UUB_sum'], 2),
        'L_g_dir': round(rk_def['L_g_dir'], 2),
        'L_g_spec': round(rk_def['L_g_spec'], 2),
        'e_q_init': round(rk_def['e_q_init'], 4),
        'e_v_WithKin': round(rk_def['e_v_init'], 4),
        'e_v_NoKin': round(rnk_def['e_v_init'], 4),
        'eps_model': round(rk_def['eps_model'], 4),
        'eps_probe_in': round(rk_def['eps_probe_in'], 2),
        'Lat_Gron_Kin': round(b_def['lat_gron_kin'], 4),
        'Lat_Gron_NoKin': round(b_def['lat_gron_nokin'], 4),
        'Dir_Bound_WithKin': round(b_def['Dir_Bound_WithKin'], 2),
        'Dir_Bound_NoKin': round(b_def['Dir_Bound_NoKin'], 2),
        'Spec_Bound_WithKin': round(b_def['Spec_Bound_WithKin'], 2),
        'Emp_OOD_RMSE': round(rk_def['emp_rmse_10'], 2),
    })

    # 2. Uncapped Full-Horizon Setting (alpha_eff = 0.10, c_alpha = 1.0, h_max = inf, eps = 0.05)
    rk_unc = train_and_measure_seed_config(tr_d, te_in, te_10x, stats1, seed=s, c_alpha=1.0, h_max=100.0, use_kin=True)
    rnk_unc = train_and_measure_seed_config(tr_d, te_in, te_10x, stats1, seed=s, c_alpha=1.0, h_max=100.0, use_kin=False)
    # In uncapped OOD 10x-20x, mean normalized horizon h_k = mean(dt_eff_10) = 3.743, sub_dt = h_k/4 = 0.936, C_symp*dt^2 = 1.83*(0.936)^2 = 1.60
    h_unc = rk_unc['mean_h_int']
    c_symp_unc = float(0.0140 * ((h_unc / 0.35) ** 2))
    b_unc = compute_young_and_uub(rk_unc, rnk_unc, alpha_eff=0.10, gamma=0.15, eps=0.05, h_k=h_unc, C_symp_dt2=c_symp_unc)

    rows_uncapped.append({
        'Setting': 'Uncapped_FullHorizon_a010_hInf',
        'Seed': s,
        'sigma_Wpot': round(rk_unc['sigma_Wpot'], 4),
        'sigma_Wpot_neg': round(rk_unc['sigma_Wpot_neg'], 4),
        'L_V_full': round(rk_unc['L_V_full'], 4),
        'L_nl_full': round(rk_unc['L_nl_full'], 4),
        'L_V_neg': round(rk_unc['L_V_neg'], 4),
        'L_nl_neg': round(rk_unc['L_nl_neg'], 4),
        'K_Young': round(b_unc['K_Young'], 4),
        'nu_Young': round(b_unc['nu_Young'], 4),
        'Strict_Contractive': b_unc['Strict_Contractive'],
        'R_inf': round(b_unc['R_inf'], 2),
        'R_inf_star': round(b_unc['R_inf_star'], 2),
        'R_UUB_sum': round(b_unc['R_UUB_sum'], 2),
        'L_g_dir': round(rk_unc['L_g_dir'], 2),
        'L_g_spec': round(rk_unc['L_g_spec'], 2),
        'e_q_init': round(rk_unc['e_q_init'], 4),
        'e_v_WithKin': round(rk_unc['e_v_init'], 4),
        'e_v_NoKin': round(rnk_unc['e_v_init'], 4),
        'eps_model': round(rk_unc['eps_model'], 4),
        'eps_probe_in': round(rk_unc['eps_probe_in'], 2),
        'Lat_Gron_Kin': round(b_unc['lat_gron_kin'], 2),
        'Lat_Gron_NoKin': round(b_unc['lat_gron_nokin'], 2),
        'Dir_Bound_WithKin': round(b_unc['Dir_Bound_WithKin'], 2),
        'Dir_Bound_NoKin': round(b_unc['Dir_Bound_NoKin'], 2),
        'Spec_Bound_WithKin': round(b_unc['Spec_Bound_WithKin'], 2),
        'Emp_OOD_RMSE': round(rk_unc['emp_rmse_10'], 2),
    })

df_def = pd.DataFrame(rows_default)
df_unc = pd.DataFrame(rows_uncapped)
df_both = pd.concat([df_def, df_unc], ignore_index=True)
df_both.to_csv('results/ress_revision/Theorem3_Empirical_Constants.csv', index=False)

print("=== PANEL A: DEFAULT CAPPED (alpha_eff = 0.35, h_max = 0.35) ===")
print(df_def.drop(columns=['Setting']).to_string(index=False))
print("\nPANEL A MEAN +/- SD:")
print(df_def.drop(columns=['Setting', 'Strict_Contractive']).agg(['mean', 'std']).round(4))

print("\n=== PANEL B: UNCAPPED FULL-HORIZON (alpha_eff = 0.10, h_max = inf, mean h_k = 3.74) ===")
print(df_unc.drop(columns=['Setting']).to_string(index=False))
print("\nPANEL B MEAN +/- SD:")
print(df_unc.drop(columns=['Setting', 'Strict_Contractive']).agg(['mean', 'std']).round(4))
