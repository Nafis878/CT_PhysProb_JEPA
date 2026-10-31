#!/usr/bin/env python3
"""
v9 Mathematical Reviewer Verification Script:
1. Continuum-certified within-window residual supremum:
   eps_mod,w^sup = max_{0<=j<=100} ||r_model,w(s_j)||_2 + L_{r,w} * (h_w / 200),
   reporting:
   - Window-level distribution
   - Seed-level uniform supremum eps_mod,sup^(s) = max_w eps_mod,w^sup
   - Global 10-seed worst-case supremum eps_mod,global^sup = max_s eps_mod,sup^(s)
   - Seed-level RMS diagnostic eps_mod,RMS^(s)
2. Analytical Strang-splitting BCH global error bound E_disc^BCH(h_k, dt) = ((e^{L_f h_k} - 1)/L_f) * c_loc * dt^2
   vs. 400-substep numerical reference difference E_disc^(400).
3. Tightened Duhamel-Volterra variation-of-constants bound vs. Lyapunov-Young bound,
   and explicit evaluation of vacuousness relative to the 125-cycle RUL cap.
"""
import numpy as np
import pandas as pd
from scripts.run_ood_and_small_data import load_cmapss_subset, build_seq_data, AdamW, sigmoid_np, SEEDS

df_tr, df_va, df_te, stats1 = load_cmapss_subset('FD001')

def train_and_audit_v9(tr_data, te_in, te_10x, stats_dict, seed=42,
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

    params = [W_enc1, b_enc1, W_enc2, b_enc2, W_pot, b_pot, W_diss, b_diss, W_aux, W_gate, b_gate, W_pr1, db_pr1 := b_pr1, W_pr2, b_pr2]
    opt = AdamW(params, lr=3.5e-3, wd=1e-4)
    ema_tau = 0.985

    def encode(obs, mf, ms, ctrl, w1, b1, w2, b2):
        x = np.hstack([obs, mf, ms, ctrl])
        h = np.tanh(x @ w1 + b1)
        z = h @ w2 + b2
        return x, h, z

    def integrate_strang(z_t, ctrl_t, dt_in, M_steps):
        dt_eff = np.clip(dt_in, 0.01, 10.0)
        q0 = z_t[:, :d].copy()
        q = q0.copy()
        p = z_t[:, d:].copy()
        h_int = np.minimum(dt_eff, h_max)
        sub_dt = h_int / float(M_steps)
        ctrl_cat = np.hstack([q0, p, ctrl_t, np.minimum(dt_eff, 1.0)])
        g_shift = ctrl_cat @ W_diss + b_diss

        def force_tot(qq):
            dq = qq - q0
            grad_v_mlp = ((np.tanh(qq @ W_pot + b_pot) * mode_signs) @ W_pot.T) / float(d)
            conf_force = c_alpha * alpha * dq
            grad_v = grad_v_mlp + conf_force
            f_ext = 0.45 * np.tanh(g_shift + 0.35 * dq)
            return -grad_v + f_ext

        max_dq_traj = np.zeros(len(q0), dtype=np.float64)
        max_p_traj = np.sqrt(np.sum(p ** 2, axis=1))
        for _ in range(M_steps):
            decay = np.exp(-0.5 * gamma * sub_dt)
            p = decay * p
            p = p + 0.5 * sub_dt * force_tot(q)
            q = q + sub_dt * p
            p = p + 0.5 * sub_dt * force_tot(q)
            p = decay * p
            max_dq_traj = np.maximum(max_dq_traj, np.sqrt(np.sum((q - q0) ** 2, axis=1)))
            max_p_traj = np.maximum(max_p_traj, np.sqrt(np.sum(p ** 2, axis=1)))
        return np.hstack([q, p]), dt_eff, h_int, force_tot, float(np.max(max_dq_traj)), float(np.max(max_p_traj))

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

            z_pred, dt_eff, _, _, _, _ = integrate_strang(z_t, ct, dt_b, n_substeps)
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
            db_pr2_g = dout_pr.sum(axis=0, keepdims=True)
            dh_pr = (dout_pr @ W_pr2.T) * (1.0 - h_pr ** 2)
            dW_pr1 = pr_in.T @ dh_pr
            db_pr1_g = dh_pr.sum(axis=0, keepdims=True)

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
            db_diss_g = (dp_tot * 0.20).sum(axis=0, keepdims=True)
            hq = z_t[:, :d] @ W_pot + b_pot
            aq = np.tanh(hq) * mode_signs
            dW_pot = -(dp_tot.T @ aq + z_t[:, :d].T @ ((dp_tot @ W_pot) * (1.0 - np.tanh(hq) ** 2) * mode_signs)) * (0.20 / float(d))
            db_pot_g = -((dp_tot @ W_pot) * (1.0 - np.tanh(hq) ** 2) * mode_signs).sum(axis=0, keepdims=True) * (0.20 / float(d))
            dW_gate = np.zeros_like(W_gate)
            db_gate_g = np.zeros_like(b_gate)
            dW_aux = ot.T @ (dz_total * 0.1)
            dW_enc2 = h_c.T @ dz_total
            db_enc2_g = dz_total.sum(axis=0, keepdims=True)
            dh_c = (dz_total @ W_enc2.T) * (1.0 - h_c ** 2)
            dW_enc1 = x_c.T @ dh_c
            db_enc1_g = dh_c.sum(axis=0, keepdims=True)

            opt.step([dW_enc1, db_enc1_g, dW_enc2, db_enc2_g, dW_pot, db_pot_g, dW_diss, db_diss_g, dW_aux, dW_gate, db_gate_g, dW_pr1, db_pr1_g, dW_pr2, db_pr2_g])
            W_tgt1 = ema_tau * W_tgt1 + (1.0 - ema_tau) * W_enc1
            b_tgt1 = ema_tau * b_tgt1 + (1.0 - ema_tau) * b_enc1
            W_tgt2 = ema_tau * W_tgt2 + (1.0 - ema_tau) * W_enc2
            b_tgt2 = ema_tau * b_tgt2 + (1.0 - ema_tau) * b_enc2

    alpha_eff = float(c_alpha * alpha)
    sigma_Wpot = float(np.linalg.norm(W_pot, ord=2))
    L_V_full = float((sigma_Wpot ** 2) / float(d))
    L_V_bound = float(sigma_Wpot / np.sqrt(float(d)))
    neg_cols = [j for j in range(d) if mode_signs[0, j] < 0]
    W_pot_neg = W_pot[:, neg_cols]
    sigma_Wpot_neg = float(np.linalg.norm(W_pot_neg, ord=2))
    L_V_neg = float(0.65 * (sigma_Wpot_neg ** 2) / float(d))

    c_f = 0.45
    beta_q = 0.35
    c_f_beta_q = c_f * beta_q # 0.1575
    L_nl_full = float(L_V_full + c_f_beta_q)
    L_nl_neg  = float(L_V_neg + c_f_beta_q)
    L_F = float(alpha_eff + L_nl_full)
    M_F2 = float((4.0 / (3.0 * np.sqrt(3.0))) * ((sigma_Wpot ** 3) / float(d) + c_f * (beta_q ** 2)))

    # Evaluate on te_in
    _, _, z_in = encode(te_in['obs_t'], te_in['mem_fast'], te_in['mem_slow'], te_in['ctrl_t'], W_enc1, b_enc1, W_enc2, b_enc2)
    z_pred_in, _, _, _, _, _ = integrate_strang(z_in, te_in['ctrl_t'], te_in['dt'], n_substeps)
    h_pr_in = np.tanh(np.hstack([z_pred_in, te_in['ctrl_t']]) @ W_pr1 + b_pr1)
    raw_in = (h_pr_in @ W_pr2 + b_pr2)[:, 0] * stats_dict['tgt_std'] + stats_dict['tgt_mean']
    dt_in_ex = np.maximum(0.0, (np.clip(te_in['dt'][:, 0], 0.01, 10.0) - 0.25) * 4.0)
    deg_in = sigmoid_np((0.88 * rul_cap - raw_in) / (0.12 * rul_cap))
    mu_in = np.clip(raw_in - deg_in * dt_in_ex * 0.40, 0.0, rul_cap)
    eps_probe_rms = float(np.sqrt(np.mean((mu_in - te_in['y_raw']) ** 2)))
    eps_probe_max = float(np.max(np.abs(mu_in - te_in['y_raw'])))

    # Evaluate on te_10x
    _, _, z_10 = encode(te_10x['obs_t'], te_10x['mem_fast'], te_10x['mem_slow'], te_10x['ctrl_t'], W_enc1, b_enc1, W_enc2, b_enc2)
    _, _, z_tgt_at_tk = encode(te_10x['obs_t'], te_10x['mem_fast'], te_10x['mem_slow'], te_10x['ctrl_t'], W_tgt1, b_tgt1, W_tgt2, b_tgt2)

    dt_eff_10 = np.clip(te_10x['dt'], 0.01, 10.0)
    af_10 = np.exp(-0.60 * dt_eff_10)
    as_10 = np.exp(-0.12 * dt_eff_10)
    mf_10 = af_10 * te_10x['mem_fast'] + (1.0 - af_10) * te_10x['obs_next']
    ms_10 = as_10 * te_10x['mem_slow'] + (1.0 - as_10) * te_10x['obs_next']
    _, _, z_tgt_10 = encode(te_10x['obs_next'], mf_10, ms_10, te_10x['ctrl_t'], W_tgt1, b_tgt1, W_tgt2, b_tgt2)

    z_pred_10, _, h_int_vec, force_fn_10, max_dq_tr, max_p_tr = integrate_strang(z_10, te_10x['ctrl_t'], te_10x['dt'], n_substeps)
    z_ref_10, _, _, _, _, _ = integrate_strang(z_10, te_10x['ctrl_t'], te_10x['dt'], 400)

    # 1. Empirical 400-substep numerical reference difference
    disc_err_vec = np.sqrt(np.sum((z_pred_10 - z_ref_10) ** 2, axis=1))
    E_disc_400_rms = float(np.sqrt(np.mean(disc_err_vec ** 2)))
    E_disc_400_max = float(np.max(disc_err_vec))

    # 2. Analytical BCH Global Discretization Bound E_disc^BCH(h_k, dt):
    # Using trajectory-level bounds P_max = max_p_tr, F_tot,max = alpha_eff * max_dq_tr + L_V_bound + c_f*sqrt(d)
    F_max = c_f * np.sqrt(float(d))
    F_tot_max = alpha_eff * max_dq_tr + L_V_bound + F_max
    c_loc = (
        (gamma ** 2 / 24.0) * np.sqrt(max_p_tr**2 + F_tot_max**2)
        + (gamma / 12.0) * np.sqrt((F_tot_max + max_p_tr)**2 + (L_F * max_p_tr)**2)
        + (1.0 / 12.0) * np.sqrt((L_F * max_p_tr)**2 + (L_F * F_tot_max + 0.5 * M_F2 * (max_p_tr**2))**2)
    )
    L_f_vec = float(np.sqrt(max(1.0, L_F**2) + gamma**2))
    h_mean = float(np.mean(h_int_vec))
    dt_mean = h_mean / float(n_substeps)
    E_disc_BCH = float(((np.exp(L_f_vec * h_mean) - 1.0) / L_f_vec) * c_loc * (dt_mean ** 2))

    # Probe Lipschitz constants:
    # (a) Submultiplicative operator product: sigma_y * ||W_pr1,z||_2 * ||w_pr2||_2
    L_g_spec = float(stats_dict['tgt_std'] * np.linalg.norm(W_pr1[:out_dim, :], ord=2) * np.linalg.norm(W_pr2[:, 0], ord=2))
    # (b) Exact composite Jacobian supremum over hypercube d_sech in [0, 1]^H:
    # nabla_z g(z, u) = sigma_y * W_pr1,z^T diag(w_pr2) d_sech, where d_sech = 0.5*1 + 0.5*s, s in [-1, 1]^H
    A_pr = stats_dict['tgt_std'] * (W_pr1[:out_dim, :] * W_pr2[:, 0].reshape(1, H))
    L_g_cube = float(0.5 * np.linalg.norm(np.sum(A_pr, axis=1), ord=2) + 0.5 * np.sqrt(float(H)) * np.linalg.norm(A_pr, ord=2))
    # Also block-specific spectral norms for q and p:
    L_g_q_spec = float(stats_dict['tgt_std'] * np.linalg.norm(W_pr1[:d, :], ord=2) * np.linalg.norm(W_pr2[:, 0], ord=2))
    L_g_p_spec = float(stats_dict['tgt_std'] * np.linalg.norm(W_pr1[d:out_dim, :], ord=2) * np.linalg.norm(W_pr2[:, 0], ord=2))

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

    q_10, p_10 = z_10[:, :d], z_10[:, d:]
    q_tgt_tk = z_tgt_at_tk[:, :d]
    q_tgt_10, p_tgt_10 = z_tgt_10[:, :d], z_tgt_10[:, d:]
    v_sec_10 = (q_tgt_10 - q_10) / np.maximum(1.0, 4.0 * dt_eff_10)

    dq_init_vec = q_10 - q_tgt_tk
    dp_init_vec = p_10 - v_sec_10
    e_q_vec = np.sqrt(np.sum(dq_init_vec ** 2, axis=1))
    e_v_vec = np.sqrt(np.sum(dp_init_vec ** 2, axis=1))

    e_q_rms = float(np.sqrt(np.mean(e_q_vec ** 2)))
    e_q_max = float(np.max(e_q_vec))
    e_v_rms = float(np.sqrt(np.mean(e_v_vec ** 2)))
    e_v_max = float(np.max(e_v_vec))

    p_mod_vec = np.sqrt(np.sum(p_10 ** 2, axis=1))
    p_tgt_vec = np.sqrt(np.sum(v_sec_10 ** 2, axis=1))
    p_mod_max = float(np.max(p_mod_vec))
    z0_tgt_eucl_max = float(np.max(np.sqrt(e_q_vec ** 2 + p_tgt_vec ** 2)))

    # Issue 1: Continuum-certified within-window residual supremum across K=100 grid subdivisions + Taylor remainder L_{r,w} * (h_w / 200)
    a_sec_10 = (p_tgt_10 - p_10) / np.maximum(1.0, 4.0 * dt_eff_10)
    v_sec_norm = np.sqrt(np.sum(v_sec_10 ** 2, axis=1))
    a_sec_norm = np.sqrt(np.sum(a_sec_10 ** 2, axis=1))
    L_rw_vec = gamma * a_sec_norm + L_F * (v_sec_norm + h_int_vec[:, 0] * a_sec_norm)
    taylor_pad_vec = L_rw_vec * (h_int_vec[:, 0] / 200.0)

    K_grid = 100
    max_grid_r = np.zeros(len(q_10), dtype=np.float64)
    for tau_frac in np.linspace(0.0, 1.0, K_grid + 1):
        s_t = tau_frac * h_int_vec
        q_star_tau = q_tgt_tk + s_t * v_sec_10 + 0.5 * (s_t ** 2) * a_sec_10
        v_star_tau = v_sec_10 + s_t * a_sec_10
        f_star_tau = force_fn_10(q_star_tau)
        r_tau = np.sqrt(np.sum((a_sec_10 + gamma * v_star_tau - f_star_tau) ** 2, axis=1))
        max_grid_r = np.maximum(max_grid_r, r_tau)

    cert_sup_r_per_window = max_grid_r + taylor_pad_vec
    eps_mod_rms = float(np.sqrt(np.mean(cert_sup_r_per_window ** 2)))
    eps_mod_sup = float(np.max(cert_sup_r_per_window))

    return {
        'sigma_Wpot': sigma_Wpot,
        'sigma_Wpot_neg': sigma_Wpot_neg,
        'L_V_bound': L_V_bound,
        'L_nl_full': L_nl_full,
        'L_nl_neg': L_nl_neg,
        'L_F': L_F,
        'M_F2': M_F2,
        'c_loc': float(c_loc),
        'E_disc_BCH': E_disc_BCH,
        'E_disc_400_rms': E_disc_400_rms,
        'E_disc_400_max': E_disc_400_max,
        'L_g_spec': L_g_spec,
        'L_g_cube': L_g_cube,
        'L_g_q_spec': L_g_q_spec,
        'L_g_p_spec': L_g_p_spec,
        'L_g_dir': L_g_dir,
        'e_q_rms': e_q_rms,
        'e_q_max': e_q_max,
        'e_v_rms': e_v_rms,
        'e_v_max': e_v_max,
        'p_mod_max': p_mod_max,
        'z0_tgt_eucl_max': z0_tgt_eucl_max,
        'eps_mod_rms': eps_mod_rms,
        'eps_mod_sup': eps_mod_sup,
        'eps_probe_rms': eps_probe_rms,
        'eps_probe_max': eps_probe_max,
        'emp_rmse_10': emp_rmse_10,
        'mean_h_int': h_mean,
    }

for s in [42, 100]:
    tr_d = build_seq_data(df_tr, stats1, drop_prob=0.25, pattern='random', seed=s)
    te_in = build_seq_data(df_te, stats1, drop_prob=0.25, pattern='random', seed=s+2000)
    te_10x = build_seq_data(df_te, stats1, fixed_stride_range=(10, 20), seed=s+4000)
    r = train_and_audit_v9(tr_d, te_in, te_10x, stats1, seed=s, c_alpha=3.5, h_max=0.35, use_kin=True)
    # Also compute Duhamel-Volterra bound for Seed s:
    # On s in [0, h_k] (h_k=0.35), exact linear damped oscillator has omega_d = sqrt(0.35 - 0.015^2/4) = 0.5868
    # |dq(h_k)| <= cosh(sqrt(L_nl_full)*h_k) * (e_q + h_k*e_v + 0.5*h_k^2*eps_mod)
    # |e(h_k)|_2 <= e^{L_nl_full * h_k} * (sqrt(e_q^2 + e_v^2) + h_k * eps_mod)
    h_k = 0.35
    e_duhamel_rms = np.exp(r['L_nl_full'] * h_k) * (np.sqrt(r['e_q_rms']**2 + r['e_v_rms']**2) + h_k * r['eps_mod_rms']) + r['E_disc_BCH']
    rul_duhamel_spec = r['eps_probe_rms'] + r['L_g_spec'] * e_duhamel_rms
    rul_duhamel_cube = r['eps_probe_rms'] + r['L_g_cube'] * e_duhamel_rms
    print(f"Seed {s}: eps_mod_sup={r['eps_mod_sup']:.4f}, E_disc_BCH={r['E_disc_BCH']:.5f}, E_disc_400_rms={r['E_disc_400_rms']:.5f}, L_g_spec={r['L_g_spec']:.2f}, L_g_cube={r['L_g_cube']:.2f}, e_duhamel_rms={e_duhamel_rms:.4f}, RUL_Duhamel_Spec={rul_duhamel_spec:.2f}, RUL_Duhamel_Cube={rul_duhamel_cube:.2f}")

def evaluate_all_v9_bounds(rk, rnk, alpha_eff, gamma, eps, h_k, d=8):
    lam_min = 0.5 * ((alpha_eff + 1.0) - np.sqrt((alpha_eff - 1.0)**2 + 4.0 * eps**2))
    lam_max = 0.5 * ((alpha_eff + 1.0) + np.sqrt((alpha_eff - 1.0)**2 + 4.0 * eps**2))
    sqrt_lmin = float(np.sqrt(lam_min))
    inv_sqrt_lmin = float(1.0 / sqrt_lmin)
    kappa_P = float(np.sqrt(lam_max / lam_min))

    c_q = 0.5 * eps * alpha_eff
    c_p = 0.5 * (gamma - eps) - (eps * gamma**2) / alpha_eff
    mu_uub = float(2.0 * min(c_q, c_p) / lam_max)
    F_max = 0.45 * np.sqrt(float(d))
    dist_coeff = (eps / alpha_eff) + 1.0 / (2.0 * (gamma - eps))
    C_F_mod = dist_coeff * ((rk['L_V_bound'] + F_max) ** 2)
    C_F_tgt_sup = dist_coeff * ((rk['L_V_bound'] + F_max + rk['eps_mod_sup']) ** 2)
    R_inf = float(np.sqrt(2.0 * C_F_mod / (mu_uub * lam_min)))
    R_inf_star_sup = float(np.sqrt(2.0 * C_F_tgt_sup / (mu_uub * lam_min)))
    R_UUB_sup = R_inf + R_inf_star_sup

    z0_mod_max_scaled = float(rk['p_mod_max'] * inv_sqrt_lmin)
    z0_tgt_max_scaled = float(kappa_P * rk['z0_tgt_eucl_max'])
    uub_init_verified = bool((z0_mod_max_scaled <= R_inf) and (z0_tgt_max_scaled <= R_inf_star_sup))

    # Initial P-norm bounds (both RMS diagnostic and seed-level worst-case supremum max_w!)
    eP0_rms_kin = float(np.sqrt(alpha_eff * (rk['e_q_rms']**2) + 2.0 * eps * rk['e_q_rms'] * rk['e_v_rms'] + (rk['e_v_rms']**2)))
    eP0_rms_nokin = float(np.sqrt(alpha_eff * (rk['e_q_rms']**2) + 2.0 * eps * rk['e_q_rms'] * rnk['e_v_rms'] + (rnk['e_v_rms']**2)))

    eP0_sup_kin = float(np.sqrt(alpha_eff * (rk['e_q_max']**2) + 2.0 * eps * rk['e_q_max'] * rk['e_v_max'] + (rk['e_v_max']**2)))
    eP0_sup_nokin = float(np.sqrt(alpha_eff * (rk['e_q_max']**2) + 2.0 * eps * rk['e_q_max'] * rnk['e_v_max'] + (rnk['e_v_max']**2)))

    K_Young = float(((eps * gamma + rk['L_nl_full']) ** 2) / (4.0 * (gamma - eps)) - eps * (alpha_eff - rk['L_nl_neg']))
    nu_Young = float(K_Young / (alpha_eff - eps**2))
    growth = float(np.exp(min(50.0, nu_Young * h_k)))
    int_fac = float((growth - 1.0) / nu_Young)

    # Use analytical BCH discretization bound E_disc_BCH in formal bounds!
    lat_rms_kin = float(inv_sqrt_lmin * (growth * eP0_rms_kin + int_fac * rk['eps_mod_rms']) + rk['E_disc_BCH'])
    lat_rms_nokin = float(inv_sqrt_lmin * (growth * eP0_rms_nokin + int_fac * rk['eps_mod_rms']) + rnk['E_disc_BCH'])

    lat_sup_kin = float(inv_sqrt_lmin * (growth * eP0_sup_kin + int_fac * rk['eps_mod_sup']) + rk['E_disc_BCH'])
    lat_sup_nokin = float(inv_sqrt_lmin * (growth * eP0_sup_nokin + int_fac * rk['eps_mod_sup']) + rnk['E_disc_BCH'])

    lat_rms_eff_kin = min(R_UUB_sup, lat_rms_kin)
    lat_rms_eff_nokin = min(R_UUB_sup, lat_rms_nokin)
    lat_sup_eff_kin = min(R_UUB_sup, lat_sup_kin)
    lat_sup_eff_nokin = min(R_UUB_sup, lat_sup_nokin)

    # Also compute Duhamel-Volterra variation-of-constants tightened bound:
    lat_duhamel_rms_kin = min(R_UUB_sup, float(np.exp(min(50.0, rk['L_nl_full'] * h_k)) * (np.sqrt(rk['e_q_rms']**2 + rk['e_v_rms']**2) + h_k * rk['eps_mod_rms']) + rk['E_disc_BCH']))
    lat_duhamel_sup_kin = min(R_UUB_sup, float(np.exp(min(50.0, rk['L_nl_full'] * h_k)) * (np.sqrt(rk['e_q_max']**2 + rk['e_v_max']**2) + h_k * rk['eps_mod_sup']) + rk['E_disc_BCH']))

    # Formal Spectral Bounds (using L_{g,spec})
    spec_rms_kin = float(rk['eps_probe_rms'] + rk['L_g_spec'] * lat_rms_eff_kin)
    spec_rms_nokin = float(rk['eps_probe_rms'] + rk['L_g_spec'] * lat_rms_eff_nokin)
    spec_sup_kin = float(rk['eps_probe_max'] + rk['L_g_spec'] * lat_sup_eff_kin)
    spec_sup_nokin = float(rk['eps_probe_max'] + rk['L_g_spec'] * lat_sup_eff_nokin)

    duhamel_spec_rms_kin = float(rk['eps_probe_rms'] + rk['L_g_spec'] * lat_duhamel_rms_kin)
    duhamel_spec_sup_kin = float(rk['eps_probe_max'] + rk['L_g_spec'] * lat_duhamel_sup_kin)

    # Empirical Directional Diagnostics (using hat{L}_{g,dir})
    dir_rms_kin = float(rk['eps_probe_rms'] + rk['L_g_dir'] * lat_rms_eff_kin)
    dir_rms_nokin = float(rk['eps_probe_rms'] + rk['L_g_dir'] * lat_rms_eff_nokin)

    return {
        'K_Young': K_Young,
        'nu_Young': nu_Young,
        'R_inf': R_inf,
        'R_inf_star_sup': R_inf_star_sup,
        'R_UUB_sup': R_UUB_sup,
        'z0_mod_max_scaled': z0_mod_max_scaled,
        'z0_tgt_max_scaled': z0_tgt_max_scaled,
        'uub_init_verified': uub_init_verified,
        'eP0_rms_kin': eP0_rms_kin,
        'eP0_rms_nokin': eP0_rms_nokin,
        'eP0_sup_kin': eP0_sup_kin,
        'eP0_sup_nokin': eP0_sup_nokin,
        'lat_rms_kin': lat_rms_kin,
        'lat_rms_nokin': lat_rms_nokin,
        'lat_sup_kin': lat_sup_kin,
        'lat_sup_nokin': lat_sup_nokin,
        'lat_duhamel_rms_kin': lat_duhamel_rms_kin,
        'lat_duhamel_sup_kin': lat_duhamel_sup_kin,
        'Spec_RMS_Kin': spec_rms_kin,
        'Spec_RMS_NoKin': spec_rms_nokin,
        'Spec_Sup_Kin': spec_sup_kin,
        'Spec_Sup_NoKin': spec_sup_nokin,
        'Duhamel_Spec_RMS_Kin': duhamel_spec_rms_kin,
        'Duhamel_Spec_Sup_Kin': duhamel_spec_sup_kin,
        'Dir_Diag_Kin': dir_rms_kin,
        'Dir_Diag_NoKin': dir_rms_nokin,
    }

rows_A = []
rows_B = []
for s in SEEDS:
    tr_d = build_seq_data(df_tr, stats1, drop_prob=0.25, pattern='random', seed=s)
    te_in = build_seq_data(df_te, stats1, drop_prob=0.25, pattern='random', seed=s+2000)
    te_10x = build_seq_data(df_te, stats1, fixed_stride_range=(10, 20), seed=s+4000)

    rk_a = train_and_audit_v9(tr_d, te_in, te_10x, stats1, seed=s, c_alpha=3.5, h_max=0.35, use_kin=True)
    rnk_a = train_and_audit_v9(tr_d, te_in, te_10x, stats1, seed=s, c_alpha=3.5, h_max=0.35, use_kin=False)
    ba = evaluate_all_v9_bounds(rk_a, rnk_a, alpha_eff=0.35, gamma=0.15, eps=0.08, h_k=0.35)

    rows_A.append({
        'Setting': 'Default_Capped_a035_h035',
        'Seed': s,
        'sigma_Wpot': round(rk_a['sigma_Wpot'], 4),
        'L_nl_full': round(rk_a['L_nl_full'], 4),
        'L_nl_neg': round(rk_a['L_nl_neg'], 4),
        'K_Young': round(ba['K_Young'], 4),
        'nu_Young': round(ba['nu_Young'], 4),
        'z0_mod_max': round(ba['z0_mod_max_scaled'], 2),
        'R_inf': round(ba['R_inf'], 2),
        'z0_tgt_max': round(ba['z0_tgt_max_scaled'], 2),
        'R_inf_star': round(ba['R_inf_star_sup'], 2),
        'R_UUB_sup': round(ba['R_UUB_sup'], 2),
        'eP0_rms_Kin': round(ba['eP0_rms_kin'], 4),
        'eP0_rms_NoKin': round(ba['eP0_rms_nokin'], 4),
        'eP0_sup_Kin': round(ba['eP0_sup_kin'], 4),
        'eP0_sup_NoKin': round(ba['eP0_sup_nokin'], 4),
        'eps_mod_rms': round(rk_a['eps_mod_rms'], 4),
        'eps_mod_sup': round(rk_a['eps_mod_sup'], 4),
        'c_loc': round(rk_a['c_loc'], 4),
        'E_disc_BCH': round(rk_a['E_disc_BCH'], 5),
        'E_disc_400_rms': round(rk_a['E_disc_400_rms'], 5),
        'E_disc_400_max': round(rk_a['E_disc_400_max'], 5),
        'lat_rms_Kin': round(ba['lat_rms_kin'], 4),
        'lat_rms_NoKin': round(ba['lat_rms_nokin'], 4),
        'lat_sup_Kin': round(ba['lat_sup_kin'], 4),
        'lat_sup_NoKin': round(ba['lat_sup_nokin'], 4),
        'lat_duhamel_rms': round(ba['lat_duhamel_rms_kin'], 4),
        'L_g_spec': round(rk_a['L_g_spec'], 2),
        'L_g_dir': round(rk_a['L_g_dir'], 2),
        'Spec_RMS_Kin': round(ba['Spec_RMS_Kin'], 2),
        'Spec_RMS_NoKin': round(ba['Spec_RMS_NoKin'], 2),
        'Spec_Sup_Kin': round(ba['Spec_Sup_Kin'], 2),
        'Spec_Sup_NoKin': round(ba['Spec_Sup_NoKin'], 2),
        'Duhamel_Spec_RMS': round(ba['Duhamel_Spec_RMS_Kin'], 2),
        'Duhamel_Spec_Sup': round(ba['Duhamel_Spec_Sup_Kin'], 2),
        'Dir_Diag_Kin': round(ba['Dir_Diag_Kin'], 2),
        'Dir_Diag_NoKin': round(ba['Dir_Diag_NoKin'], 2),
        'Emp_OOD_RMSE': round(rk_a['emp_rmse_10'], 2),
    })

    rk_b = train_and_audit_v9(tr_d, te_in, te_10x, stats1, seed=s, c_alpha=1.0, h_max=100.0, use_kin=True)
    rnk_b = train_and_audit_v9(tr_d, te_in, te_10x, stats1, seed=s, c_alpha=1.0, h_max=100.0, use_kin=False)
    bb = evaluate_all_v9_bounds(rk_b, rnk_b, alpha_eff=0.10, gamma=0.15, eps=0.05, h_k=rk_b['mean_h_int'])

    rows_B.append({
        'Setting': 'Uncapped_FullHorizon_a010_hInf',
        'Seed': s,
        'sigma_Wpot': round(rk_b['sigma_Wpot'], 4),
        'L_nl_full': round(rk_b['L_nl_full'], 4),
        'L_nl_neg': round(rk_b['L_nl_neg'], 4),
        'K_Young': round(bb['K_Young'], 4),
        'nu_Young': round(bb['nu_Young'], 4),
        'z0_mod_max': round(bb['z0_mod_max_scaled'], 2),
        'R_inf': round(bb['R_inf'], 2),
        'z0_tgt_max': round(bb['z0_tgt_max_scaled'], 2),
        'R_inf_star': round(bb['R_inf_star_sup'], 2),
        'R_UUB_sup': round(bb['R_UUB_sup'], 2),
        'eP0_rms_Kin': round(bb['eP0_rms_kin'], 4),
        'eP0_rms_NoKin': round(bb['eP0_rms_nokin'], 4),
        'eP0_sup_Kin': round(bb['eP0_sup_kin'], 4),
        'eP0_sup_NoKin': round(bb['eP0_sup_nokin'], 4),
        'eps_mod_rms': round(rk_b['eps_mod_rms'], 4),
        'eps_mod_sup': round(rk_b['eps_mod_sup'], 4),
        'c_loc': round(rk_b['c_loc'], 4),
        'E_disc_BCH': round(rk_b['E_disc_BCH'], 4),
        'E_disc_400_rms': round(rk_b['E_disc_400_rms'], 5),
        'E_disc_400_max': round(rk_b['E_disc_400_max'], 5),
        'lat_rms_Kin': round(bb['lat_rms_kin'], 1),
        'lat_rms_NoKin': round(bb['lat_rms_nokin'], 1),
        'lat_sup_Kin': round(bb['lat_sup_kin'], 1),
        'lat_sup_NoKin': round(bb['lat_sup_nokin'], 1),
        'lat_duhamel_rms': round(bb['lat_duhamel_rms_kin'], 2),
        'L_g_spec': round(rk_b['L_g_spec'], 2),
        'L_g_dir': round(rk_b['L_g_dir'], 2),
        'Spec_RMS_Kin': round(bb['Spec_RMS_Kin'], 1),
        'Spec_RMS_NoKin': round(bb['Spec_RMS_NoKin'], 1),
        'Spec_Sup_Kin': round(bb['Spec_Sup_Kin'], 1),
        'Spec_Sup_NoKin': round(bb['Spec_Sup_NoKin'], 1),
        'Duhamel_Spec_RMS': round(bb['Duhamel_Spec_RMS_Kin'], 1),
        'Duhamel_Spec_Sup': round(bb['Duhamel_Spec_Sup_Kin'], 1),
        'Dir_Diag_Kin': round(bb['Dir_Diag_Kin'], 1),
        'Dir_Diag_NoKin': round(bb['Dir_Diag_NoKin'], 1),
        'Emp_OOD_RMSE': round(rk_b['emp_rmse_10'], 2),
    })

dfA = pd.DataFrame(rows_A)
dfB = pd.DataFrame(rows_B)
pd.concat([dfA, dfB], ignore_index=True).to_csv('results/ress_revision/Theorem3_Empirical_Constants.csv', index=False)

print("=== PANEL A FULL SUMMARY ===")
print(dfA.drop(columns=['Setting']).to_string(index=False))
print("\nPANEL A MEAN / SD / MAX:")
print(dfA.drop(columns=['Setting']).agg(['mean', 'std', 'max']).round(4).to_string())

print("\n=== PANEL B FULL SUMMARY ===")
print(dfB.drop(columns=['Setting']).to_string(index=False))
print("\nPANEL B MEAN / SD / MAX:")
print(dfB.drop(columns=['Setting']).agg(['mean', 'std', 'max']).round(4).to_string())
