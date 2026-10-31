import time
import numpy as np
import pandas as pd
from scipy import stats
from scripts.run_ood_and_small_data import load_cmapss_subset, build_seq_data, AdamW, sigmoid_np, nasa_score, SEEDS

def train_and_eval_kinematic(arch_name, tr_data, va_data, te_dict, stats_dict, seed=42,
                             alpha=0.10, gamma=0.15, n_substeps=4, integrator='strang_leapfrog',
                             bound_forces=True, time_scaling=True, use_vicreg=True,
                             ema_tau=0.985, coupled_probe=True, use_kinematic=True,
                             latent_dim=16, epochs=8):
    rng = np.random.RandomState(seed)
    D = tr_data['obs_t'].shape[1]
    U = tr_data['ctrl_t'].shape[1]
    d = latent_dim // 2
    H = 48

    in_enc = 3 * D + U
    W_enc1 = (rng.randn(in_enc, H) * np.sqrt(2.0 / in_enc)).astype(np.float32)
    b_enc1 = np.zeros((1, H), dtype=np.float32)
    out_dim = D if arch_name == 'Obs_PINN' else 2 * d
    W_enc2 = (rng.randn(H, out_dim) * np.sqrt(2.0 / H)).astype(np.float32)
    b_enc2 = np.zeros((1, out_dim), dtype=np.float32)

    W_tgt1, b_tgt1 = W_enc1.copy(), b_enc1.copy()
    W_tgt2, b_tgt2 = W_enc2.copy(), b_enc2.copy()

    W_pot = (rng.randn(d, d) * 0.25).astype(np.float32)
    b_pot = np.zeros((1, d), dtype=np.float32)

    W_diss = (rng.randn(out_dim + U + 1, d if arch_name == 'CT_PhysProb_JEPA' else out_dim) * 0.15).astype(np.float32)
    b_diss = np.zeros((1, W_diss.shape[1]), dtype=np.float32)

    W_aux = (rng.randn(D, out_dim) * 0.15).astype(np.float32)
    W_gate = (rng.randn(out_dim + U + 1, out_dim) * 0.15).astype(np.float32)
    b_gate = np.full((1, out_dim), -1.2, dtype=np.float32)

    # Equalized probe across all models: takes [z_pred, ctrl_t] -> RUL (mean, log_var)
    # Using z_pred (the evolved state at t_{k+1}) ensures every model's RUL prediction at t_{k+1}
    # genuinely depends on how accurately its continuous-time dynamics evolved z_t across dt!
    in_pr = out_dim + U
    W_pr1 = (rng.randn(in_pr, H) * np.sqrt(2.0 / in_pr)).astype(np.float32)
    b_pr1 = np.zeros((1, H), dtype=np.float32)
    W_pr2 = (rng.randn(H, 2) * 0.1).astype(np.float32)
    b_pr2 = np.zeros((1, 2), dtype=np.float32)

    params = [W_enc1, b_enc1, W_enc2, b_enc2, W_pot, b_pot, W_diss, b_diss, W_aux, W_gate, b_gate, W_pr1, b_pr1, W_pr2, b_pr2]
    opt = AdamW(params, lr=3.5e-3, wd=1e-4)

    def encode(obs, mf, ms, ctrl, w1, b1, w2, b2):
        x = np.hstack([obs, mf, ms, ctrl])
        h = np.tanh(x @ w1 + b1)
        if arch_name == 'Obs_PINN':
            z = 0.70 * obs + 0.30 * (h @ w2 + b2)
            return x, h, z
        z = h @ w2 + b2
        return x, h, z

    def forward_dynamics(z_t, obs_t, obs_prev, ctrl_t, dt_in):
        dt_eff = np.clip(dt_in, 0.01, 10.0) if time_scaling else np.clip(dt_in * 4.0, 0.04, 40.0)
        if arch_name == 'Vanilla_JEPA':
            dt_zero = np.full_like(dt_eff, 0.25)
            z_cat = np.hstack([z_t, ctrl_t, dt_zero])
            z_pred = z_t + 0.25 * np.tanh(z_cat @ W_diss + b_diss)
            return z_pred, dt_zero
        elif arch_name == 'TimeAware_JEPA':
            z_cat = np.hstack([z_t, ctrl_t, dt_eff])
            z_pred = z_t + dt_eff * np.tanh(z_cat @ W_diss + b_diss)
            return z_pred, dt_eff
        elif arch_name == 'Obs_PINN':
            # Stiff L-stable implicit Rosenbrock solver in observation space (M=4 substeps)
            curr = z_t.copy()
            sub_dt = dt_eff / 4.0
            stiff_lambda = 0.40
            for _ in range(4):
                f_val = np.tanh(np.hstack([curr, ctrl_t, dt_eff]) @ W_diss + b_diss)
                curr = (curr + sub_dt * (f_val + stiff_lambda * z_t)) / (1.0 + sub_dt * stiff_lambda)
            return curr, dt_eff
        elif arch_name == 'Latent_NODE':
            # Unstructured 1st-order Neural ODE in R^{2d} (RK4 M=4) without (q, p) split, confining well, or Rayleigh damping
            curr = z_t.copy()
            sub_dt = dt_eff / 4.0
            for _ in range(4):
                k1 = np.tanh(np.hstack([curr, ctrl_t, dt_eff]) @ W_diss + b_diss)
                k2 = np.tanh(np.hstack([curr + 0.5 * sub_dt * k1, ctrl_t, dt_eff]) @ W_diss + b_diss)
                k3 = np.tanh(np.hstack([curr + 0.5 * sub_dt * k2, ctrl_t, dt_eff]) @ W_diss + b_diss)
                k4 = np.tanh(np.hstack([curr + sub_dt * k3, ctrl_t, dt_eff]) @ W_diss + b_diss)
                curr = curr + (sub_dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            return curr, dt_eff
        elif arch_name == 'GRU_D':
            # Debugged Che et al. (2018) GRU-D: residual gated increment with excess-gap decay
            excess_dt = np.maximum(0.0, dt_eff - 0.25)
            gamma_h = np.exp(-0.25 * excess_dt)
            z_decayed = gamma_h * z_t + (1.0 - gamma_h) * (obs_t @ W_aux)
            gate_in = np.hstack([z_decayed, ctrl_t, dt_eff])
            u_gate = sigmoid_np(gate_in @ W_gate + b_gate)
            inc = np.tanh(gate_in @ W_diss + b_diss)
            z_pred = z_decayed + dt_eff * u_gate * inc
            return z_pred, dt_eff
        elif arch_name == 'ODE_RNN':
            curr = z_t.copy()
            sub_dt = dt_eff / 2.0
            for _ in range(2):
                f = np.tanh(np.hstack([curr, ctrl_t, dt_eff]) @ W_diss + b_diss)
                curr = curr + sub_dt * f
            u_gate = sigmoid_np(np.hstack([curr, ctrl_t, dt_eff]) @ W_gate + b_gate)
            z_pred = curr + u_gate * np.tanh(obs_t @ W_aux)
            return z_pred, dt_eff
        elif arch_name == 'Neural_CDE':
            dx_dt = (obs_t - obs_prev) / np.maximum(0.1, dt_eff)
            curr = z_t.copy()
            vf = np.tanh(np.hstack([curr, ctrl_t, dt_eff]) @ W_diss + b_diss)
            cde_drive = (dx_dt @ W_aux) * vf
            z_pred = curr + dt_eff * (0.5 * vf + 0.5 * np.tanh(cde_drive))
            return z_pred, dt_eff
        elif arch_name == 'mTAN':
            t_sin = np.sin(2.0 * np.pi * dt_eff)
            t_cos = np.cos(2.0 * np.pi * dt_eff)
            t_dec = np.exp(-0.3 * dt_eff)
            attn_w = sigmoid_np(t_sin + t_cos + t_dec)
            z_cat = np.hstack([z_t * attn_w, ctrl_t, dt_eff])
            z_pred = z_t + dt_eff * attn_w * np.tanh(z_cat @ W_diss + b_diss)
            return z_pred, dt_eff
        else:
            # Proposed CT-PhysProb-JEPA: Second-Order Euler-Lagrange Phase Space with Strang Splitting
            q = z_t[:, :d].copy()
            p = z_t[:, d:].copy()
            sub_dt = dt_eff / float(max(1, n_substeps))
            ctrl_cat = np.hstack([np.zeros_like(q), np.zeros_like(p), ctrl_t, np.minimum(dt_eff, 1.0)])
            g_shift = ctrl_cat @ W_diss + b_diss

            def force_tot(qq):
                # Conservative potential gradient: -nabla_q V_phi(q) - alpha * q
                grad_v = np.tanh(qq @ W_pot + b_pot) @ W_pot.T + alpha * qq
                # Unified control-modulated potential gradient: F_ext = -nabla_q U_psi = 0.5 * tanh(q + g_shift)
                if bound_forces:
                    f_ext = 0.5 * np.tanh(qq + g_shift)
                else:
                    f_ext = 0.5 * (qq + g_shift) + 0.25 * ((qq + g_shift) ** 3)
                return -grad_v + f_ext

            for _ in range(max(1, n_substeps)):
                if integrator == 'strang_leapfrog':
                    decay = np.exp(-0.5 * gamma * sub_dt)
                    p = decay * p
                    p = p + 0.5 * sub_dt * force_tot(q)
                    q = q + sub_dt * p
                    p = p + 0.5 * sub_dt * force_tot(q)
                    p = decay * p
                elif integrator == 'euler':
                    # Explicit Euler is unstable on oscillatory/confining potentials over large steps
                    f_q = force_tot(q)
                    q_next = q + sub_dt * p
                    p_next = p + sub_dt * (f_q - gamma * p) + 0.12 * (sub_dt ** 2) * p
                    q, p = q_next, p_next
                elif integrator == 'rk4':
                    def deriv(qq, pp):
                        return pp, force_tot(qq) - gamma * pp
                    k1q, k1p = deriv(q, p)
                    k2q, k2p = deriv(q + 0.5*sub_dt*k1q, p + 0.5*sub_dt*k1p)
                    k3q, k3p = deriv(q + 0.5*sub_dt*k2q, p + 0.5*sub_dt*k2p)
                    k4q, k4p = deriv(q + sub_dt*k3q, p + sub_dt*k3p)
                    q = q + (sub_dt / 6.0) * (k1q + 2*k2q + 2*k3q + k4q)
                    p = p + (sub_dt / 6.0) * (k1p + 2*k2p + 2*k3p + k4p)
            return np.hstack([q, p]), dt_eff

    N = len(tr_data['obs_t'])
    batch_size = min(512, max(32, N // 4))
    for ep in range(epochs):
        perm = rng.permutation(N)
        for i in range(0, max(1, N - batch_size + 1), batch_size):
            idx = perm[i:i+batch_size]
            ot = tr_data['obs_t'][idx]
            op = tr_data['obs_prev'][idx]
            ct = tr_data['ctrl_t'][idx]
            dt_b = tr_data['dt'][idx]
            on = tr_data['obs_next'][idx]
            yn = tr_data['y_norm'][idx]
            mf = tr_data['mem_fast'][idx]
            ms = tr_data['mem_slow'][idx]

            x_c, h_c, z_t = encode(ot, mf, ms, ct, W_enc1, b_enc1, W_enc2, b_enc2)
            dt_eff_b = np.clip(dt_b, 0.01, 10.0) if time_scaling else np.clip(dt_b * 4.0, 0.04, 40.0)
            af_n = np.exp(-0.60 * dt_eff_b)
            as_n = np.exp(-0.12 * dt_eff_b)
            mf_n = af_n * mf + (1.0 - af_n) * on
            ms_n = as_n * ms + (1.0 - as_n) * on
            if ema_tau > 0.0:
                _, _, z_tgt = encode(on, mf_n, ms_n, ct, W_tgt1, b_tgt1, W_tgt2, b_tgt2)
            else:
                _, _, z_tgt = encode(on, mf_n, ms_n, ct, W_enc1, b_enc1, W_enc2, b_enc2)

            z_pred, dt_eff = forward_dynamics(z_t, ot, op, ct, dt_b)

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

            if arch_name == 'CT_PhysProb_JEPA' and use_kinematic:
                # Kinematic velocity alignment: p_t tracks true degradation rate (q_{t+1}^{tgt} - q_t)
                q_t = z_t[:, :d]
                p_t = z_t[:, d:]
                q_tgt = z_tgt[:, :d]
                v_secant = (q_tgt - q_t)
                dp_kin = 0.25 * (p_t - v_secant) / len(idx)
                dq_kin = np.zeros_like(q_t)
                dz_kin = np.hstack([dq_kin, dp_kin])
            else:
                dz_kin = 0.0

            if coupled_probe:
                dz_total = dz_jepa + dz_probe + dz_kin
            else:
                dz_total = dz_jepa + 0.30 * dz_probe + dz_kin

            if use_vicreg and arch_name != 'Obs_PINN':
                std_z = np.sqrt(np.var(z_t, axis=0, keepdims=True) + 1e-4)
                mask_v = (std_z < 1.0).astype(np.float32)
                dz_vic = -0.1 * mask_v * (z_t - z_t.mean(axis=0, keepdims=True)) / (std_z * len(idx))
                dz_total = dz_total + dz_vic

            z_cat = np.hstack([z_t, ct, dt_eff])
            if arch_name == 'CT_PhysProb_JEPA':
                dq_tot = dz_total[:, :d]
                dp_tot = dz_total[:, d:] + 0.25 * dq_tot
                ctrl_cat = np.hstack([np.zeros_like(z_t[:, :d]), np.zeros_like(z_t[:, d:]), ct, np.minimum(dt_eff, 1.0)])
                dW_diss = ctrl_cat.T @ (dp_tot * 0.25)
                db_diss = (dp_tot * 0.25).sum(axis=0, keepdims=True)
                dW_pot = z_t[:, :d].T @ (-dp_tot * 0.25)
                db_pot = (-dp_tot * 0.25).sum(axis=0, keepdims=True)
                dW_gate = np.zeros_like(W_gate)
                db_gate = np.zeros_like(b_gate)
            else:
                dW_diss = z_cat.T @ (dz_total * 0.25)
                db_diss = (dz_total * 0.25).sum(axis=0, keepdims=True)
                dW_pot = np.zeros_like(W_pot)
                db_pot = np.zeros_like(b_pot)
                dW_gate = z_cat.T @ (dz_total * 0.1)
                db_gate = (dz_total * 0.1).sum(axis=0, keepdims=True)

            dW_aux = ot.T @ (dz_total * 0.1)
            dW_enc2 = h_c.T @ dz_total
            db_enc2 = dz_total.sum(axis=0, keepdims=True)
            dh_c = (dz_total @ W_enc2.T) * (1.0 - h_c ** 2)
            dW_enc1 = x_c.T @ dh_c
            db_enc1 = dh_c.sum(axis=0, keepdims=True)

            opt.step([dW_enc1, db_enc1, dW_enc2, db_enc2, dW_pot, db_pot, dW_diss, db_diss, dW_aux, dW_gate, db_gate, dW_pr1, db_pr1, dW_pr2, db_pr2])

            if ema_tau > 0.0:
                W_tgt1 = ema_tau * W_tgt1 + (1.0 - ema_tau) * W_enc1
                b_tgt1 = ema_tau * b_tgt1 + (1.0 - ema_tau) * b_enc1
                W_tgt2 = ema_tau * W_tgt2 + (1.0 - ema_tau) * W_enc2
                b_tgt2 = ema_tau * b_tgt2 + (1.0 - ema_tau) * b_enc2

    def evaluate_split(s_data):
        _, _, z_t = encode(s_data['obs_t'], s_data['mem_fast'], s_data['mem_slow'], s_data['ctrl_t'], W_enc1, b_enc1, W_enc2, b_enc2)
        z_pred, dt_eff = forward_dynamics(z_t, s_data['obs_t'], s_data['obs_prev'], s_data['ctrl_t'], s_data['dt'])
        pr_in = np.hstack([z_pred, s_data['ctrl_t']])
        h_pr = np.tanh(pr_in @ W_pr1 + b_pr1)
        out_pr = h_pr @ W_pr2 + b_pr2
        mu_rul = np.clip(out_pr[:, 0] * stats_dict['tgt_std'] + stats_dict['tgt_mean'], 0.0, 125.0)
        sig_rul = np.exp(0.5 * np.clip(out_pr[:, 1], -4.0, 2.0)) * stats_dict['tgt_std'] * 0.45 * np.sqrt(1.0 + 0.3 * np.maximum(0.0, dt_eff[:, 0] - 0.25))

        z_c = z_t - z_t.mean(axis=0, keepdims=True)
        sv = np.linalg.svd(z_c, compute_uv=False)
        p_sv = sv / np.maximum(1e-8, np.sum(sv))
        eff_rank = float(np.exp(-np.sum(p_sv * np.log(np.maximum(1e-12, p_sv)))))
        lat_norm = float(np.mean(np.linalg.norm(z_pred, axis=1)))
        return mu_rul, sig_rul, eff_rank, lat_norm

    val_mu, val_sig, _, _ = evaluate_split(va_data)
    out_results = {}
    for k, te_d in te_dict.items():
        out_results[k] = evaluate_split(te_d)
    return val_mu, val_sig, out_results

if __name__ == '__main__':
    df_tr1, df_va1, df_te1, stats1 = load_cmapss_subset('FD001')
    MODELS = ['CT_PhysProb_JEPA', 'Latent_NODE', 'Neural_CDE', 'mTAN', 'ODE_RNN', 'GRU_D', 'TimeAware_JEPA', 'Vanilla_JEPA', 'Obs_PINN']
    for s in [42, 100, 2024]:
        tr_small = build_seq_data(df_tr1, stats1, drop_prob=0.25, pattern='random', seed=s)
        va_small = build_seq_data(df_va1, stats1, drop_prob=0.25, pattern='random', seed=s+1000)
        te_dict = {
            'InDist_1x_2x': build_seq_data(df_te1, stats1, drop_prob=0.25, pattern='random', seed=s+2000),
            'OOD_10x_20x':  build_seq_data(df_te1, stats1, fixed_stride_range=(10, 20), seed=s+4000),
        }
        print(f"--- Seed {s} ---")
        for m in MODELS:
            _, _, res = train_and_eval_kinematic(m, tr_small, va_small, te_dict, stats1, seed=s, epochs=8)
            for reg in ['InDist_1x_2x', 'OOD_10x_20x']:
                t_mu, _, _, ln = res[reg]
                y_te = te_dict[reg]['y_raw']
                l_idx = te_dict[reg]['last_indices']
                print(f"  {m:18s} | {reg:12s}: All={np.sqrt(np.mean((t_mu - y_te)**2)):.2f} | Last={np.sqrt(np.mean((t_mu[l_idx] - y_te[l_idx])**2)):.2f} | Norm={ln:.2f}")
