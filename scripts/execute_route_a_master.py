import os
import time
import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.integrate import solve_ivp

os.makedirs('results/ress_revision', exist_ok=True)
SEEDS = [42, 100, 2024, 777, 999, 101, 202, 303, 404, 505]

# Copy downloaded FD002, FD003, FD004 into data/raw/CMAPSS/
os.makedirs('data/raw/CMAPSS', exist_ok=True)
for fn in ['train_FD002.txt', 'test_FD002.txt', 'RUL_FD002.txt',
           'train_FD003.txt', 'test_FD003.txt', 'RUL_FD003.txt',
           'train_FD004.txt', 'test_FD004.txt', 'RUL_FD004.txt']:
    src = f'/working_dir/c_345ec2334c102f49/downloads/{fn}'
    dst = f'data/raw/CMAPSS/{fn}'
    if os.path.exists(src) and not os.path.exists(dst):
        import shutil
        shutil.copyfile(src, dst)

OBS_COLS = [f's{i}' for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]]
CTRL_COLS = ['os1', 'os2', 'os3']

def load_cmapss_subset(subset='FD001'):
    cols = ['unit', 'cycle', 'os1', 'os2', 'os3'] + [f's{i}' for i in range(1, 22)]
    df_tr_raw = pd.read_csv(f'data/raw/CMAPSS/train_{subset}.txt', sep=r'\s+', header=None, names=cols)
    df_te_raw = pd.read_csv(f'data/raw/CMAPSS/test_{subset}.txt', sep=r'\s+', header=None, names=cols)
    rul_te_raw = pd.read_csv(f'data/raw/CMAPSS/RUL_{subset}.txt', sep=r'\s+', header=None, names=['rul_end'])
    rul_te_raw['unit'] = np.arange(1, len(rul_te_raw) + 1)

    max_c_tr = df_tr_raw.groupby('unit')['cycle'].transform('max')
    df_tr_raw['RUL'] = (max_c_tr - df_tr_raw['cycle']).clip(upper=125).astype(np.float32)
    df_tr_raw['dt'] = 1.0

    df_te_raw = df_te_raw.merge(rul_te_raw, on='unit')
    max_c_te = df_te_raw.groupby('unit')['cycle'].transform('max')
    df_te_raw['RUL'] = (max_c_te - df_te_raw['cycle'] + df_te_raw['rul_end']).clip(upper=125).astype(np.float32)
    df_te_raw['dt'] = 1.0

    with open(f'data/splits/cmapss_{subset}.json') as f:
        split_info = json.load(f)

    df_train = df_tr_raw[df_tr_raw['unit'].isin(split_info['train'])].reset_index(drop=True)
    df_val   = df_tr_raw[df_tr_raw['unit'].isin(split_info['val'])].reset_index(drop=True)
    df_test  = df_te_raw.reset_index(drop=True)

    # For multi-operating-condition subsets (FD002, FD004), normalize sensors conditional on operating regime cluster or jointly with op settings
    t_stats = {
        'obs_mean': df_train[OBS_COLS].values.astype(np.float32).mean(axis=0, keepdims=True),
        'obs_std':  df_train[OBS_COLS].values.astype(np.float32).std(axis=0, keepdims=True) + 1e-5,
        'ctrl_mean': df_train[CTRL_COLS].values.astype(np.float32).mean(axis=0, keepdims=True),
        'ctrl_std':  df_train[CTRL_COLS].values.astype(np.float32).std(axis=0, keepdims=True) + 1e-5,
        'tau_0': float(np.median(df_train['dt'].values)) + 1e-5,
        'tgt_mean': float(df_train['RUL'].values.mean()),
        'tgt_std':  float(df_train['RUL'].values.std()) + 1e-5,
    }
    return df_train, df_val, df_test, t_stats

def nasa_score(pred, true):
    d = np.asarray(pred, dtype=np.float64) - np.asarray(true, dtype=np.float64)
    d = np.clip(d, -100.0, 60.0)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))

def build_seq_data(df, stats_dict, drop_prob=0.0, pattern='random', seed=42, fixed_stride_range=None):
    rng = np.random.RandomState(seed)
    obs_t_l, ctrl_t_l, dt_l, obs_next_l, y_norm_l, y_raw_l = [], [], [], [], [], []
    mem_fast_l, mem_slow_l, obs_prev_l, unit_id_l = [], [], [], []
    last_indices = []
    cur = 0

    for uid, group in df.groupby('unit'):
        group = group.reset_index(drop=True)
        n_rows = len(group)
        cum_time = group['dt'].cumsum().values.astype(np.float32)
        obs_raw = group[OBS_COLS].values.astype(np.float32)

        if fixed_stride_range is not None:
            # Deterministic/stochastic OOD long-gap subsampling with gaps in [min_gap, max_gap]
            g_min, g_max = fixed_stride_range
            keep_idx = [0]
            idx_c = 0
            while idx_c < n_rows - 1:
                step_g = rng.randint(g_min, g_max + 1)
                idx_c = min(n_rows - 1, idx_c + step_g)
                keep_idx.append(idx_c)
            keep_idx = sorted(list(set(keep_idx)))
            if len(keep_idx) < 3:
                keep_idx = [0, n_rows // 2, n_rows - 1]
            group = group.iloc[keep_idx].reset_index(drop=True)
            cum_time = cum_time[keep_idx]
            obs_raw = obs_raw[keep_idx]
        elif drop_prob > 0.0:
            if pattern == 'random':
                keep = rng.rand(n_rows) > drop_prob
            elif pattern == 'burst':
                keep = np.ones(n_rows, dtype=bool)
                n_drop_total = int(n_rows * drop_prob)
                n_bursts = 2
                b_len = max(2, n_drop_total // n_bursts)
                for b in range(n_bursts):
                    low = max(1, int((b / n_bursts) * (n_rows - b_len - 1)))
                    high = max(low + 1, int(((b + 1) / n_bursts) * (n_rows - b_len - 1)))
                    s_idx = rng.randint(low, high)
                    keep[s_idx : s_idx + b_len] = False
            elif pattern == 'mnar':
                obs_z = np.mean(np.abs((obs_raw - stats_dict['obs_mean']) / stats_dict['obs_std']), axis=1)
                logits = (obs_z - np.median(obs_z)) * 1.5
                probs = 1.0 / (1.0 + np.exp(-logits))
                probs = np.clip(probs * (drop_prob / max(1e-4, np.mean(probs))), 0.05, 0.95)
                keep = rng.rand(n_rows) > probs
            else:
                keep = np.ones(n_rows, dtype=bool)
            keep[0] = True
            keep[-1] = True
            if keep.sum() < 3:
                keep[:3] = True
            group = group[keep].reset_index(drop=True)
            cum_time = cum_time[keep]
            obs_raw = obs_raw[keep]

        obs = (obs_raw - stats_dict['obs_mean']) / stats_dict['obs_std']
        ctrl = (group[CTRL_COLS].values.astype(np.float32) - stats_dict['ctrl_mean']) / stats_dict['ctrl_std']
        y_r = group['RUL'].values.astype(np.float32)
        y_n = (y_r - stats_dict['tgt_mean']) / stats_dict['tgt_std']

        m = len(group) - 1
        raw_dt = np.maximum(1e-4, cum_time[1:] - cum_time[:-1])
        dt_tilde = (raw_dt / (4.0 * stats_dict['tau_0'])).astype(np.float32)

        mem_f = np.zeros_like(obs)
        mem_s = np.zeros_like(obs)
        obs_p = np.zeros_like(obs)
        mem_f[0] = obs[0]
        mem_s[0] = obs[0]
        obs_p[0] = obs[0]
        for k in range(1, len(obs)):
            dt_k = raw_dt[k-1]
            af = np.exp(-0.15 * dt_k)
            aslow = np.exp(-0.03 * dt_k)
            mem_f[k] = af * mem_f[k-1] + (1.0 - af) * obs[k]
            mem_s[k] = aslow * mem_s[k-1] + (1.0 - aslow) * obs[k]
            obs_p[k] = obs[k-1]

        obs_t_l.append(obs[:-1])
        ctrl_t_l.append(ctrl[:-1])
        dt_l.append(dt_tilde[:, None])
        obs_next_l.append(obs[1:])
        y_norm_l.append(y_n[1:, None])
        y_raw_l.append(y_r[1:])
        mem_fast_l.append(mem_f[:-1])
        mem_slow_l.append(mem_s[:-1])
        obs_prev_l.append(obs_p[:-1])
        unit_id_l.append(np.full(m, uid, dtype=int))
        cur += m
        last_indices.append(cur - 1)

    return {
        'obs_t': np.vstack(obs_t_l),
        'ctrl_t': np.vstack(ctrl_t_l),
        'dt': np.vstack(dt_l),
        'obs_next': np.vstack(obs_next_l),
        'y_norm': np.vstack(y_norm_l),
        'y_raw': np.concatenate(y_raw_l),
        'mem_fast': np.vstack(mem_fast_l),
        'mem_slow': np.vstack(mem_slow_l),
        'obs_prev': np.vstack(obs_prev_l),
        'unit_id': np.concatenate(unit_id_l),
        'last_indices': np.array(last_indices, dtype=int)
    }

class AdamW:
    def __init__(self, params, lr=3e-3, wd=1e-4):
        self.params = params
        self.lr = lr
        self.wd = wd
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        b1, b2 = 0.9, 0.999
        for i, (p, g) in enumerate(zip(self.params, grads)):
            g = np.clip(g, -2.0, 2.0)
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * g
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * (g * g)
            m_hat = self.m[i] / (1.0 - b1 ** self.t)
            v_hat = self.v[i] / (1.0 - b2 ** self.t)
            p -= self.lr * (m_hat / (np.sqrt(v_hat) + 1e-8) + self.wd * p)

def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))

def run_model_architecture(arch_name, tr_data, va_data, te_data, stats_dict, seed=42,
                           alpha=0.10, gamma=0.15, n_substeps=4, integrator='strang_leapfrog',
                           bound_forces=True, time_scaling=True, use_vicreg=True,
                           ema_tau=0.985, coupled_probe=True, latent_dim=16, epochs=10):
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

    W_pot = (rng.randn(d, d) * 0.2).astype(np.float32)
    b_pot = np.zeros((1, d), dtype=np.float32)

    # For CT_PhysProb_JEPA: unified control-modulated potential shift g_psi(u_k, dt_eff) in R^d
    W_diss = (rng.randn(out_dim + U + 1, d if arch_name == 'CT_PhysProb_JEPA' else out_dim) * 0.15).astype(np.float32)
    b_diss = np.zeros((1, W_diss.shape[1]), dtype=np.float32)

    W_aux = (rng.randn(D, out_dim) * 0.15).astype(np.float32)
    # Gate weights for debugged GRU-D
    W_gate = (rng.randn(out_dim + U + 1, out_dim) * 0.15).astype(np.float32)
    b_gate = np.full((1, out_dim), -1.0, dtype=np.float32)

    in_pr = 2 * out_dim + U + 1
    W_pr1 = (rng.randn(in_pr, H) * np.sqrt(2.0 / in_pr)).astype(np.float32)
    b_pr1 = np.zeros((1, H), dtype=np.float32)
    W_pr2 = (rng.randn(H, 2) * 0.1).astype(np.float32)
    b_pr2 = np.zeros((1, 2), dtype=np.float32)

    params = [W_enc1, b_enc1, W_enc2, b_enc2, W_pot, b_pot, W_diss, b_diss, W_aux, W_gate, b_gate, W_pr1, b_pr1, W_pr2, b_pr2]
    opt = AdamW(params, lr=3e-3, wd=1e-4)

    def encode(obs, mf, ms, ctrl, w1, b1, w2, b2):
        x = np.hstack([obs, mf, ms, ctrl])
        h = np.tanh(x @ w1 + b1)
        if arch_name == 'Obs_PINN':
            # Stiff-capable observation-space projection (denoised smoothed observation state)
            z = 0.65 * obs + 0.35 * (h @ w2 + b2)
            return x, h, z
        z = h @ w2 + b2
        return x, h, z

    def forward_dynamics(z_t, obs_t, obs_prev, ctrl_t, dt_in):
        # Allow true long gaps up to dt_eff = 10.0 (equivalent to 40 cycles) so OOD gap extrapolation is genuinely tested!
        dt_eff = np.clip(dt_in, 0.01, 10.0) if time_scaling else np.clip(dt_in * 4.0, 0.04, 40.0)
        if arch_name == 'Vanilla_JEPA':
            dt_zero = np.full_like(dt_eff, 0.25)
            z_cat = np.hstack([z_t, ctrl_t, dt_zero])
            z_pred = z_t + 0.25 * np.tanh(z_cat @ W_diss + b_diss)
            return z_pred, dt_zero
        elif arch_name == 'TimeAware_JEPA':
            z_cat = np.hstack([z_t, ctrl_t, dt_eff])
            # Direct single-step MLP jump scaled by dt_eff (drifts when dt_eff >> training range)
            z_pred = z_t + dt_eff * np.tanh(z_cat @ W_diss + b_diss)
            return z_pred, dt_eff
        elif arch_name == 'Obs_PINN':
            # Stiff L-stable Rosenbrock / Implicit-Euler substep solver in observation space (M=4 substeps)
            curr = z_t.copy()
            sub_dt = dt_eff / 4.0
            stiff_lambda = 0.35
            for _ in range(4):
                f_val = np.tanh(np.hstack([curr, ctrl_t, dt_eff]) @ W_diss + b_diss)
                # L-stable implicit damping step: (I + sub_dt * stiff_lambda) curr_{m+1} = curr_m + sub_dt * (f_val + stiff_lambda * obs_t)
                curr = (curr + sub_dt * (f_val + stiff_lambda * obs_t)) / (1.0 + sub_dt * stiff_lambda)
            return curr, dt_eff
        elif arch_name == 'Latent_NODE':
            # First-order unstructured Neural ODE in R^{2d} (RK4 M=4) without Euler-Lagrange (q, p) split, confining well, or Rayleigh damping
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
            # Debugged Che et al. (2018) GRU-D:
            # Decay only over excess gap (dt_eff - 0.25)+ so at 0% missingness (dt_eff=0.25) decay=1.0!
            excess_dt = np.maximum(0.0, dt_eff - 0.25)
            gamma_h = np.exp(-0.35 * excess_dt)
            z_decayed = gamma_h * z_t + (1.0 - gamma_h) * np.tanh(obs_t @ W_aux)
            gate_in = np.hstack([z_decayed, ctrl_t, dt_eff])
            u_gate = sigmoid_np(gate_in @ W_gate + b_gate)
            cand = np.tanh(gate_in @ W_diss + b_diss)
            z_pred = (1.0 - u_gate) * z_decayed + u_gate * cand
            return z_pred, dt_eff
        elif arch_name == 'ODE_RNN':
            curr = z_t.copy()
            sub_dt = dt_eff / 2.0
            for _ in range(2):
                f = np.tanh(np.hstack([curr, ctrl_t, dt_eff]) @ W_diss + b_diss)
                curr = curr + sub_dt * f
            u_gate = sigmoid_np(np.hstack([curr, ctrl_t, dt_eff]) @ W_gate + b_gate)
            z_pred = (1.0 - u_gate) * curr + u_gate * np.tanh(obs_t @ W_aux)
            return z_pred, dt_eff
        elif arch_name == 'Neural_CDE':
            dx_dt = (obs_t - obs_prev) / np.maximum(0.1, dt_eff)
            curr = z_t.copy()
            vf = np.tanh(np.hstack([curr, ctrl_t, dt_eff]) @ W_diss + b_diss)
            cde_drive = (dx_dt @ W_aux) * vf
            z_pred = curr + dt_eff * (0.45 * vf + 0.45 * np.tanh(cde_drive))
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
            # Proposed CT-PhysProb-JEPA with Unified Potential-Gradient Forcing (Theorem 1 Exact Match)
            q = z_t[:, :d].copy()
            p = z_t[:, d:].copy()
            sub_dt = dt_eff / float(max(1, n_substeps))
            # Control-dependent shift g_psi(u_k, dt_eff) (depends only on u_k, dt_eff, NOT on p!)
            ctrl_cat = np.hstack([np.zeros_like(q), np.zeros_like(p), ctrl_t, dt_eff])
            g_shift = ctrl_cat @ W_diss + b_diss

            def force_tot(qq):
                # Conservative potential gradient: -nabla_q V_phi(q) - alpha * q
                grad_v = np.tanh(qq @ W_pot + b_pot) @ W_pot.T + alpha * qq
                # Control-modulated potential gradient: F_ext(q, u, dt) = -nabla_q U_psi(q; u, dt) = 0.5 * tanh(q + g_shift)
                if bound_forces:
                    f_ext = 0.5 * np.tanh(qq + g_shift)
                else:
                    f_ext = 0.5 * (qq + g_shift) + 0.15 * ((qq + g_shift) ** 3)
                return -grad_v + f_ext

            for _ in range(max(1, n_substeps)):
                if integrator == 'strang_leapfrog':
                    decay = np.exp(-0.5 * gamma * sub_dt)
                    # (1) Half-step exact Rayleigh damping
                    p = decay * p
                    # (2) Half-step conservative kick
                    p = p + 0.5 * sub_dt * force_tot(q)
                    # (3) Full-step position drift
                    q = q + sub_dt * p
                    # (4) Half-step conservative kick
                    p = p + 0.5 * sub_dt * force_tot(q)
                    # (5) Half-step exact Rayleigh damping
                    p = decay * p
                elif integrator == 'euler':
                    f_q = force_tot(q)
                    q_next = q + sub_dt * p
                    p_next = p + sub_dt * (f_q - gamma * p)
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
    batch_size = min(256, max(32, N // 4))
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
            mf_n = 0.85 * mf + 0.15 * on
            ms_n = 0.97 * ms + 0.03 * on
            if ema_tau > 0.0:
                _, _, z_tgt = encode(on, mf_n, ms_n, ct, W_tgt1, b_tgt1, W_tgt2, b_tgt2)
            else:
                _, _, z_tgt = encode(on, mf_n, ms_n, ct, W_enc1, b_enc1, W_enc2, b_enc2)

            z_pred, dt_eff = forward_dynamics(z_t, ot, op, ct, dt_b)

            pr_in = np.hstack([z_pred, z_t, ct, dt_eff])
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
            dz_pred = 0.5 * (z_pred - z_tgt) / len(idx)
            if coupled_probe:
                dz_total = dz_pred + dpr_in[:, :out_dim] + dpr_in[:, out_dim:2*out_dim]
            else:
                dz_total = dz_pred + 0.35 * dpr_in[:, out_dim:2*out_dim]

            if use_vicreg and arch_name != 'Obs_PINN':
                std_z = np.sqrt(np.var(z_t, axis=0, keepdims=True) + 1e-4)
                mask_v = (std_z < 1.0).astype(np.float32)
                dz_vic = -0.1 * mask_v * (z_t - z_t.mean(axis=0, keepdims=True)) / (std_z * len(idx))
                dz_total = dz_total + dz_vic

            z_cat = np.hstack([z_t, ct, dt_eff])
            if arch_name == 'CT_PhysProb_JEPA':
                dp_pred = dz_total[:, d:]
                ctrl_cat = np.hstack([np.zeros_like(z_t[:, :d]), np.zeros_like(z_t[:, d:]), ct, dt_eff])
                dW_diss = ctrl_cat.T @ (dp_pred * 0.2)
                db_diss = (dp_pred * 0.2).sum(axis=0, keepdims=True)
                dW_pot = z_t[:, :d].T @ (-dp_pred * 0.2)
                db_pot = (-dp_pred * 0.2).sum(axis=0, keepdims=True)
                dW_gate = np.zeros_like(W_gate)
                db_gate = np.zeros_like(b_gate)
            else:
                dW_diss = z_cat.T @ (dz_total * 0.2)
                db_diss = (dz_total * 0.2).sum(axis=0, keepdims=True)
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
        pr_in = np.hstack([z_pred, z_t, s_data['ctrl_t'], dt_eff])
        h_pr = np.tanh(pr_in @ W_pr1 + b_pr1)
        out_pr = h_pr @ W_pr2 + b_pr2
        mu_rul = np.clip(out_pr[:, 0] * stats_dict['tgt_std'] + stats_dict['tgt_mean'], 0.0, 125.0)
        # Uncertainty grows with both learned heteroscedastic log_var and elapsed gap length dt_eff
        sig_rul = np.exp(0.5 * np.clip(out_pr[:, 1], -4.0, 2.0)) * stats_dict['tgt_std'] * 0.45 * np.sqrt(1.0 + 0.3 * np.maximum(0.0, dt_eff[:, 0] - 0.25))

        z_c = z_t - z_t.mean(axis=0, keepdims=True)
        sv = np.linalg.svd(z_c, compute_uv=False)
        p_sv = sv / np.maximum(1e-8, np.sum(sv))
        eff_rank = float(np.exp(-np.sum(p_sv * np.log(np.maximum(1e-12, p_sv)))))
        lat_norm = float(np.mean(np.linalg.norm(z_pred, axis=1)))
        return mu_rul, sig_rul, eff_rank, lat_norm

    val_mu, val_sig, _, _ = evaluate_split(va_data)
    te_mu, te_sig, eff_rank, lat_norm = evaluate_split(te_data)
    return val_mu, val_sig, te_mu, te_sig, eff_rank, lat_norm

print("Testing Route A Experiment 1: OOD 10x Gap Extrapolation (Train on small gaps dt in {1,2}, Test on 10x gaps dt in [10,20])...")
df_tr1, df_va1, df_te1, stats1 = load_cmapss_subset('FD001')
df_tr2, df_va2, df_te2, stats2 = load_cmapss_subset('FD002')

MODELS = ['CT_PhysProb_JEPA', 'Latent_NODE', 'Neural_CDE', 'mTAN', 'ODE_RNN', 'GRU_D', 'TimeAware_JEPA', 'Vanilla_JEPA', 'Obs_PINN']

ood_rows = []
for ds_name, d_tr, d_va, d_te, st in [('FD001', df_tr1, df_va1, df_te1, stats1), ('FD002', df_tr2, df_va2, df_te2, stats2)]:
    for s in SEEDS:
        # Train on small gaps (20% random drop, mean dt = 1.25 cycles)
        tr_small = build_seq_data(d_tr, st, drop_prob=0.20, pattern='random', seed=s)
        va_small = build_seq_data(d_va, st, drop_prob=0.20, pattern='random', seed=s+1000)
        # Test on: (a) In-dist small gaps, (b) 5x OOD gaps [5,10], (c) 10x OOD gaps [10,20], (d) 15x OOD gaps [15,25]
        te_regimes = [
            ('InDist_1x_2x', build_seq_data(d_te, st, drop_prob=0.20, pattern='random', seed=s+2000)),
            ('OOD_5x_10x',   build_seq_data(d_te, st, fixed_stride_range=(5, 10), seed=s+3000)),
            ('OOD_10x_20x',  build_seq_data(d_te, st, fixed_stride_range=(10, 20), seed=s+4000)),
            ('OOD_15x_25x',  build_seq_data(d_te, st, fixed_stride_range=(15, 25), seed=s+5000)),
        ]
        for m_name in MODELS:
            for reg_name, te_d in te_regimes:
                _, _, t_mu, _, erank, lnorm = run_model_architecture(m_name, tr_small, va_small, te_d, st, seed=s, epochs=10)
                y_te = te_d['y_raw']
                last_idx = te_d['last_indices']
                rmse_all = float(np.sqrt(np.mean((t_mu - y_te)**2)))
                rmse_last = float(np.sqrt(np.mean((t_mu[last_idx] - y_te[last_idx])**2)))
                ood_rows.append({
                    'Dataset': ds_name, 'Seed': s, 'Test_Gap_Regime': reg_name, 'Model': m_name,
                    'RMSE_All': round(rmse_all, 4), 'RMSE_Last': round(rmse_last, 4),
                    'Eff_Rank': round(erank, 3), 'Latent_Norm': round(lnorm, 3)
                })

df_ood = pd.DataFrame(ood_rows)
df_ood.to_csv('results/ress_revision/10Seed_OOD_10xGap_Extrapolation.csv', index=False)
for (ds_name, reg_name), g in df_ood.groupby(['Dataset', 'Test_Gap_Regime']):
    print(f"=== {ds_name} | {reg_name} ===")
    ct_v = g[g['Model']=='CT_PhysProb_JEPA'].sort_values('Seed')['RMSE_All'].values
    for m_name in MODELS:
        mv = g[g['Model']==m_name].sort_values('Seed')
        va = mv['RMSE_All'].values
        vl = mv['RMSE_Last'].values
        ln = mv['Latent_Norm'].values
        pt = stats.ttest_rel(ct_v, va)[1] if m_name != 'CT_PhysProb_JEPA' else np.nan
        print(f"  {m_name:18s}: All={va.mean():.2f}±{va.std(ddof=1):.2f} | Last={vl.mean():.2f}±{vl.std(ddof=1):.2f} | Norm={ln.mean():.2f} | p_t={pt:.4g}")
