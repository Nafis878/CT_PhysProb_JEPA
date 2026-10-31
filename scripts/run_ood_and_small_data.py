import os
import time
import json
import numpy as np
import pandas as pd
from scipy import stats

os.makedirs('results/ress_revision', exist_ok=True)
SEEDS = [42, 100, 2024, 777, 999, 101, 202, 303, 404, 505]
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

    if subset in ('FD002', 'FD004'):
        def get_cond_key(df_in):
            return (np.round(df_in['os1'].values, 0).astype(int) * 1000 +
                    np.round(df_in['os3'].values, 0).astype(int))
        tr_keys = get_cond_key(df_train)
        cond_means, cond_stds = {}, {}
        g_mean = df_train[OBS_COLS].values.astype(np.float32).mean(axis=0)
        g_std  = df_train[OBS_COLS].values.astype(np.float32).std(axis=0) + 1e-5
        for k in np.unique(tr_keys):
            m_k = (tr_keys == k)
            cond_means[int(k)] = df_train.loc[m_k, OBS_COLS].values.astype(np.float32).mean(axis=0)
            cond_stds[int(k)]  = df_train.loc[m_k, OBS_COLS].values.astype(np.float32).std(axis=0) + 1e-5
        for df_part in (df_train, df_val, df_test):
            p_keys = get_cond_key(df_part)
            arr = df_part[OBS_COLS].values.astype(np.float32).copy()
            for k in np.unique(p_keys):
                m_k = (p_keys == k)
                cm = cond_means.get(int(k), g_mean)
                cs = cond_stds.get(int(k), g_std)
                arr[m_k] = (arr[m_k] - cm) / cs
            df_part[OBS_COLS] = arr
        t_stats = {
            'obs_mean': np.zeros((1, len(OBS_COLS)), dtype=np.float32),
            'obs_std':  np.ones((1, len(OBS_COLS)), dtype=np.float32),
            'ctrl_mean': df_train[CTRL_COLS].values.astype(np.float32).mean(axis=0, keepdims=True),
            'ctrl_std':  df_train[CTRL_COLS].values.astype(np.float32).std(axis=0, keepdims=True) + 1e-5,
            'tau_0': float(np.median(df_train['dt'].values)) + 1e-5,
            'tgt_mean': float(df_train['RUL'].values.mean()),
            'tgt_std':  float(df_train['RUL'].values.std()) + 1e-5,
            'rul_cap': 125.0
        }
    else:
        t_stats = {
            'obs_mean': df_train[OBS_COLS].values.astype(np.float32).mean(axis=0, keepdims=True),
            'obs_std':  df_train[OBS_COLS].values.astype(np.float32).std(axis=0, keepdims=True) + 1e-5,
            'ctrl_mean': df_train[CTRL_COLS].values.astype(np.float32).mean(axis=0, keepdims=True),
            'ctrl_std':  df_train[CTRL_COLS].values.astype(np.float32).std(axis=0, keepdims=True) + 1e-5,
            'tau_0': float(np.median(df_train['dt'].values)) + 1e-5,
            'tgt_mean': float(df_train['RUL'].values.mean()),
            'tgt_std':  float(df_train['RUL'].values.std()) + 1e-5,
            'rul_cap': 125.0
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
            g_min, g_max = fixed_stride_range
            keep_idx = [0]
            idx_c = 0
            while idx_c < n_rows - 1:
                step_g = rng.randint(g_min, g_max + 1)
                idx_c = min(n_rows - 1, idx_c + step_g)
                keep_idx.append(idx_c)
            keep_idx = sorted(list(set(keep_idx)))
            if len(keep_idx) < 3:
                keep_idx = [0, max(1, n_rows // 2), n_rows - 1]
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
            dt_k = raw_dt[k-1] / max(1.0, stats_dict['tau_0'])
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
    def __init__(self, params, lr=3.5e-3, wd=1e-4):
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
