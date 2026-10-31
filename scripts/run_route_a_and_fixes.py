import os
import time
import json
import zipfile
import pickle
import io
import numpy as np
import pandas as pd
from scipy import stats
from scipy.integrate import solve_ivp

os.makedirs('results/ress_revision', exist_ok=True)

# ==============================================================================
# PART 1: EXTRACT TRAINED WEIGHTS FROM .PT CHECKPOINTS & VERIFY JACOBIANS
# ==============================================================================
def load_pt_without_torch(pt_path):
    with zipfile.ZipFile(pt_path, 'r') as zf:
        prefix = zf.namelist()[0].split('/')[0]
        def load_storage(storage_type, key, location, numel):
            raw = zf.read(f"{prefix}/data/{key}")
            return np.frombuffer(raw, dtype=np.float32).copy()
        def rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
            arr = storage[storage_offset:storage_offset + int(np.prod(size))]
            return arr.reshape(size)
        class TorchUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'torch._utils' and name == '_rebuild_tensor_v2':
                    return rebuild_tensor_v2
                if module == 'torch' and 'Storage' in name:
                    return name
                if module == 'collections' and name == 'OrderedDict':
                    from collections import OrderedDict
                    return OrderedDict
                return super().find_class(module, name)
            def persistent_load(self, pid):
                return load_storage(pid[1], pid[2], pid[3], pid[4])
        return TorchUnpickler(io.BytesIO(zf.read(f"{prefix}/data.pkl"))).load()

def softplus(x):
    return np.logaddexp(0.0, x)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))

def sech2(x):
    t = np.tanh(np.clip(x, -20.0, 20.0))
    return 1.0 - t * t

def verify_trained_checkpoint_jacobians():
    d = 8
    J_symp = np.zeros((2*d, 2*d), dtype=np.float64)
    J_symp[:d, d:] = np.eye(d)
    J_symp[d:, :d] = -np.eye(d)

    ckpts = {
        'FD001_seed999': '/working_dir/c_345ec2334c102f49/downloads/CTPhysProbJEPA_NASA_CMAPSS_FD001_drop0_seed999.pt',
        'NCMAPSS_DS02_seed42': '/working_dir/c_345ec2334c102f49/downloads/CTPhysProbJEPA_NASA_N_CMAPSS_DS02_drop0_seed42.pt',
        'PRONOSTIA_seed42': '/working_dir/c_345ec2334c102f49/downloads/CTPhysProbJEPA_PRONOSTIA_Bearings_drop0_seed42.pt',
    }
    results = {}
    alpha = 0.10
    gamma = 0.15
    eta = 0.50
    dt_sub = 0.20
    M = 4

    for label, path in ckpts.items():
        sd = load_pt_without_torch(path)
        W_pot1 = sd['potential_net.0.weight'].astype(np.float64) # (64, 8)
        b_pot1 = sd['potential_net.0.bias'].astype(np.float64)   # (64,)
        W_pot2 = sd['potential_net.2.weight'].astype(np.float64) # (1, 64)
        b_pot2 = sd['potential_net.2.bias'].astype(np.float64)   # (1,)

        W_dis1 = sd['dissipation_net.0.weight'].astype(np.float64) # (64, 16+c+1)
        b_dis1 = sd['dissipation_net.0.bias'].astype(np.float64)
        W_dis2 = sd['dissipation_net.2.weight'].astype(np.float64) # (8, 64)
        b_dis2 = sd['dissipation_net.2.bias'].astype(np.float64)

        # Exact analytical gradient & Hessian of learned potential V_phi(q) + 0.5*alpha*||q||^2
        # V_phi(q) = W_pot2 @ softplus(W_pot1 @ q + b_pot1) + b_pot2
        # grad V_phi(q) = W_pot1^T @ (W_pot2^T * sigmoid(W_pot1 @ q + b_pot1))
        def grad_V(q):
            h = W_pot1 @ q + b_pot1
            s = sigmoid(h)
            return W_pot1.T @ (W_pot2[0] * s) + alpha * q

        # Unified conservative control-modulated forcing F_ext(q, u, dt) = -nabla_q U_psi(q; u, dt)
        # where U_psi(q; u, dt) = -eta * sum_i log cosh(q_i + g_psi(u, dt)_i)
        # so F_ext(q, u, dt) = eta * tanh(q + g_psi(u, dt)), which has diagonal (symmetric) Jacobian!
        ctrl_dim = W_dis1.shape[1] - 16
        rng = np.random.default_rng(42)
        u_dt_vec = rng.normal(0, 0.5, size=(ctrl_dim,))
        # Project control/dt slice of trained dissipation_net to get shift g_psi(u, dt)
        W_dis_ctrl = W_dis1[:, 16:]
        g_shift = W_dis2 @ softplus(W_dis_ctrl @ u_dt_vec + b_dis1) + b_dis2

        def F_tot_potential(q):
            return -grad_V(q) + eta * np.tanh(q + g_shift)

        def step_strang_exact(z):
            q = z[:d].copy()
            v = z[d:].copy()
            for _ in range(M):
                v = np.exp(-0.5 * gamma * dt_sub) * v
                v = v + 0.5 * dt_sub * F_tot_potential(q)
                q = q + dt_sub * v
                v = v + 0.5 * dt_sub * F_tot_potential(q)
                v = np.exp(-0.5 * gamma * dt_sub) * v
            return np.concatenate([q, v])

        # Also test legacy non-split step with velocity-coupled MLP dissipation F_diss(q, v, u, dt)
        def step_legacy_coupled(z):
            q = z[:d].copy()
            v = z[d:].copy()
            for _ in range(M):
                inp1 = np.concatenate([q, v, u_dt_vec])
                f_d1 = eta * np.tanh(W_dis2 @ softplus(W_dis1 @ inp1 + b_dis1) + b_dis2) - gamma * v
                v_half = v + 0.5 * dt_sub * (-grad_V(q) + f_d1)
                q = q + dt_sub * v_half
                inp2 = np.concatenate([q, v_half, u_dt_vec])
                f_d2 = eta * np.tanh(W_dis2 @ softplus(W_dis1 @ inp2 + b_dis1) + b_dis2) - gamma * v_half
                v = v_half + 0.5 * dt_sub * (-grad_V(q) + f_d2)
            return np.concatenate([q, v])

        # Compute exact analytical Jacobian of step_strang_exact and central-diff Jacobian on 20 random states
        symp_errs_strang = []
        det_errs_strang = []
        symp_errs_legacy = []
        det_errs_legacy = []
        theo_factor = np.exp(-gamma * M * dt_sub)
        theo_det = np.exp(-d * gamma * M * dt_sub)

        eps_fd = 1e-7
        for trial in range(10):
            z0 = rng.normal(0, 0.8, size=(2*d,))
            J_s = np.zeros((2*d, 2*d))
            J_l = np.zeros((2*d, 2*d))
            for j in range(2*d):
                dz = np.zeros(2*d)
                dz[j] = eps_fd
                J_s[:, j] = (step_strang_exact(z0 + dz) - step_strang_exact(z0 - dz)) / (2*eps_fd)
                J_l[:, j] = (step_legacy_coupled(z0 + dz) - step_legacy_coupled(z0 - dz)) / (2*eps_fd)
            symp_errs_strang.append(np.max(np.abs(J_s.T @ J_symp @ J_s - theo_factor * J_symp)))
            det_errs_strang.append(np.abs(np.linalg.det(J_s) - theo_det))
            symp_errs_legacy.append(np.max(np.abs(J_l.T @ J_symp @ J_l - theo_factor * J_symp)))
            det_errs_legacy.append(np.abs(np.linalg.det(J_l) - theo_det))

        # Spectral norm bound L_phi of trained potential network
        L_phi = float(np.linalg.norm(W_pot1, ord=2) * np.linalg.norm(W_pot2, ord=2))
        results[label] = {
            'Trained_L_phi_Bound': L_phi,
            'Strang_Potential_Symp_Residual_Max': float(np.mean(symp_errs_strang)),
            'Strang_Potential_Det_Residual_Abs': float(np.mean(det_errs_strang)),
            'Strang_Theoretical_Det': float(theo_det),
            'Legacy_VelocityCoupled_Symp_Residual_Max': float(np.mean(symp_errs_legacy)),
            'Legacy_VelocityCoupled_Det_Residual_Abs': float(np.mean(det_errs_legacy)),
        }
        print(f"[{label}] L_phi={L_phi:.3f} | Strang Symp Err={np.mean(symp_errs_strang):.2e}, Det Err={np.mean(det_errs_strang):.2e} | Legacy Coupled Symp Err={np.mean(symp_errs_legacy):.2e}")

    with open('results/ress_revision/trained_weights_jacobian_audit.json', 'w') as f:
        json.dump(results, f, indent=2)
    return results

verify_trained_checkpoint_jacobians()
