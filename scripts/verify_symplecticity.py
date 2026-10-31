"""Exact Numerical Verification of Conformal Symplecticity & Volume Contraction
for the Strang-Split Leapfrog Integrator vs. Original Unsplit Integrator.
"""
import numpy as np
import json

rng = np.random.RandomState(42)
d = 8  # q_dim = 8, phase space 2d = 16
H = 32
alpha = 0.10
gamma = 0.15
dt_tilde = 0.80
M = 4
sub_dt = dt_tilde / M

# Random potential network weights V_phi(q) = W2^T log(cosh(W1 q + b1))
W1 = rng.randn(H, d) * 0.25
b1 = rng.randn(H) * 0.1
W2 = rng.randn(H) * 0.25

def grad_V(q):
    # Exact gradient of V(q) = V_phi(q) + 0.5 * alpha * ||q||^2
    # V_phi(q) = sum(W2 * log(cosh(W1 q + b1))) -> grad = W1^T (W2 * tanh(W1 q + b1))
    u = W1 @ q + b1
    return W1.T @ (W2 * np.tanh(u)) + alpha * q

def step_conservative_leapfrog(z, delta_t):
    q, p = z[:d].copy(), z[d:].copy()
    p_mid = p - 0.5 * delta_t * grad_V(q)
    q_next = q + delta_t * p_mid
    p_next = p_mid - 0.5 * delta_t * grad_V(q_next)
    return np.concatenate([q_next, p_next])

def step_strang_conformal(z, delta_t):
    q, p = z[:d].copy(), z[d:].copy()
    # Half-step exact Rayleigh damping
    decay = np.exp(-0.5 * gamma * delta_t)
    p = decay * p
    # Conservative symplectic leapfrog kick-drift-kick
    z_cons = step_conservative_leapfrog(np.concatenate([q, p]), delta_t)
    q_next, p_next = z_cons[:d], z_cons[d:]
    # Half-step exact Rayleigh damping
    p_next = decay * p_next
    return np.concatenate([q_next, p_next])

def full_4substep_strang(z):
    curr = z.copy()
    for _ in range(M):
        curr = step_strang_conformal(curr, sub_dt)
    return curr

def compute_jacobian(fn, z, eps=1e-7):
    n = len(z)
    J = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        dz = np.zeros(n, dtype=np.float64)
        dz[i] = eps
        J[:, i] = (fn(z + dz) - fn(z - dz)) / (2.0 * eps)
    return J

J_mat = np.block([[np.zeros((d, d)), np.eye(d)], [-np.eye(d), np.zeros((d, d))]])

z0 = rng.randn(2 * d) * 0.5
J_cons = compute_jacobian(lambda z: step_conservative_leapfrog(z, sub_dt), z0)
J_strang_1 = compute_jacobian(lambda z: step_strang_conformal(z, sub_dt), z0)
J_strang_4 = compute_jacobian(full_4substep_strang, z0)

symp_err_cons = np.max(np.abs(J_cons.T @ J_mat @ J_cons - J_mat))
det_cons = np.linalg.det(J_cons)

target_scale_1 = np.exp(-gamma * sub_dt)
conf_err_1 = np.max(np.abs(J_strang_1.T @ J_mat @ J_strang_1 - target_scale_1 * J_mat))
det_strang_1 = np.linalg.det(J_strang_1)
det_theory_1 = np.exp(-d * gamma * sub_dt)

target_scale_4 = np.exp(-gamma * dt_tilde)
conf_err_4 = np.max(np.abs(J_strang_4.T @ J_mat @ J_strang_4 - target_scale_4 * J_mat))
det_strang_4 = np.linalg.det(J_strang_4)
det_theory_4 = np.exp(-d * gamma * dt_tilde)

res = {
    "d": d,
    "alpha": alpha,
    "gamma": gamma,
    "dt_tilde": dt_tilde,
    "M_substeps": M,
    "conservative_substep": {
        "symplectic_max_abs_err": float(symp_err_cons),
        "det_J_numerical": float(det_cons),
        "det_J_theoretical": 1.0,
        "abs_det_err": float(abs(det_cons - 1.0))
    },
    "strang_1_substep": {
        "conformal_symplectic_max_abs_err": float(conf_err_1),
        "det_J_numerical": float(det_strang_1),
        "det_J_theoretical": float(det_theory_1),
        "abs_det_err": float(abs(det_strang_1 - det_theory_1))
    },
    "strang_4_substeps": {
        "conformal_symplectic_max_abs_err": float(conf_err_4),
        "det_J_numerical": float(det_strang_4),
        "det_J_theoretical": float(det_theory_4),
        "abs_det_err": float(abs(det_strang_4 - det_theory_4))
    }
}

with open('results/ress_revision/symplecticity_verification.json', 'w') as f:
    json.dump(res, f, indent=2)

print(json.dumps(res, indent=2))
