#!/usr/bin/env python3
"""
Applies all 5 remaining mathematical fixes to manuscript/main.tex:
1. Issue 1 (Initial-Error Bound): Derives ||e(t_k)||_P <= sqrt(alpha_eff*e_q^2 + 2*eps*e_q*e_v + e_v^2)
   directly from Eq. (36) and divides both initial and integral terms by sqrt(lambda_min(P)).
   Recalculates all RUL bounds and the exact % reduction from L_kin (62.5% reduction on formal spectral bound,
   57.4% reduction on empirical directional diagnostic).
2. Issue 2 (Model Residual): Replaces point-wise secant average with the within-window continuous-time supremum
   eps_model,w^sup = sup_{t in [t_w, t_w+h_w]} ||r_model,w(t)||_2, reporting both the fleet-wide uniform supremum
   eps_model,sup = max_w eps_model,w^sup (1.0738 +/- 0.1585) and the Minkowski L2 RMS statistic eps_model,RMS (0.5860 +/- 0.0774).
3. Issue 3 (Probe Lipschitz Constant): Uses exclusively the formal operator spectral norm L_{g,spec} (115.30 +/- 6.30)
   inside Theorem 3 for formal mathematical guarantees (371.93 +/- 46.99 cycles RMS bound, 734.57 +/- 81.22 cycles supremum bound),
   and explicitly presents hat{L}_{g,dir} (14.27 +/- 1.02) and its 57.18 +/- 7.50-cycle value as an empirical plug-in diagnostic,
   NOT a formal mathematical guarantee.
4. Issue 4 (Numerical Integration Error): Separates continuous-time ODE tracking error from Strang-splitting
   discretization error E_disc(h_k, dt) = ||z_hat^(M) - z_hat^exact(t_k+h_k)||_2 via the triangle inequality,
   derives the O(dt^2) BCH bound, and reports exact measured E_disc against an M=400 reference solver across all 10 seeds.
5. Issue 5 (Two-Trajectory UUB): Reports worst-case scaled initial conditions z0_mod^max and z0_tgt^max alongside
   R_inf and R_inf^* across all 10 seeds in Table 8 (Panels A & B), proving z0_mod^max << R_inf and z0_tgt^max << R_inf^* on 10/10 seeds.
"""

with open('manuscript/main.tex', 'r') as f:
    tex = f.read()

# 1. Update Abstract
old_abs = """Using Young's inequality on the Lyapunov error derivative, we prove a finite-horizon Gr\\"{o}nwall--Lyapunov extrapolation bound capped by a two-trajectory Uniform Ultimate Boundedness (UUB) radius $R_\\infty + R_\\infty^*$, showing that secant velocity alignment reduces the bounded-horizon RUL error bound by $42.8\\%$ ($58.14 \\pm 10.65 \\to 33.25 \\pm 3.22$ cycles), whereas removing the horizon cap ($h_{\\max}=\\infty, \\alpha_{\\mathrm{eff}}=0.10$) causes exponential Gr\\"{o}nwall blow-up to the UUB ceiling and degrades end-to-end retrained RMSE ($32.63 \\pm 10.40$ cycles)."""

new_abs = """Using Young's inequality on the Lyapunov error derivative and separating continuous-time ODE tracking error from $O(\\delta t^2)$ Strang-splitting discretization error ($E_{\\mathrm{disc}} = 1.0\\times 10^{-4}$), we prove a finite-horizon Gr\\"{o}nwall--Lyapunov extrapolation bound capped by a verified two-trajectory Uniform Ultimate Boundedness (UUB) radius $R_\\infty + R_\\infty^*$. Secant velocity alignment cuts initial $P$-norm error by $67.0\\%$, reducing the formal spectral-norm RUL bound by $62.5\\%$ ($992.11 \\pm 194.93 \\to 371.93 \\pm 46.99$ cycles) and an empirical directional sensitivity diagnostic by $57.4\\%$ ($134.33 \\pm 28.70 \\to 57.18 \\pm 7.50$ cycles), whereas removing the horizon cap ($h_{\\max}=\\infty, \\alpha_{\\mathrm{eff}}=0.10$) causes exponential Gr\\"{o}nwall blow-up to the UUB ceiling and degrades end-to-end retrained RMSE ($32.63 \\pm 10.40$ cycles)."""
assert old_abs in tex, "Abstract target not found"
tex = tex.replace(old_abs, new_abs, 1)

# 2. Update Introduction Contribution 1
old_c1 = """In Appendix~\\ref{app:proof_thm3} (Table~\\ref{tab:thm3_per_seed}), complete 10-seed calculations show that in the bounded-horizon setting ($h_k=0.35, \\nu_{\\mathrm{Young}}=1.5185 \\pm 0.2823\\text{ s}^{-1}$), $\\mathcal{L}_{\\mathrm{kin}}$ cuts initial velocity error $e_{v,k}$ by $67.9\\%$ ($0.8768$ vs.\\ $2.7288$) and reduces the directional RUL bound by $42.8\\%$ (from $58.14 \\pm 10.65$ to $33.25 \\pm 3.22$ cycles, bracketing the $19.50 \\pm 0.54$ empirical RMSE), whereas in the uncapped full-horizon setting ($\\bar{h}_k=3.743, \\nu_{\\mathrm{Young}}=3.7179 \\pm 0.6606\\text{ s}^{-1}$), exponential Gr\\"{o}nwall growth saturates at the UUB ceiling $\\bar{R}_{\\mathrm{UUB}} = 616.53 \\pm 14.95$."""

new_c1 = """In Appendix~\\ref{app:proof_thm3} (Table~\\ref{tab:thm3_per_seed}), we verify across all 10 seeds that both model and target initial states lie strictly inside their UUB absorbing balls ($z_{0,\\mathrm{mod}}^{\\max} = 2.84 \\ll R_\\infty = 68.57$ and $z_{0,\\mathrm{tgt}}^{\\max} = 1.39 \\ll R_\\infty^* = 111.01$), separate continuous-time tracking error from measured Strang discretization error ($E_{\\mathrm{disc,RMS}} = 0.00010$ vs.\\ $0.0587$ uncapped), and distinguish the formal operator spectral-norm guarantee ($L_{g,\\mathrm{spec}} = 115.30 \\pm 6.30$, where $\\mathcal{L}_{\\mathrm{kin}}$ cuts the formal bound by $62.5\\%$ from $992.11 \\pm 194.93$ to $371.93 \\pm 46.99$ cycles) from an empirical directional sensitivity diagnostic ($\\hat{L}_{g,\\mathrm{dir}} = 14.27 \\pm 1.02$, reduced by $57.4\\%$ from $134.33 \\pm 28.70$ to $57.18 \\pm 7.50$ cycles)."""
assert old_c1 in tex, "Contrib 1 target not found"
tex = tex.replace(old_c1, new_c1, 1)

# 3. Replace Section 4.4 and Theorem 3
start_s44 = tex.index("\\subsection{Unified Young--Gr\\\"{o}nwall--Lyapunov Extrapolation Bound and Per-Seed Spectral Norm Audit}")
end_s44 = tex.index("\\subsection{Feedback-Free Offline Predicted-RUL Mondrian Calibration and Exchangeability Assumptions}")

new_s44_block = """\\subsection{Unified Young--Gr\\"{o}nwall--Lyapunov Extrapolation Bound and Per-Seed Spectral Norm Audit}
While Theorem~\\ref{thm:uub} guarantees that individual trajectories remain globally bounded for any $\\alpha_{\\mathrm{eff}} > 0$, bounding the terminal error between the $M$-step Strang-split state $\\hat{z}^{(M)} = \\Psi_{\\delta t}^{(M)}(z_k)$ ($\\delta t = h_k / M$) and the target trajectory $z^*(t_k + h_k) = [q^*(t_k + h_k); \\dot{q}^*(t_k + h_k)]$ requires separating the continuous-time ODE tracking error $e(t) = [\\delta q(t); \\delta p(t)] = \\hat{z}^{\\mathrm{exact}}(t) - z^*(t)$ from the numerical Strang-splitting discretization error $E_{\\mathrm{disc}}(h_k, \\delta t) \\triangleq \\|\\hat{z}^{(M)} - \\hat{z}^{\\mathrm{exact}}(t_k + h_k)\\|_2$ via the triangle inequality:
\\begin{equation}
\\big\\|\\hat{z}^{(M)} - z^*(t_k + h_k)\\big\\|_2 \\le \\big\\|e(t_k + h_k)\\big\\|_2 + E_{\\mathrm{disc}}(h_k, \\delta t).
\\label{eq:tri_split_err}
\\end{equation}
Partitioning $W_{\\mathrm{pot}}$ into its convex modes ($s_i = +1 > 0$, where $s_i \\mathrm{sech}^2(\\cdot)w_i w_i^T \\succeq 0$ adds to restoring stiffness) and non-convex modes ($s_i = -0.65 < 0$, submatrix $W_{\\mathrm{pot},-} \\in \\mathbb{R}^{d \\times 3}$), we define the two-sided Lipschitz constant $L_{\\mathrm{nl,full}}$ and one-sided non-convex monotonicity constant $L_{\\mathrm{nl}}^{(-)}$:
\\begin{align}
L_{\\mathrm{nl,full}} &= \\underbrace{\\frac{1}{d}\\|W_{\\mathrm{pot}}\\|_2^2}_{L_{V,\\mathrm{full}}} + c_f \\beta_q, \\label{eq:lnl_full} \\\\
L_{\\mathrm{nl}}^{(-)} &= \\underbrace{\\frac{0.65}{d}\\|W_{\\mathrm{pot},-}\\|_2^2}_{L_{V,-}} + c_f \\beta_q, \\label{eq:lnl_neg}
\\end{align}
where $c_f \\beta_q = 0.45 \\times 0.35 = 0.1575$. Differentiating $W(e) = \\frac{1}{2}e^T P e = \\frac{1}{2}\\|e\\|_P^2$ generates both a position inner product $\\epsilon\\delta q^T \\Delta F_{\\mathrm{nl}}(\\hat{q}, q^*) \\le \\epsilon L_{\\mathrm{nl}}^{(-)}\\|\\delta q\\|_2^2$ and a velocity-force cross-term $\\delta p^T \\Delta F_{\\mathrm{nl}}(\\hat{q}, q^*) \\le L_{\\mathrm{nl,full}}\\|\\delta q\\|_2 \\|\\delta p\\|_2$. Bounding $(\\epsilon\\gamma + L_{\\mathrm{nl,full}})\\|\\delta q\\|_2 \\|\\delta p\\|_2$ via Young's inequality (Appendix~\\ref{app:proof_thm3}) establishes:
\\begin{enumerate}
    \\item \\textbf{Why Strict Incremental Contraction Fails on Trained Checkpoints ($0/10$ Seeds):} Absorbing the velocity-force cross-term $(\\epsilon\\gamma + L_{\\mathrm{nl,full}})\\|\\delta q\\|_2 \\|\\delta p\\|_2$ into $-(\\gamma - \\epsilon)\\|\\delta p\\|_2^2$ via Young's inequality requires the strict small-gain condition $(\\epsilon\\gamma + L_{\\mathrm{nl,full}})^2 < 2\\epsilon(\\gamma - \\epsilon)(\\alpha_{\\mathrm{eff}} - L_{\\mathrm{nl}}^{(-)})$. Because $\\gamma = 0.15$ and $c_f\\beta_q = 0.1575$, $(\\epsilon\\gamma + L_{\\mathrm{nl,full}})^2 = 0.1489 \\pm 0.0273 \\gg 2\\epsilon(\\gamma - \\epsilon)(\\alpha_{\\mathrm{eff}} - L_{\\mathrm{nl}}^{(-)}) = 0.00141 \\pm 0.00027$, so \\textbf{neither the default ($\\alpha_{\\mathrm{eff}}=0.35$) nor the uncapped ($\\alpha_{\\mathrm{eff}}=0.10$) configuration is incrementally contractive in $W(e)$}.
    \\item \\textbf{Finite-Horizon Young--Gr\\"{o}nwall Regime with Verified Two-Trajectory UUB Capping ($10/10$ Seeds):} Choosing Young's parameter $\\eta = \\gamma - \\epsilon > 0$ cancels the positive $\\|\\delta p\\|_2^2$ cross-term and yields the finite-horizon Gr\\"{o}nwall growth rate $\\nu_{\\mathrm{Young}} = K_{\\mathrm{Young}} / (\\alpha_{\\mathrm{eff}} - \\epsilon^2) > 0$, where $K_{\\mathrm{Young}} = \\frac{(\\epsilon\\gamma + L_{\\mathrm{nl,full}})^2}{4(\\gamma - \\epsilon)} - \\epsilon(\\alpha_{\\mathrm{eff}} - L_{\\mathrm{nl}}^{(-)}) > 0$. Furthermore, directly from the completed-square identity $\\|e(t_k)\\|_P^2 = \\|\\delta p(t_k) + \\epsilon\\delta q(t_k)\\|_2^2 + (\\alpha_{\\mathrm{eff}} - \\epsilon^2)\\|\\delta q(t_k)\\|_2^2$, the initial $P$-norm error satisfies $\\|e(t_k)\\|_P \\le \\bar{e}_{P,k} \\triangleq \\sqrt{\\alpha_{\\mathrm{eff}}e_{q,k}^2 + 2\\epsilon e_{q,k}e_{v,k} + e_{v,k}^2}$, where $e_{v,k}$ enters with weight $1$ (not $h_k^2$), making kinematic velocity alignment $\\mathcal{L}_{\\mathrm{kin}}$ directly decisive for controlling $\\|e(t_k)\\|_P$.
\\end{enumerate}

\\begin{theorem}[Unified Gap Extrapolation Error Bound]
\\label{thm:extrap_bound}
Let $\\hat{z}^{\\mathrm{exact}}(t) = [\\hat{q}(t); \\hat{p}(t)]$ denote the continuous-time model trajectory of Eq.~\\eqref{eq:eom} on $t \\in [t_k, t_k + h_k]$ starting from $\\hat{z}^{\\mathrm{exact}}(t_k) = [q_k; \\dot{q}_k]$ with $\\|\\tilde{F}(\\hat{q})\\|_2 \\le L_V + F_{\\max}$ ($L_V = d^{-1/2}\\|W_{\\mathrm{pot}}\\|_2, F_{\\max} = c_f\\sqrt{d}$), and let $z^*(t) = [q^*(t); \\dot{q}^*(t)]$ denote a $C^2$ target trajectory satisfying $\\ddot{q}^*(t) + \\gamma \\dot{q}^*(t) + \\alpha_{\\mathrm{eff}}(q^*(t) - q_k) = \\tilde{F}(q^*(t)) + r_{\\mathrm{model}}(t)$ with within-window uniform residual supremum $\\varepsilon_{\\mathrm{model},k}^{\\sup} \\triangleq \\sup_{t \\in [t_k, t_k+h_k]}\\|r_{\\mathrm{model}}(t)\\|_2 < \\infty$. Let $e_{q,k} = \\|q_k - q^*(t_k)\\|_2$, $e_{v,k} = \\|\\dot{q}_k - \\dot{q}^*(t_k)\\|_2$, and $\\bar{e}_{P,k} = \\sqrt{\\alpha_{\\mathrm{eff}}e_{q,k}^2 + 2\\epsilon e_{q,k}e_{v,k} + e_{v,k}^2} \\ge \\|e(t_k)\\|_P$. Define the scaled initial displacement norms $z_{0,\\mathrm{mod},k} = \\|\\dot{q}_k\\|_2 / \\sqrt{\\lambda_{\\min}(P)}$ and $z_{0,\\mathrm{tgt},k} = \\kappa_P\\sqrt{e_{q,k}^2 + \\|\\dot{q}^*(t_k)\\|_2^2}$, and the asymptotic UUB radii $R_\\infty = \\sqrt{\\frac{2C_F}{\\mu\\lambda_{\\min}(P)}}$ and $R_{\\infty,k}^* = R_\\infty\\frac{L_V + F_{\\max} + \\varepsilon_{\\mathrm{model},k}^{\\sup}}{L_V + F_{\\max}}$. Then for all $t \\in [t_k, t_k + h_k]$, $\\|e(t)\\|_2 \\le \\bar{R}_{\\mathrm{UUB},k} \\triangleq \\max(z_{0,\\mathrm{mod},k}, R_\\infty) + \\max(z_{0,\\mathrm{tgt},k}, R_{\\infty,k}^*)$ (which equals $R_\\infty + R_{\\infty,k}^*$ whenever $z_{0,\\mathrm{mod},k} \\le R_\\infty$ and $z_{0,\\mathrm{tgt},k} \\le R_{\\infty,k}^*$, verified on $10/10$ seeds in Table~\\ref{tab:thm3_per_seed}). Let $L_{g,\\mathrm{spec}} = \\sigma_y \\|W_{\\mathrm{pr1},z}\\|_2 \\|W_{\\mathrm{pr2}}\\|_2$ denote the global operator spectral Lipschitz constant of the RUL probe $g_\\omega$, and let $E_{\\mathrm{disc}}(h_k, \\delta t) = \\|\\hat{z}^{(M)} - \\hat{z}^{\\mathrm{exact}}(t_k + h_k)\\|_2 = O(\\delta t^2)$ denote the $M$-step Strang-splitting discretization error.
\\begin{enumerate}
    \\item \\textbf{Small-Gain Contractive Regime ($(\\epsilon\\gamma + L_{\\mathrm{nl,full}})^2 < 2\\epsilon(\\gamma - \\epsilon)(\\alpha_{\\mathrm{eff}} - L_{\\mathrm{nl}}^{(-)})$, Satisfied on $0/10$ Trained Seeds):} With $\\eta = \\frac{\\gamma - \\epsilon}{2}$, $c_{q,\\mathrm{net}} = \\epsilon(\\alpha_{\\mathrm{eff}} - L_{\\mathrm{nl}}^{(-)}) - \\frac{(\\epsilon\\gamma + L_{\\mathrm{nl,full}})^2}{2(\\gamma - \\epsilon)} > 0$, $c_{p,\\mathrm{net}} = \\frac{\\gamma - \\epsilon}{2} > 0$, and $\\mu_{\\mathrm{net}} = \\frac{2\\min(c_{q,\\mathrm{net}}, c_{p,\\mathrm{net}})}{\\lambda_{\\max}(P)} > 0$:
    \\begin{align}
    &\\big|\\hat{\\mu}_{k+1} - y^*(t_{k+1})\\big| \\le \\varepsilon_{\\mathrm{probe},k} + L_{g,\\mathrm{spec}}\\Bigg[ E_{\\mathrm{disc}}(h_k, \\delta t) \\nonumber \\\\
    &\\quad + \\frac{1}{\\sqrt{\\lambda_{\\min}(P)}}\\Bigg( e^{-\\frac{1}{2}\\mu_{\\mathrm{net}} h_k}\\bar{e}_{P,k} + \\frac{1 - e^{-\\frac{1}{2}\\mu_{\\mathrm{net}} h_k}}{\\frac{1}{2}\\mu_{\\mathrm{net}}}\\varepsilon_{\\mathrm{model},k}^{\\sup} \\Bigg) \\Bigg].
    \\label{eq:thm3_bound}
    \\end{align}
    \\item \\textbf{Finite-Horizon Young--Gr\\"{o}nwall Regime ($(\\epsilon\\gamma + L_{\\mathrm{nl,full}})^2 \\ge 2\\epsilon(\\gamma - \\epsilon)(\\alpha_{\\mathrm{eff}} - L_{\\mathrm{nl}}^{(-)})$, Governing $10/10$ Trained Seeds for Both $\\alpha_{\\mathrm{eff}}=0.35$ and $\\alpha_{\\mathrm{eff}}=0.10$):} Setting $\\eta = \\gamma - \\epsilon$ yields $K_{\\mathrm{Young}} = \\frac{(\\epsilon\\gamma + L_{\\mathrm{nl,full}})^2}{4(\\gamma - \\epsilon)} - \\epsilon(\\alpha_{\\mathrm{eff}} - L_{\\mathrm{nl}}^{(-)}) > 0$ and $\\nu_{\\mathrm{Young}} = \\frac{K_{\\mathrm{Young}}}{\\alpha_{\\mathrm{eff}} - \\epsilon^2} > 0$:
    \\begin{align}
    &\\big|\\hat{\\mu}_{k+1} - y^*(t_{k+1})\\big| \\le \\varepsilon_{\\mathrm{probe},k} + L_{g,\\mathrm{spec}} E_{\\mathrm{disc}}(h_k, \\delta t) \\nonumber \\\\
    &\\quad + L_{g,\\mathrm{spec}} \\min\\!\\Bigg\\{\\bar{R}_{\\mathrm{UUB},k}, \\; \\frac{1}{\\sqrt{\\lambda_{\\min}(P)}}\\Bigg( e^{\\nu_{\\mathrm{Young}} h_k}\\bar{e}_{P,k} \\nonumber \\\\
    &\\quad + \\frac{e^{\\nu_{\\mathrm{Young}} h_k} - 1}{\\nu_{\\mathrm{Young}}}\\varepsilon_{\\mathrm{model},k}^{\\sup} \\Bigg)\\Bigg\\}.
    \\label{eq:thm3_gronwall}
    \\end{align}
    Taking the $\\ell_2$ root-mean-square ($\\mathrm{RMS}_w$) across test windows $w \\in \\{1,\\dots,N_{\\mathrm{win}}\\}$ via Minkowski's triangle inequality yields the formal fleet-wide RMS bound in terms of $\\bar{e}_{P,\\mathrm{RMS}}$, $\\varepsilon_{\\mathrm{model,RMS}} = \\mathrm{RMS}_w(\\varepsilon_{\\mathrm{model},w}^{\\sup})$, and $E_{\\mathrm{disc,RMS}}$, while taking $\\max_w$ with $\\varepsilon_{\\mathrm{model,sup}} = \\max_w \\varepsilon_{\\mathrm{model},w}^{\\sup}$ yields the worst-case uniform supremum bound.
\\end{enumerate}
\\end{theorem}
\\emph{Formal Spectral Guarantee vs.\\ Empirical Directional Sensitivity Diagnostic (Table~\\ref{tab:thm3_per_seed}):} See Appendix~\\ref{app:proof_thm3} for the complete proof and per-seed verification across all 10 seeds:
\\begin{itemize}
    \\item \\textbf{Formal Spectral-Norm Guarantee ($L_{g,\\mathrm{spec}} = 115.30 \\pm 6.30\\text{ cycles/unit}$):} In Panel~A ($\\alpha_{\\mathrm{eff}}=0.35, h_k=0.35$), all 10 seeds satisfy the initial UUB inclusion conditions ($z_{0,\\mathrm{mod}}^{\\max} = 2.84 \\pm 0.25 \\ll R_\\infty = 68.57 \\pm 1.55$ and $z_{0,\\mathrm{tgt}}^{\\max} = 1.39 \\pm 0.28 \\ll R_\\infty^* = 111.01 \\pm 5.67$, giving $\\bar{R}_{\\mathrm{UUB}} = 179.58 \\pm 5.47$), and the measured Strang discretization error against a 400-substep reference solution is $E_{\\mathrm{disc,RMS}} = 0.00010 \\pm 0.00001$ ($E_{\\mathrm{disc,max}} = 0.00020 \\pm 0.00003$). Secant velocity alignment ($\\mathcal{L}_{\\mathrm{kin}}$) cuts initial velocity error $e_{v,\\mathrm{RMS}}$ by \\textbf{$67.9\\%$} ($2.7288 \\to 0.8768$), initial $P$-norm error $\\bar{e}_{P,\\mathrm{RMS}}$ by \\textbf{$67.0\\%$} ($2.7520 \\to 0.9076$), and total Euclidean latent error $\\bar{e}_{\\mathrm{lat,RMS}}$ by \\textbf{$63.3\\%$} ($8.4947 \\to 3.1192$ units). Consequently, $\\mathcal{L}_{\\mathrm{kin}}$ reduces the \\textbf{formal spectral-norm RMS bound by $62.5\\%$, from $992.11 \\pm 194.93$ cycles (without $\\mathcal{L}_{\\mathrm{kin}}$) to $371.93 \\pm 46.99$ cycles (with $\\mathcal{L}_{\\mathrm{kin}}$)} (and worst-case supremum bound $734.57 \\pm 81.22$ cycles).
    \\item \\textbf{Semi-Empirical Directional Sensitivity Diagnostic ($\\hat{L}_{g,\\mathrm{dir}} = 14.27 \\pm 1.02\\text{ cycles/unit}$):} Because the global spectral norm $L_{g,\\mathrm{spec}}$ bounds the worst-case singular direction of $W_{\\mathrm{pr1},z}$ over $\\mathbb{R}^{2d}$ assuming $\\mathrm{sech}^2(\\cdot) = 1$, it is $8.1\\times$ larger than the empirical directional chord sensitivity $\\hat{L}_{g,\\mathrm{dir}} = \\mathrm{RMS}_w(|g_\\omega(\\hat{z}_w) - g_\\omega(z_w^*)|) / \\mathrm{RMS}_w(\\|\\hat{z}_w - z_w^*\\|_2) = 14.27 \\pm 1.02\\text{ cycles/unit}$ along test trajectories. Substituting $\\hat{L}_{g,\\mathrm{dir}}$ in place of $L_{g,\\mathrm{spec}}$ in Eq.~\\eqref{eq:thm3_gronwall} provides a \\textbf{post-hoc semi-empirical diagnostic (not a formal a priori guarantee)} that drops by \\textbf{$57.4\\%$, from $134.33 \\pm 28.70$ cycles (without $\\mathcal{L}_{\\mathrm{kin}}$) to $57.18 \\pm 7.50$ cycles (with $\\mathcal{L}_{\\mathrm{kin}}$)}, bracketing the empirical \\texttt{OOD 10x--20x} RMSE ($19.50 \\pm 0.54$ cycles) on every seed.
\\end{itemize}

"""

tex = tex[:start_s44] + new_s44_block + tex[end_s44:]

# 4. Update Section 6.4 Item 1, Section 8 Item 4, and Section 9 Conclusion to reflect the exact v8 numbers
tex = tex.replace(
    "saturates at the UUB ceiling $R_\\infty + R_\\infty^* = 616.53 \\pm 14.95$",
    "saturates at the verified UUB ceiling $\\bar{R}_{\\mathrm{UUB}} = R_\\infty + R_\\infty^* = 674.26 \\pm 19.36$ (while Strang discretization error $E_{\\mathrm{disc,RMS}}$ grows $587\\times$ from $0.00010$ to $0.0587$)"
)
tex = tex.replace(
    "keeps $\\nu_{\\mathrm{Young}}h_k \\approx 0.53$ and yields a non-vacuous $33.25 \\pm 3.22$-cycle RUL bound, uncapped full-horizon integration ($h_{\\max}=\\infty, \\alpha_{\\mathrm{eff}}=0.10, \\bar{h}_k=3.743$) causes exponential Gr\\"{o}nwall blow-up to the UUB ceiling ($616.53 \\pm 14.95$ latent units)",
    "keeps $\\nu_{\\mathrm{Young}}h_k \\approx 0.53$ (reducing the formal spectral bound by $62.5\\%$ from $992.11$ to $371.93$ cycles and the empirical directional diagnostic by $57.4\\%$ from $134.33$ to $57.18$ cycles), uncapped full-horizon integration ($h_{\\max}=\\infty, \\alpha_{\\mathrm{eff}}=0.10, \\bar{h}_k=3.743$) causes exponential Gr\\"{o}nwall blow-up to the UUB ceiling ($\\bar{R}_{\\mathrm{UUB}} = 674.26 \\pm 19.36$ latent units)"
)
tex = tex.replace(
    "reduces the finite-horizon Young--Gr\\"{o}nwall extrapolation bound by $42.8\\%$ via secant velocity alignment",
    "reduces the formal spectral-norm Young--Gr\\"{o}nwall extrapolation bound by $62.5\\%$ ($57.4\\%$ on the empirical directional diagnostic) via secant velocity alignment"
)

# 5. Replace Appendix C and Table 8 with the complete v8 derivation and verification table
start_appc = tex.index("\\section{Proof of Theorem 3, Two-Trajectory UUB Derivation, and Complete 10-Seed Default \\& Uncapped Reproducibility Table}")
end_appc = tex.index("\\bibliographystyle{IEEEtran}")

new_appc_block = """\\section{Proof of Theorem 3, Two-Trajectory UUB Verification, Strang Discretization Error, and Complete 10-Seed Reproducibility Table}
\\label{app:proof_thm3}

\\begin{table*}[!t]
\\centering
\\caption{Complete Per-Seed Trained Spectral Norms, Young's Inequality Constants ($K_{\\mathrm{Young}}, \\nu_{\\mathrm{Young}}$), Verified Two-Trajectory UUB Initial Conditions \\& Radii ($z_{0,\\mathrm{mod}}^{\\max} \\le R_\\infty, z_{0,\\mathrm{tgt}}^{\\max} \\le R_\\infty^*, \\bar{R}_{\\mathrm{UUB}} = R_\\infty + R_\\infty^*$), Equation~(36) Initial $P$-Norms ($\\bar{e}_{P,\\mathrm{RMS}}$), Within-Window Residual RMS \\& Uniform Supremum ($\\varepsilon_{\\mathrm{mod,RMS}}, \\varepsilon_{\\mathrm{mod,sup}}$), Measured Strang Discretization Error ($E_{\\mathrm{disc,RMS}}$), Formal Spectral-Norm Bounds ($L_{g,\\mathrm{spec}}$), and Semi-Empirical Directional Diagnostics ($\\hat{L}_{g,\\mathrm{dir}}$) Across All 10 Seeds on FD001.}
\\label{tab:thm3_per_seed}
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{cccccccccccccccccccccc}
\\toprule
\\textbf{Seed} & \\textbf{$\\|W_{\\mathrm{pot}}\\|_2$} & \\textbf{$L_{\\mathrm{nl,full}}$} & \\textbf{$L_{\\mathrm{nl}}^{(-)}$} & \\textbf{$K_{\\mathrm{Young}}$} & \\textbf{$\\nu_{\\mathrm{Young}}$} & \\textbf{$z_{0,\\mathrm{mod}}^{\\max} / R_\\infty$} & \\textbf{$z_{0,\\mathrm{tgt}}^{\\max} / R_\\infty^*$} & \\textbf{$\\bar{R}_{\\mathrm{UUB}}$} & \\textbf{$e_{q,\\mathrm{RMS}}$} & \\textbf{$e_{v,\\mathrm{RMS}}$ (Kin/No)} & \\textbf{$\\bar{e}_{P,\\mathrm{RMS}}$ (Kin/No)} & \\textbf{$\\varepsilon_{\\mathrm{mod}}$ (RMS/Sup)} & \\textbf{$E_{\\mathrm{disc,RMS}}$} & \\textbf{$\\bar{e}_{\\mathrm{lat,RMS}}$ (Kin/No)} & \\textbf{$L_{g,\\mathrm{spec}}$} & \\textbf{$\\hat{L}_{g,\\mathrm{dir}}$} & \\textbf{Formal Spec Bound (Kin/No)} & \\textbf{ Formal Spec Sup} & \\textbf{Emp Dir Diag (Kin/No)$^\\dagger$} & \\textbf{Emp OOD RMSE} \\\\
\\midrule
\\multicolumn{21}{l}{\\textbf{Panel A: Default Bounded-Horizon Configuration ($\\alpha_{\\mathrm{eff}}=0.35, c_\\alpha=3.5, h_k = h_{\\max} = 0.35, \\gamma=0.15, \\epsilon = 0.08, \\lambda_{\\min}(P)=0.3403, \\kappa_P = 1.7225$) --- UUB Initial Inclusion Verified on 10/10 Seeds}} \\\\
\\midrule
$42$   & $1.3075$ & $0.3712$ & $0.2319$ & $0.5150$ & $1.4987$ & $2.75 / 68.58$ & $1.36 / 120.13$ & $188.71$ & $0.2193$ & $0.80 / 2.16$ & $0.82 / 2.18$ & $0.58 / 1.30$ & $0.00008$ & $2.84 / 6.76$ & $106.01$ & $13.57$ & $313.4 / 728.9$  & $661.3$ & $50.60 / 103.80$ & $19.39$ \\\\
$100$  & $1.1795$ & $0.3314$ & $0.1944$ & $0.4087$ & $1.1895$ & $2.76 / 66.79$ & $1.99 / 116.36$ & $183.15$ & $0.2905$ & $0.85 / 2.71$ & $0.89 / 2.74$ & $0.53 / 1.25$ & $0.00009$ & $2.70 / 7.52$ & $130.33$ & $14.95$ & $364.2 / 992.5$  & $727.7$ & $52.74 / 124.81$ & $19.80$ \\\\
$2024$ & $1.1213$ & $0.3147$ & $0.1891$ & $0.3683$ & $1.0718$ & $2.91 / 65.98$ & $1.15 / 111.66$ & $177.64$ & $0.2433$ & $0.91 / 3.02$ & $0.94 / 3.04$ & $0.64 / 1.16$ & $0.00010$ & $2.82 / 8.05$ & $111.71$ & $13.70$ & $327.4 / 912.5$  & $635.4$ & $51.42 / 123.20$ & $19.73$ \\\\
$777$  & $1.4229$ & $0.4106$ & $0.2090$ & $0.6265$ & $1.8234$ & $2.85 / 70.19$ & $1.66 / 114.57$ & $184.76$ & $0.2297$ & $0.85 / 2.67$ & $0.88 / 2.70$ & $0.65 / 1.12$ & $0.00011$ & $3.41 / 9.30$ & $111.65$ & $14.65$ & $392.6 / 1050.0$ & $853.1$ & $61.99 / 148.24$ & $20.53$ \\\\
$999$  & $1.3986$ & $0.4020$ & $0.2637$ & $0.6052$ & $1.7614$ & $2.49 / 69.85$ & $1.25 / 105.43$ & $175.28$ & $0.2047$ & $0.75 / 1.94$ & $0.78 / 1.96$ & $0.44 / 0.90$ & $0.00008$ & $2.84 / 6.60$ & $116.92$ & $15.43$ & $344.5 / 784.8$  & $707.2$ & $56.61 / 114.74$ & $19.53$ \\\\
$101$  & $1.4111$ & $0.4064$ & $0.2123$ & $0.6142$ & $1.7875$ & $2.86 / 70.03$ & $1.50 / 105.97$ & $175.99$ & $0.2318$ & $0.87 / 2.53$ & $0.90 / 2.55$ & $0.52 / 0.91$ & $0.00011$ & $3.32 / 8.60$ & $117.55$ & $13.41$ & $403.1 / 1022.8$ & $834.1$ & $57.00 / 127.69$ & $19.00$ \\\\
$202$  & $1.3236$ & $0.3765$ & $0.2322$ & $0.5295$ & $1.5412$ & $2.69 / 68.80$ & $1.07 / 108.26$ & $177.06$ & $0.2686$ & $0.94 / 2.99$ & $0.97 / 3.02$ & $0.62 / 1.00$ & $0.00010$ & $3.36 / 9.36$ & $113.40$ & $13.10$ & $393.7 / 1074.5$ & $680.8$ & $56.85 / 135.51$ & $19.61$ \\\\
$303$  & $1.4340$ & $0.4145$ & $0.2234$ & $0.6396$ & $1.8616$ & $2.77 / 70.35$ & $1.21 / 107.98$ & $178.33$ & $0.2215$ & $0.93 / 3.12$ & $0.96 / 3.14$ & $0.60 / 0.95$ & $0.00011$ & $3.67 / 10.83$ & $117.37$ & $15.72$ & $443.5 / 1284.1$ & $790.1$ & $70.31 / 182.88$ & $19.16$ \\\\
$404$  & $1.2503$ & $0.3529$ & $0.2529$ & $0.4678$ & $1.3614$ & $3.48 / 67.78$ & $1.43 / 116.55$ & $184.33$ & $0.3048$ & $1.07 / 3.80$ & $1.11 / 3.82$ & $0.71 / 1.23$ & $0.00012$ & $3.61 / 11.10$ & $114.48$ & $15.18$ & $426.0 / 1283.7$ & $809.9$ & $67.47 / 181.18$ & $19.76$ \\\\
$505$  & $1.2209$ & $0.3438$ & $0.2328$ & $0.4428$ & $1.2888$ & $2.80 / 67.37$ & $1.23 / 103.22$ & $170.59$ & $0.2174$ & $0.79 / 2.35$ & $0.82 / 2.38$ & $0.56 / 0.91$ & $0.00010$ & $2.62 / 6.82$ & $113.62$ & $13.01$ & $310.9 / 787.3$  & $646.3$ & $46.76 / 101.29$ & $18.51$ \\\\
\\midrule
\\textbf{Mean} & $\\mathbf{1.3070}$ & $\\mathbf{0.3724}$ & $\\mathbf{0.2242}$ & $\\mathbf{0.5218}$ & $\\mathbf{1.5185}$ & $\\mathbf{2.84 / 68.57}$ & $\\mathbf{1.39 / 111.01}$ & $\\mathbf{179.58}$ & $\\mathbf{0.2432}$ & $\\mathbf{0.88 / 2.73}$ & $\\mathbf{0.91 / 2.75}$ & $\\mathbf{0.59 / 1.07}$ & $\\mathbf{0.00010}$ & $\\mathbf{3.12 / 8.49}$ & $\\mathbf{115.30}$ & $\\mathbf{14.27}$ & $\\mathbf{371.9 / 992.1}$ & $\\mathbf{734.6}$ & $\\mathbf{57.18 / 134.33}$ & $\\mathbf{19.50}$ \\\\
\\textbf{SD}   & $0.1108$ & $0.0357$ & $0.0237$ & $0.0977$ & $0.2844$ & $0.25 / 1.55$  & $0.28 / 5.67$   & $5.47$   & $0.0336$ & $0.09 / 0.53$ & $0.10 / 0.54$ & $0.08 / 0.16$ & $0.00001$ & $0.39 / 1.64$ & $6.30$   & $1.02$  & $47.0 / 194.9$   & $81.2$  & $7.50 / 28.70$   & $0.54$ \\\\
\\midrule
\\multicolumn{21}{l}{\\textbf{Panel B: Uncapped Full-Horizon Configuration ($\\alpha_{\\mathrm{eff}}=0.10, c_\\alpha=1.0, h_{\\max}=\\infty, \\bar{h}_k=3.743, \\gamma=0.15, \\epsilon=0.05, \\lambda_{\\min}(P)=0.0972, \\kappa_P=3.2115$) --- End-to-End Retrained Checkpoints$^\\ddagger$}} \\\\
\\midrule
$42$   & $1.3005$ & $0.3689$ & $0.2326$ & $0.3609$ & $3.7013$ & $5.35 / 260.98$ & $2.54 / 438.77$ & $699.75$ & $0.2519$ & $0.84 / 2.22$ & $0.86 / 2.23$ & $0.56 / 1.18$ & $0.06106$ & $1.62\\mathrm{e}6$ & $105.45$ & $6.95$  & $73803.5^\\S$ (UUB) & $73848.4^\\S$ & $4876.2^\\S$ (UUB) & $26.01$ \\\\
$100$  & $1.1737$ & $0.3297$ & $0.1959$ & $0.2890$ & $2.9644$ & $5.48 / 254.22$ & $3.84 / 438.92$ & $693.14$ & $0.3273$ & $0.90 / 2.74$ & $0.92 / 2.76$ & $0.50 / 1.23$ & $0.04864$ & $1.11\\mathrm{e}5$ & $126.04$ & $6.81$  & $87373.9^\\S$ (UUB) & $87423.9^\\S$ & $4732.7^\\S$ (UUB) & $22.68$ \\\\
$2024$ & $1.1218$ & $0.3148$ & $0.1885$ & $0.2641$ & $2.7088$ & $5.78 / 251.46$ & $2.27 / 416.32$ & $667.78$ & $0.2856$ & $0.98 / 3.16$ & $1.00 / 3.17$ & $0.62 / 1.09$ & $0.06092$ & $6.36\\mathrm{e}4$ & $113.68$ & $10.73$ & $75924.3^\\S$ (UUB) & $75979.6^\\S$ & $7180.1^\\S$ (UUB) & $42.49$ \\\\
$777$  & $1.3730$ & $0.3931$ & $0.2096$ & $0.4068$ & $4.1720$ & $5.73 / 264.84$ & $3.11 / 429.82$ & $694.66$ & $0.2623$ & $0.89 / 2.71$ & $0.91 / 2.72$ & $0.65 / 1.10$ & $0.06597$ & $9.11\\mathrm{e}6$ & $110.29$ & $7.50$  & $76629.5^\\S$ (UUB) & $76678.2^\\S$ & $5219.0^\\S$ (UUB) & $29.36$ \\\\
$999$  & $1.4013$ & $0.4030$ & $0.2663$ & $0.4295$ & $4.4054$ & $5.00 / 266.35$ & $2.50 / 398.64$ & $664.99$ & $0.2222$ & $0.78 / 1.93$ & $0.79 / 1.95$ & $0.42 / 0.88$ & $0.05246$ & $1.86\\mathrm{e}7$ & $112.41$ & $8.32$  & $74766.7^\\S$ (UUB) & $74812.9^\\S$ & $5545.4^\\S$ (UUB) & $27.15$ \\\\
$101$  & $1.3901$ & $0.3990$ & $0.2128$ & $0.4188$ & $4.2958$ & $5.68 / 265.75$ & $2.80 / 389.18$ & $654.93$ & $0.2608$ & $0.90 / 2.52$ & $0.92 / 2.53$ & $0.48 / 0.82$ & $0.05024$ & $1.56\\mathrm{e}7$ & $116.61$ & $6.53$  & $76383.4^\\S$ (UUB) & $76438.7^\\S$ & $4287.4^\\S$ (UUB) & $25.00$ \\\\
$202$  & $1.3166$ & $0.3742$ & $0.2338$ & $0.3709$ & $3.8040$ & $5.37 / 261.84$ & $2.18 / 404.10$ & $665.94$ & $0.3101$ & $1.00 / 3.08$ & $1.02 / 3.10$ & $0.60 / 0.94$ & $0.05727$ & $2.66\\mathrm{e}6$ & $111.31$ & $8.06$  & $74135.3^\\S$ (UUB) & $74182.5^\\S$ & $5381.6^\\S$ (UUB) & $27.15$ \\\\
$303$  & $1.4267$ & $0.4119$ & $0.2232$ & $0.4460$ & $4.5739$ & $5.52 / 267.70$ & $2.35 / 408.15$ & $675.85$ & $0.2672$ & $0.98 / 3.18$ & $1.00 / 3.19$ & $0.58 / 0.93$ & $0.06789$ & $4.96\\mathrm{e}7$ & $117.14$ & $7.29$  & $79178.5^\\S$ (UUB) & $79223.2^\\S$ & $4939.6^\\S$ (UUB) & $27.71$ \\\\
$404$  & $1.2406$ & $0.3499$ & $0.2520$ & $0.3269$ & $3.3530$ & $6.90 / 257.79$ & $2.84 / 428.57$ & $686.36$ & $0.3600$ & $1.16 / 3.88$ & $1.19 / 3.90$ & $0.64 / 1.13$ & $0.06603$ & $6.80\\mathrm{e}5$ & $115.13$ & $14.42$ & $79034.2^\\S$ (UUB) & $79080.7^\\S$ & $9906.4^\\S$ (UUB) & $52.41$ \\\\
$505$  & $1.2127$ & $0.3413$ & $0.2361$ & $0.3110$ & $3.1900$ & $5.24 / 256.30$ & $2.44 / 382.86$ & $639.16$ & $0.2480$ & $0.82 / 2.35$ & $0.83 / 2.36$ & $0.52 / 0.84$ & $0.05654$ & $2.53\\mathrm{e}5$ & $116.02$ & $12.14$ & $74165.8^\\S$ (UUB) & $74209.5^\\S$ & $7773.9^\\S$ (UUB) & $46.35$ \\\\
\\midrule
\\textbf{Mean} & $\\mathbf{1.2957}$ & $\\mathbf{0.3686}$ & $\\mathbf{0.2251}$ & $\\mathbf{0.3624}$ & $\\mathbf{3.7169}$ & $\\mathbf{5.61 / 260.72}$ & $\\mathbf{2.69 / 413.53}$ & $\\mathbf{674.26}$ & $\\mathbf{0.2795}$ & $\\mathbf{0.92 / 2.78}$ & $\\mathbf{0.94 / 2.79}$ & $\\mathbf{0.56 / 1.01}$ & $\\mathbf{0.05870}$ & $\\mathbf{9.83\\mathrm{e}6}$ & $\\mathbf{114.41}$ & $\\mathbf{8.88}$ & $\\mathbf{77139.5^\\S}$ & $\\mathbf{77187.8^\\S}$ & $\\mathbf{5984.2^\\S}$ & $\\mathbf{32.63}$ \\\\
\\textbf{SD}   & $0.1047$ & $0.0335$ & $0.0242$ & $0.0630$ & $0.6456$ & $0.51 / 5.58$   & $0.50 / 20.15$  & $19.36$  & $0.0416$ & $0.11 / 0.57$ & $0.11 / 0.57$ & $0.07 / 0.15$ & $0.00680$ & $1.56\\mathrm{e}7$ & $5.40$   & $2.66$ & $4074.7^\\S$  & $4075.4^\\S$  & $1761.4^\\S$  & $10.40$ \\\\
\\bottomrule
\\multicolumn{21}{l}{\\footnotesize $^\\dagger$\\textbf{Emp Dir Diag} substitutes the empirical test-set directional chord ratio $\\hat{L}_{g,\\mathrm{dir}}$ in place of $L_{g,\\mathrm{spec}}$ and is reported strictly as a semi-empirical sensitivity diagnostic, whereas \\textbf{Formal Spec Bound} ($L_{g,\\mathrm{spec}}$) is the formal a priori mathematical guarantee.} \\\\
\\multicolumn{21}{l}{\\footnotesize $^\\ddagger$Artifact disclosure: In \\texttt{Ablation\\_MatchedAnchor\\_hmax\\_Dim.csv} (Table~\\ref{tab:ablation_design}), the $18.90 \\pm 0.58$ uncapped row was evaluated via post-hoc drift adjustment on the Panel~A checkpoints without saving separate uncapped \\texttt{.pt} files; Panel~B reports the exact end-to-end retrained uncapped checkpoints ($32.63 \\pm 10.40$ RMSE).} \\\\
\\multicolumn{21}{l}{\\footnotesize $^\\S$In Panel~B ($\\bar{h}_k=3.743$), the continuous-time Gr\\"{o}nwall term ($\\bar{e}_{\\mathrm{lat,RMS}} \\sim 6.36\\times 10^4$ to $4.96\\times 10^7$) exceeds $\\bar{R}_{\\mathrm{UUB}} = R_\\infty + R_\\infty^*$ on all 10 seeds, so $\\min\\{\\bar{R}_{\\mathrm{UUB}}, \\text{Gr\\"{o}nwall}\\} = \\bar{R}_{\\mathrm{UUB}}$ (vacuous relative to $Y_{\\max}=125$).}
\\end{tabular}%
}
\\end{table*}

\\textbf{1. Triangle-Inequality Separation of Continuous-Time Tracking Error and Strang Discretization Error:} Let $\\hat{z}^{(M)} = \\Psi_{\\delta t}^{(M)}(z_k)$ denote the $M$-step Strang-split discrete state at $t_k + h_k$ ($\\delta t = h_k / M$), and let $\\hat{z}^{\\mathrm{exact}}(t)$ denote the exact continuous-time solution of Eq.~\\eqref{eq:eom} on $t \\in [t_k, t_k + h_k]$ starting from $\\hat{z}^{\\mathrm{exact}}(t_k) = z_k = [q_k; \\dot{q}_k]$. By the triangle inequality, $\\|\\hat{z}^{(M)} - z^*(t_k + h_k)\\|_2 \\le \\|e(t_k + h_k)\\|_2 + E_{\\mathrm{disc}}(h_k, \\delta t)$, where $e(t) = [\\delta q(t); \\delta p(t)] = \\hat{z}^{\\mathrm{exact}}(t) - z^*(t)$ is the continuous-time tracking error and $E_{\\mathrm{disc}}(h_k, \\delta t) = \\|\\hat{z}^{(M)} - \\hat{z}^{\\mathrm{exact}}(t_k + h_k)\\|_2$. For the symmetric Strang splitting $\\Psi_{\\delta t} = \\mathcal{D}_{\\delta t/2} \\circ \\mathcal{S}_{\\delta t} \\circ \\mathcal{D}_{\\delta t/2}$ of the $C^3$ vector field in Eq.~\\eqref{eq:eom} with global Lipschitz constant $L_f = \\sqrt{\\max(1, (\\alpha_{\\mathrm{eff}} + L_{\\mathrm{nl,full}})^2) + \\gamma^2}$, Baker--Campbell--Hausdorff (BCH) expansion gives local error $\\|\\Psi_{\\delta t}(z) - \\Phi_{\\delta t}^{\\mathrm{exact}}(z)\\|_2 \\le c_{\\mathrm{loc}}\\delta t^3$ and global error $E_{\\mathrm{disc}}(h_k, \\delta t) \\le \\frac{e^{L_f h_k}-1}{L_f}c_{\\mathrm{loc}}\\delta t^2$ \\cite{hairer2006geometric}. Direct comparison against a 400-substep reference solution across all 10 seeds (Table~\\ref{tab:thm3_per_seed}) confirms $E_{\\mathrm{disc,RMS}} = 0.00010 \\pm 0.00001$ ($E_{\\mathrm{disc,max}} = 0.00020 \\pm 0.00003$) in Panel~A ($h_k=0.35, \\delta t=0.0875$) vs.\\ $0.05870 \\pm 0.00680$ in Panel~B ($\\bar{h}_k=3.743, \\overline{\\delta t}=0.936$).

\\textbf{2. Exact Initial-Error Bound and Norm Conversion from Equation~(36):} Subtracting $\\ddot{q}^* = F_{\\mathrm{tot}}(q^*) - \\gamma \\dot{q}^* + r_{\\mathrm{model}}(t)$ from Eq.~\\eqref{eq:eom} gives:
\\begin{equation}
\\frac{d}{dt}\\begin{bmatrix} \\delta q \\\\ \\delta p \\end{bmatrix} = \\begin{bmatrix} \\delta p \\\\ -\\alpha_{\\mathrm{eff}}\\delta q - \\gamma \\delta p + \\Delta F_{\\mathrm{nl}}(\\hat{q}, q^*) - r_{\\mathrm{model}}(t) \\end{bmatrix}.
\\label{eq:err_dyn_app}
\\end{equation}
Completing the square in $W(e) = \\frac{1}{2}e^T P e = \\frac{1}{2}\\|\\delta p\\|_2^2 + \\frac{1}{2}\\alpha_{\\mathrm{eff}}\\|\\delta q\\|_2^2 + \\epsilon \\delta q^T \\delta p$ yields the exact identity:
\\begin{equation}
\\|e\\|_P^2 = 2W(e) = \\|\\delta p + \\epsilon \\delta q\\|_2^2 + (\\alpha_{\\mathrm{eff}} - \\epsilon^2)\\|\\delta q\\|_2^2,
\\label{eq:comp_square}
\\end{equation}
which implies both $\\|\\delta p + \\epsilon \\delta q\\|_2 \\le \\|e\\|_P$ and $\\|\\delta q\\|_2^2 \\le \\frac{\\|e\\|_P^2}{\\alpha_{\\mathrm{eff}} - \\epsilon^2}$. At $t = t_k$, where $\\|\\delta q(t_k)\\|_2 = e_{q,k}$ and $\\|\\delta p(t_k)\\|_2 = e_{v,k}$, Expanding Eq.~\\eqref{eq:comp_square} and applying Cauchy--Schwarz ($\\delta q(t_k)^T \\delta p(t_k) \\le e_{q,k}e_{v,k}$) yields the exact initial $P$-norm bound:
\\begin{align}
\\|e(t_k)\\|_P &= \\sqrt{\\|\\delta p(t_k)\\|_2^2 + 2\\epsilon \\delta q(t_k)^T \\delta p(t_k) + \\alpha_{\\mathrm{eff}}\\|\\delta q(t_k)\\|_2^2} \\nonumber \\\\
&\\le \\sqrt{\\alpha_{\\mathrm{eff}} e_{q,k}^2 + 2\\epsilon e_{q,k} e_{v,k} + e_{v,k}^2} \\triangleq \\bar{e}_{P,k}.
\\label{eq:init_p_bound}
\\end{align}
At $t = t_k + h_k$, the Rayleigh quotient $\\lambda_{\\min}(P)\\|e(t_k + h_k)\\|_2^2 \\le \\|e(t_k + h_k)\\|_P^2$ converts the terminal $P$-norm bound to the Euclidean norm via $\\|e(t_k + h_k)\\|_2 \\le \\frac{1}{\\sqrt{\\lambda_{\\min}(P)}}\\|e(t_k + h_k)\\|_P$.

\\textbf{3. Young's Inequality Derivation and Within-Window Uniform Residual Supremum ($\\varepsilon_{\\mathrm{model},k}^{\\sup}$):} Differentiating $W(e(t))$ along Eq.~\\eqref{eq:err_dyn_app}, using $\\epsilon\\delta q^T \\Delta F_{\\mathrm{nl}} \\le \\epsilon L_{\\mathrm{nl}}^{(-)}\\|\\delta q\\|_2^2$ (by positive semi-definiteness of the 5 convex modes $s_i = +1$), $|\\delta p^T \\Delta F_{\\mathrm{nl}}| \\le L_{\\mathrm{nl,full}}\\|\\delta q\\|_2 \\|\\delta p\\|_2$ (by two-sided Lipschitz continuity), and $|-(\\delta p + \\epsilon\\delta q)^T r_{\\mathrm{model}}(t)| \\le \\|e(t)\\|_P \\varepsilon_{\\mathrm{model},k}^{\\sup}$ where $\\varepsilon_{\\mathrm{model},k}^{\\sup} = \\sup_{t \\in [t_k, t_k+h_k]}\\|r_{\\mathrm{model}}(t)\\|_2$ is the genuine uniform supremum over the window $[t_k, t_k + h_k]$, gives:
\\begin{align}
\\frac{d}{dt}W(e) &\\le -\\epsilon\\big(\\alpha_{\\mathrm{eff}} - L_{\\mathrm{nl}}^{(-)}\\big)\\|\\delta q\\|_2^2 - (\\gamma - \\epsilon)\\|\\delta p\\|_2^2 \\nonumber \\\\
&\\quad + \\big(\\epsilon\\gamma + L_{\\mathrm{nl,full}}\\big)\\|\\delta q\\|_2 \\|\\delta p\\|_2 + \\|e\\|_P \\varepsilon_{\\mathrm{model},k}^{\\sup}.
\\label{eq:corrected_eq36}
\\end{align}
Applying Young's inequality $(\\epsilon\\gamma + L_{\\mathrm{nl,full}})\\|\\delta q\\|_2 \\|\\delta p\\|_2 \\le \\frac{(\\epsilon\\gamma + L_{\\mathrm{nl,full}})^2}{4\\eta}\\|\\delta q\\|_2^2 + \\eta \\|\\delta p\\|_2^2$ for $\\eta \\in (0, \\gamma - \\epsilon]$ gives:
\\begin{align}
\\frac{d}{dt}W(e) &\\le -\\left[\\epsilon\\big(\\alpha_{\\mathrm{eff}} - L_{\\mathrm{nl}}^{(-)}\\big) - \\frac{\\big(\\epsilon\\gamma + L_{\\mathrm{nl,full}}\\big)^2}{4\\eta}\\right]\\|\\delta q\\|_2^2 \\nonumber \\\\
&\\quad - (\\gamma - \\epsilon - \\eta)\\|\\delta p\\|_2^2 + \\|e\\|_P \\varepsilon_{\\mathrm{model},k}^{\\sup}.
\\label{eq:young_master}
\\end{align}
For $\\eta = \\gamma - \\epsilon$ (Case~2, governing $10/10$ seeds), using $\\|\\delta q\\|_2^2 \\le \\frac{\\|e\\|_P^2}{\\alpha_{\\mathrm{eff}} - \\epsilon^2}$ and dividing by $\\|e\\|_P$ yields $\\frac{d}{dt}\\|e(t)\\|_P \\le \\nu_{\\mathrm{Young}}\\|e(t)\\|_P + \\varepsilon_{\\mathrm{model},k}^{\\sup}$ with $\\nu_{\\mathrm{Young}} = \\frac{K_{\\mathrm{Young}}}{\\alpha_{\\mathrm{eff}} - \\epsilon^2}$. Integrating on $[t_k, t_k + h_k]$ via Gr\\"{o}nwall's lemma and dividing by $\\sqrt{\\lambda_{\\min}(P)}$ yields Eq.~\\eqref{eq:thm3_gronwall}.

\\textbf{4. Verification of Two-Trajectory UUB Initial Conditions Across All 10 Seeds:} Integrating $\\dot{W}(\\tilde{\\hat{z}}) \\le -\\mu W(\\tilde{\\hat{z}}) + C_F$ and $\\dot{W}(\\tilde{z}^*) \\le -\\mu W(\\tilde{z}^*) + C_F^*$ on $[t_k, t]$ gives $\\|\\tilde{\\hat{z}}(t)\\|_2 \\le \\max(z_{0,\\mathrm{mod},k}, R_\\infty)$ and $\\|\\tilde{z}^*(t)\\|_2 \\le \\max(z_{0,\\mathrm{tgt},k}, R_{\\infty,k}^*)$. As verified in Table~\\ref{tab:thm3_per_seed} across all 10 seeds and all test windows, the worst-case initial conditions satisfy $z_{0,\\mathrm{mod}}^{\\max} = 2.84 \\pm 0.25 \\le R_\\infty = 68.57 \\pm 1.55$ and $z_{0,\\mathrm{tgt}}^{\\max} = 1.39 \\pm 0.28 \\le R_\\infty^* = 111.01 \\pm 5.67$ in Panel~A ($5.61 \\le 260.72$ and $2.69 \\le 413.53$ in Panel~B), so both trajectories start strictly inside their absorbing balls on $10/10$ seeds.

\\textbf{5. Worked Per-Seed Arithmetic Verification (Seed 42 in Table~\\ref{tab:thm3_per_seed}):} For Seed 42 (Panel~A: $\\alpha_{\\mathrm{eff}}=0.35, \\gamma=0.15, \\epsilon=0.08, \\sqrt{\\lambda_{\\min}(P)}=0.58336$), $K_{\\mathrm{Young}} = 0.5150$, $\\nu_{\\mathrm{Young}} = 1.4987\\text{ s}^{-1}$, $e^{\\nu_{\\mathrm{Young}}h_k} = 1.6897$, and $\\frac{e^{\\nu_{\\mathrm{Young}}h_k}-1}{\\nu_{\\mathrm{Young}}} = 0.4602$. From Eq.~\\eqref{eq:init_p_bound} with $e_{q,\\mathrm{RMS}}=0.2193, e_{v,\\mathrm{RMS}}^{\\mathrm{Kin}}=0.7969$ ($e_{v,\\mathrm{RMS}}^{\\mathrm{NoKin}}=2.1561$), $\\bar{e}_{P,\\mathrm{RMS}}^{\\mathrm{Kin}} = \\sqrt{0.35(0.2193)^2 + 0.16(0.2193)(0.7969) + 0.7969^2} = 0.8245$ (vs.\\ $\\bar{e}_{P,\\mathrm{RMS}}^{\\mathrm{NoKin}} = 2.1774$). Dividing by $\\sqrt{\\lambda_{\\min}(P)} = 0.58336$ and adding $E_{\\mathrm{disc,RMS}} = 0.00008$ with $\\varepsilon_{\\mathrm{mod,RMS}} = 0.5768$ gives $\\bar{e}_{\\mathrm{lat,RMS}}^{\\mathrm{Kin}} = \\frac{1.6897(0.8245) + 0.4602(0.5768)}{0.58336} + 0.00008 = 2.8434$ units (vs.\\ $6.7623$ without $\\mathcal{L}_{\\mathrm{kin}}$). Multiplying by $L_{g,\\mathrm{spec}} = 106.01$ and adding $\\varepsilon_{\\mathrm{probe,RMS}} = 12.00$ yields the \\textbf{formal spectral bound $12.00 + 106.01(2.8434) = 313.43$ cycles} (vs.\\ $728.87$ without $\\mathcal{L}_{\\mathrm{kin}}$), while substituting $\\hat{L}_{g,\\mathrm{dir}} = 13.57$ yields the \\textbf{empirical directional diagnostic $12.00 + 13.57(2.8434) = 50.60$ cycles} (vs.\\ $103.80$ without $\\mathcal{L}_{\\mathrm{kin}}$ and $19.39$ empirical RMSE). \\hfill $\\blacksquare$

"""

tex = tex[:start_appc] + new_appc_block + tex[end_appc:]

with open('manuscript/main.tex', 'w') as f:
    f.write(tex)

print("Successfully updated manuscript/main.tex with all 5 mathematical fixes!")
