with open('manuscript/main.tex') as f:
    tex = f.read()

# 1. Wrap Eq. (15) across 2 lines
old_eq15 = r"""\begin{equation}
\Delta\mathcal{H}_{\mathrm{exact}}(h_k) \triangleq \mathcal{H}\big(q(t_k+h_k), p(t_k+h_k)\big) - \mathcal{H}(q_k, p_k) + \gamma \int_{t_k}^{t_k+h_k} \|p(\tau)\|_2^2\,d\tau = 0.
\label{eq:work_energy}
\end{equation}"""

new_eq15 = r"""\begin{align}
\Delta\mathcal{H}_{\mathrm{exact}}(h_k) &\triangleq \mathcal{H}\big(q(t_k+h_k), p(t_k+h_k)\big) - \mathcal{H}(q_k, p_k) \nonumber \\
&\quad + \gamma \int_{t_k}^{t_k+h_k} \|p(\tau)\|_2^2\,d\tau = 0.
\label{eq:work_energy}
\end{align}"""

# 2. Wrap Case 2 of Theorem 3 cleanly
old_case2 = r"""yields the finite-horizon bound $\varepsilon_{\mathrm{probe}} + L_g \min\!\big\{2 R_\infty, \; \kappa_P e^{\nu_{\mathrm{net}} h_k}\sqrt{e_{q,k}^2 + h_k^2 e_{v,k}^2} + \frac{e^{\nu_{\mathrm{net}} h_k} - 1}{\nu_{\mathrm{net}}}(\varepsilon_{\mathrm{model}} + C_{\mathrm{symp}}\delta t^2)\big\}$."""

new_case2 = r"""yields the finite-horizon bound:
    \begin{align}
    \big|\hat{\mu}_{k+1} - y^*(t_{k+1})\big| &\le \varepsilon_{\mathrm{probe}} + L_g \min\!\Bigg\{2 R_\infty, \; \kappa_P e^{\nu_{\mathrm{net}} h_k}\sqrt{e_{q,k}^2 + h_k^2 e_{v,k}^2} \nonumber \\
    &\quad + \frac{e^{\nu_{\mathrm{net}} h_k} - 1}{\nu_{\mathrm{net}}}\big(\varepsilon_{\mathrm{model}} + C_{\mathrm{symp}}\delta t^2\big)\Bigg\}.
    \end{align}"""

# 3. Wrap Eq. (26) J_CBM across 2 lines
old_eq26 = r"""    \begin{equation}
    \mathcal{J}_{\mathrm{CBM}} = \frac{1}{N}\sum_{n=1}^{N} \Big( C_{\mathrm{fail}}\mathbb{I}_{\mathrm{miss},n} + C_{\mathrm{FA}}\mathbb{I}_{\mathrm{FA},n} + C_{\mathrm{waste}}\max(0, y_{\mathrm{last},n} - 15)\mathbb{I}_{\mathrm{trig},n} \Big).
    \label{eq:cbm_cost_fn}
    \end{equation}"""

new_eq26 = r"""    \begin{align}
    \mathcal{J}_{\mathrm{CBM}} &= \frac{1}{N}\sum_{n=1}^{N} \Big( C_{\mathrm{fail}}\mathbb{I}_{\mathrm{miss},n} + C_{\mathrm{FA}}\mathbb{I}_{\mathrm{FA},n} \nonumber \\
    &\quad + C_{\mathrm{waste}}\max(0, y_{\mathrm{last},n} - 15)\mathbb{I}_{\mathrm{trig},n} \Big).
    \label{eq:cbm_cost_fn}
    \end{align}"""

assert old_eq15 in tex, "old_eq15 not found"
assert old_case2 in tex, "old_case2 not found"
assert old_eq26 in tex, "old_eq26 not found"

tex = tex.replace(old_eq15, new_eq15).replace(old_case2, new_case2).replace(old_eq26, new_eq26)
with open('manuscript/main.tex', 'w') as f:
    f.write(tex)
