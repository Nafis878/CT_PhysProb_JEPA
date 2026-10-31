with open('manuscript/main.tex') as f:
    tex = f.read()

tex = tex.replace("(`Width@95%` and `Vac@95%`)", r"(\texttt{Width@95\%} and \texttt{Vac@95\%})")

old_eq30 = r"""\begin{equation}
\frac{d}{dt}W(e) \le -\big(\epsilon(\alpha_{\mathrm{eff}} - L_{\mathrm{nl}})\big)\|\delta q\|_2^2 - (\gamma - \epsilon)\|\delta p\|_2^2 - \epsilon\gamma \delta q^T \delta p + L_{\mathrm{nl}}|\delta p^T \delta q| + \|P^{1/2}e\|_2 \frac{\|r_{\mathrm{model}}\|_2}{\sqrt{\lambda_{\min}(P)}}.
\end{equation}"""

new_eq30 = r"""\begin{align}
\frac{d}{dt}W(e) &\le -\epsilon(\alpha_{\mathrm{eff}} - L_{\mathrm{nl}})\|\delta q\|_2^2 - (\gamma - \epsilon)\|\delta p\|_2^2 \nonumber \\
&\quad + (\epsilon\gamma + L_{\mathrm{nl}})|\delta q^T \delta p| + \|e\|_P \frac{\|r_{\mathrm{model}}\|_2}{\sqrt{\lambda_{\min}(P)}}.
\end{align}"""

tex = tex.replace(old_eq30, new_eq30)

with open('manuscript/main.tex', 'w') as f:
    f.write(tex)
