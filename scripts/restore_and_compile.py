import re

with open('manuscript/main.tex') as f:
    tex = f.read()

# Undo the accidental \texttt{...} wrapping that spanned across lines or equations
# Specifically, any \_ inside \ref{...}, \eqref{...}, \label{...}, or math mode ($...$, \begin{equation}...\end{equation}, \begin{align}...\end{align})
# Wait: why did repl_bt match across lines? Because of (`Raw Late Width`, `Raw Late Vac`)!
# Let's check how many times \texttt{ was inserted by repl_bt:
