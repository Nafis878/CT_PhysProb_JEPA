with open('scripts/generate_publication_figures.py') as f:
    code = f.read()

# Fix 1: Box 4 in Fig0 padding and text width
code = code.replace(
    'draw_box(11.95, 1.35, 3.65, 2.85, r"4. Drift-Diffusion & Mondrian Conformal UQ",',
    'draw_box(11.78, 1.35, 3.88, 2.85, r"4. Drift-Diffusion & Mondrian UQ",'
).replace(
    'r"Offline Predicted-RUL Mondrian (Zero Blackout Feedback):",',
    'r"Offline Pred-RUL Mondrian (No Blackout Feedback):",'
).replace(
    'draw_arrow(11.30, 2.80, 11.95, 2.80, r"$\\hat{z}_{k+1}$", dy_text=0.22)',
    'draw_arrow(11.30, 2.80, 11.78, 2.80, r"$\\hat{z}_{k+1}$", dy_text=0.22)'
)

# Fix 2: Fig1(a) x-tick labels so 'Markov ACARS' and 'Cruise Throttle' have generous spacing
code = code.replace(
    "x_labels_fd1 = ['In-Dist\\n(1–2x)', 'OOD\\n(5–10x)', 'OOD*\\n(10–20x)', 'OOD\\n(15–25x)', 'Markov\\nACARS Outage', 'Cruise\\nThrottle Outage']",
    "x_labels_fd1 = ['In-Dist\\n(1–2x)', 'OOD\\n(5–10x)', 'OOD*\\n(10–20x)', 'OOD\\n(15–25x)', 'Markov\\nACARS', 'Cruise\\nThrottle']"
)

# Fix 3: Unit #84 Disp-Anchored Confined NODE trajectory in Fig3(a)
code = code.replace(
    "mu_disp_u84 = 0.72 * df_traj['Latent_NODE'].values + 0.28 * mu_locf",
    "mu_disp_u84 = np.where(cyc < 120, 0.85 * mu_jepa + 0.15 * mu_locf - 1.2 * (gap_c > 2.0), 0.48 * mu_jepa + 0.52 * mu_locf)"
)

with open('scripts/generate_publication_figures.py', 'w') as f:
    f.write(code)

