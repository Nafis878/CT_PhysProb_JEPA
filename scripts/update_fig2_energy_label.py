with open('scripts/generate_publication_figures.py') as f:
    code = f.read()

code = code.replace(
    'label=f"{v_lbl} (Drift {edrift:.1f}%)"',
    'label=f"{v_lbl} (Bal. Err {edrift:.1f}%)"'
).replace(
    "ax.set_title('(c) Multi-Step Rollout & Energy Drift', fontweight='bold')",
    "ax.set_title('(c) Multi-Step Rollout & Energy-Balance Error', fontweight='bold')"
)

with open('scripts/generate_publication_figures.py', 'w') as f:
    f.write(code)
