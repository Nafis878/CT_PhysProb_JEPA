#!/usr/bin/env python3
import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Load existing extended sweep raw
df_orig = pd.read_csv(ROOT / "results" / "raw" / "extended_baseline_sweep_raw.csv")
df_ext = pd.read_csv(ROOT / "results" / "raw" / "baseline_edge_extensions_raw.csv")

# Combine
df_all = pd.concat([df_orig, df_ext], ignore_index=True)
df_all.to_csv(ROOT / "results" / "raw" / "extended_baseline_sweep_raw.csv", index=False)
print("Updated extended_baseline_sweep_raw.csv with all trials.")

# Load existing configurations
with open(ROOT / "configs" / "baseline_tuned_configurations.json") as f:
    best_tuned = json.load(f)

# Update GRU_D_Original
best_tuned['GRU_D_Original'] = {
    'model': 'GRU_D_Original',
    'best_param': 'hidden_dim',
    'best_size': 80,
    'best_lr': 0.0035,
    'is_edge': True,
    'val_rmse_mean': 19.046,
    'val_rmse_std': 0.455,
    'val_bias_mean': 8.58,
    'note': 'Extended grid from 64 to 80; verified improvement over 19.264'
}

# Update Plain_2ndOrder_NODE_Kin_Original
best_tuned['Plain_2ndOrder_NODE_Kin_Original'] = {
    'model': 'Plain_2ndOrder_NODE_Kin_Original',
    'best_param': 'latent_dim',
    'best_size': 24,
    'best_lr': 0.0075,
    'is_edge': True,
    'val_rmse_mean': 23.261,
    'val_rmse_std': 0.676,
    'val_bias_mean': 3.36,
    'note': 'Extended grid to lr=7.5e-3; verified improvement over 24.458'
}

# Add NoPhysics 5-seed validation row
best_tuned['CT_PhysProb_JEPA_NoPhysics_Frozen'] = {
    'model': 'CT_PhysProb_JEPA_NoPhysics',
    'status': 'Frozen_Matched_Control',
    'latent_dim': 16,
    'hidden_dim': 48,
    'lr': 0.0035,
    'val_rmse_mean': 18.374,
    'val_rmse_std': 0.417,
    'val_bias_mean': 4.38,
    'seed_rmses': [17.752178, 18.748138, 18.221968, 18.225655, 18.921253]
}

with open(ROOT / "configs" / "baseline_tuned_configurations.json", 'w') as f:
    json.dump(best_tuned, f, indent=2)

print("Updated configs/baseline_tuned_configurations.json successfully.")
