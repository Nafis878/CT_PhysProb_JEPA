#!/usr/bin/env python3
"""
Pre-flight verification of Gates G1 - G4 before confirmatory execution.
"""
import json
import hashlib
import subprocess
from pathlib import Path

def verify_gates():
    print("=" * 80)
    print("CHECKING PRE-FLIGHT CONFIRMATORY GATES G1 - G4")
    print("=" * 80)

    # Load Manifest
    with open('results/raw/CONFIRMATORY_MANIFEST.json', 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    # ---------------------------------------------------------
    # Gate 1: Remote tag resolution
    # ---------------------------------------------------------
    git_path = r"C:\Program Files\Git\cmd\git.exe"
    res = subprocess.run([git_path, 'ls-remote', manifest['git_remote_url']], capture_output=True, text=True)
    lines = [l for l in res.stdout.splitlines() if 'preregistration-v2.5' in l]
    print("G1 Check: Git ls-remote output for v2.5:")
    for l in lines:
        print("  ", l)
    g1_pass = False
    for l in lines:
        if manifest['preregistration_v25_commit'] in l or manifest['preregistration_v25_tag_object'] in l:
            g1_pass = True
    print(f"G1 Result: {'PASSED' if g1_pass else 'FAILED'}")
    assert g1_pass, "G1 failed!"

    # ---------------------------------------------------------
    # Gate 2: Frozen config hashes
    # ---------------------------------------------------------
    with open('configs/frozen_final_configuration_v2.json', 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    with open('configs/frozen_final_configuration_v2.json', 'rb') as f:
        cfg_file_sha256 = hashlib.sha256(f.read()).hexdigest()

    print(f"G2 Check: Frozen config object hash      = {cfg['config_hash']}")
    print(f"          Manifest frozen_config_sha256   = {manifest['frozen_config_sha256']}")
    print(f"          Frozen config file SHA-256      = {cfg_file_sha256}")
    print(f"          Manifest frozen_config_file_sha = {manifest['frozen_config_file_sha256']}")

    with open('configs/baseline_tuned_configurations.json', 'r', encoding='utf-8') as f:
        baseline_cfg = json.load(f)
    baseline_match = (baseline_cfg == manifest['baseline_tuned_configurations'])
    print(f"          Baseline config matches manifest = {baseline_match}")

    g2_pass = (
        cfg['config_hash'] == manifest['frozen_config_sha256'] and
        cfg_file_sha256 == manifest['frozen_config_file_sha256'] and
        baseline_match
    )
    print(f"G2 Result: {'PASSED' if g2_pass else 'FAILED'}")
    assert g2_pass, "G2 failed!"

    # ---------------------------------------------------------
    # Gate 3: Unit-ID lists & zero overlap
    # ---------------------------------------------------------
    train_units = set(manifest['official_test_training_pool_80'])
    calib_units = set(manifest['conformal_calibration_selection']['calibration_units_20'])
    test_units = set(manifest['official_test_units'])

    print(f"G3 Check:")
    print(f"  Train units ({len(train_units)}): {sorted(list(train_units))}")
    print(f"  Calibration units ({len(calib_units)}): {sorted(list(calib_units))}")
    print(f"  Official Test units ({len(test_units)}): {sorted(list(test_units))}")

    overlap_tr_ca = train_units.intersection(calib_units)

    print(f"  Overlap Train & Calib in FD003 training pool: {overlap_tr_ca}")
    print(f"  Train units count: {len(train_units)} (must be 80)")
    print(f"  Calib units count: {len(calib_units)} (must be 20)")
    print(f"  Test units count:  {len(test_units)} (must be 100)")

    assert len(train_units) == 80
    assert len(calib_units) == 20
    assert len(test_units) == 100
    assert len(overlap_tr_ca) == 0

    g3_pass = True
    print(f"G3 Result: PASSED")

    # ---------------------------------------------------------
    # Gate 4: No prior confirmatory evaluation touched test_FD003.txt / RUL_FD003.txt
    # ---------------------------------------------------------
    conf_raw = Path('results/raw/phase3_confirmatory_raw.csv')
    h4b_raw = Path('results/raw/phase3_h4b_ablation_raw.csv')
    print(f"G4 Check:")
    print(f"  phase3_confirmatory_raw.csv exists: {conf_raw.exists()}")
    print(f"  phase3_h4b_ablation_raw.csv exists: {h4b_raw.exists()}")
    if conf_raw.exists():
        print(f"  phase3_confirmatory_raw.csv size: {conf_raw.stat().st_size} bytes")
    if h4b_raw.exists():
        print(f"  phase3_h4b_ablation_raw.csv size: {h4b_raw.stat().st_size} bytes")

    g4_pass = (not conf_raw.exists() or conf_raw.stat().st_size == 0) and (not h4b_raw.exists() or h4b_raw.stat().st_size == 0)
    print(f"G4 Result: {'PASSED' if g4_pass else 'FAILED'}")
    assert g4_pass, "G4 failed!"

    print("=" * 80)
    print("ALL 4 PRE-FLIGHT GATES (G1, G2, G3, G4) PASSED STRICTLY.")
    print("=" * 80)

if __name__ == '__main__':
    verify_gates()
