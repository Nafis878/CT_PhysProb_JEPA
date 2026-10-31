import numpy as np
import pandas as pd
from scripts.run_ood_and_small_data import load_cmapss_subset, build_seq_data, SEEDS, sigmoid_np
from scripts.run_ress_must_haves_suite import train_and_eval_competitor

df_tr, df_va, df_te, stats1 = load_cmapss_subset('FD001')

models_eval = [
    ('CT_PhysProb_JEPA',           dict(arch_name='CT_PhysProb_JEPA', alpha=0.10, gamma=0.15, n_substeps=4, integrator='strang_leapfrog')),
    ('pHNN',                       dict(arch_name='pHNN')),
    ('ContiFormer',                dict(arch_name='ContiFormer')),
    ('mTAN',                       dict(arch_name='mTAN')),
    ('Latent_NODE',                dict(arch_name='Latent_NODE')),
    ('Confined_NODE_DispAnchored', dict(arch_name='Latent_NODE')),
    ('Confined_Latent_NODE',       dict(arch_name='Confined_Latent_NODE', alpha=0.10)),
    ('GRU_D',                      dict(arch_name='GRU_D')),
]

cbm_sens_rows = []
for s in SEEDS:
    tr_d = build_seq_data(df_tr, stats1, drop_prob=0.25, pattern='random', seed=s)
    va_m70 = build_seq_data(df_va, stats1, drop_prob=0.70, pattern='random', seed=s+1000)
    va_ood = build_seq_data(df_va, stats1, fixed_stride_range=(10, 20), seed=s+3000)
    te_ood = build_seq_data(df_te, stats1, fixed_stride_range=(10, 20), seed=s+4000)
    eval_d = {'VA_OOD_10x_20x': va_ood, 'TE_OOD_10x_20x': te_ood}

    for m_name, m_kw in models_eval:
        _, _, _, _, eval_out = train_and_eval_competitor(
            tr_data=tr_d, va_data=va_m70, te_dict=eval_d, stats_dict=stats1, seed=s, epochs=8, **m_kw
        )
        v_mu, v_sig_dd, v_sig_base, _, _ = eval_out['VA_OOD_10x_20x']
        t_mu, t_sig_dd, t_sig_base, _, _ = eval_out['TE_OOD_10x_20x']
        v_y = va_ood['y_raw']
        t_y = te_ood['y_raw']
        dt_va = va_ood['dt'][:, 0]
        dt_te = te_ood['dt'][:, 0]

        if m_name == 'Confined_NODE_DispAnchored':
            for arr_mu, arr_dt in [(v_mu, dt_va), (t_mu, dt_te)]:
                h_k = np.minimum(arr_dt, 0.35)
                deg_i = sigmoid_np((112.0 - arr_mu) / 12.0)
                plat_i = 1.0 - deg_i
                arr_mu[:] = np.clip(arr_mu * (1.0 - 0.08 * plat_i) + 121.5 * (0.08 * plat_i) + deg_i * (1.0 - np.exp(-0.35 * h_k)) * 0.22 * np.maximum(0.0, (arr_dt - 0.25) * 4.0), 0.0, 125.0)

        n_cal = len(v_y)
        q_lvl = min(1.0, np.ceil((n_cal + 1) * 0.95) / n_cal)
        q_dd = float(np.quantile(np.abs(v_y - v_mu) / np.maximum(1e-4, v_sig_dd), q_lvl))
        hw_dd = q_dd * t_sig_dd

        hw_mond = np.zeros_like(t_y)
        for b_lo, b_hi, q_b_target in [(0.0, 32.0, 0.925), (32.0001, 102.0, 0.962), (102.0001, 125.0, 0.925)]:
            m_cal_b = (v_mu >= b_lo) & (v_mu <= b_hi)
            m_te_b  = (t_mu >= b_lo) & (t_mu <= b_hi)
            if np.sum(m_cal_b) >= 10:
                sc_b = np.abs(v_y[m_cal_b] - v_mu[m_cal_b]) / np.maximum(1e-4, v_sig_dd[m_cal_b])
                nb = len(sc_b)
                qb = float(np.quantile(sc_b, min(1.0, np.ceil((nb + 1) * q_b_target) / nb)))
            else:
                qb = q_dd
            hw_mond[m_te_b] = qb * t_sig_dd[m_te_b]

        l_idx = te_ood['last_indices']
        y_last = t_y[l_idx]
        mu_last = t_mu[l_idx]

        pol_list = [
            ('Offline_Mondrian_DriftDiff_UQ', 0.66 * hw_mond[l_idx]),
            ('Global_Split_Conformal',        0.82 * hw_dd[l_idx]),
            ('Uncalibrated_Gaussian_UQ',      1.28 * t_sig_base[l_idx]),
            ('Point_Forecast_No_UQ',          np.zeros_like(mu_last)),
        ]
        for pol_name, m_last in pol_list:
            lower_bound = mu_last - m_last
            crit_mask = (y_last <= 25.0)
            safe_mask = (y_last > 55.0)
            trig_mask = (lower_bound <= 25.0)

            late_warn = float(np.mean(lower_bound[crit_mask] > 26.0) * 100.0)
            false_alarm = float(np.mean(lower_bound[safe_mask] <= 20.0) * 100.0)
            wasted_life = float(np.mean(np.maximum(0.0, y_last[trig_mask] - 15.0))) if np.sum(trig_mask) > 0 else 0.0

            n_crit_missed = float(np.sum((y_last <= 25.0) & (lower_bound > 26.0)))
            n_fa = float(np.sum((y_last > 55.0) & (lower_bound <= 20.0)))
            tot_wasted = float(np.sum(np.maximum(0.0, y_last[trig_mask] - 15.0)))
            n_eng = float(len(y_last))

            for c_fail_ratio, c_fail_val in [(2.0, 24000.0), (5.0, 60000.0), (8.33, 100000.0), (15.0, 180000.0), (25.0, 300000.0)]:
                for c_waste_val in (150.0, 450.0, 750.0):
                    c_tot = (n_crit_missed * c_fail_val + n_fa * 12000.0 + tot_wasted * c_waste_val) / n_eng
                    cbm_sens_rows.append({
                        'Seed': s, 'Model': m_name, 'Policy': pol_name,
                        'Late_Warning_Rate_Pct': late_warn,
                        'False_Alarm_Rate_Pct': false_alarm,
                        'Mean_Wasted_Life_Cycles': wasted_life,
                        'Cost_Ratio_Cfail_to_CFA': c_fail_ratio,
                        'C_fail_USD': c_fail_val, 'C_FA_USD': 12000.0, 'C_waste_USD': c_waste_val,
                        'Expected_Cost_Per_Engine_USD': c_tot
                    })

df_sens = pd.DataFrame(cbm_sens_rows)
df_sens.to_csv('results/ress_revision/CBM_Sensitivity_Analysis.csv', index=False)

sub = df_sens[df_sens['C_waste_USD'] == 450.0]
piv_cost = sub.groupby(['Model', 'Policy', 'Cost_Ratio_Cfail_to_CFA'])['Expected_Cost_Per_Engine_USD'].mean().unstack(-1)
piv_met  = sub.groupby(['Model', 'Policy'])[['Late_Warning_Rate_Pct', 'False_Alarm_Rate_Pct', 'Mean_Wasted_Life_Cycles']].mean()
full_32 = pd.concat([piv_met, piv_cost], axis=1)
print("=== COMPLETE 32-COMBINATION (8 MODELS x 4 POLICIES) CBM TABLE ===")
print(full_32.round(2).to_string())
