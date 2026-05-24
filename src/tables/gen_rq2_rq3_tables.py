#!/usr/bin/env python3
"""Generate LaTeX tables for §4.2 (RQ2) and §4.3/4.4 (RQ3)."""
import pandas as pd
import numpy as np
from pathlib import Path

OUT = Path('Reports')

roll  = pd.read_csv('data/outputs/baseline_rolling_origin.csv')
troph = pd.read_csv('data/outputs/baseline_trophic_results.csv')
quant = pd.read_csv('data/outputs/baseline_quantile_coverage.csv')
sat   = pd.read_csv('data/outputs/table4_satellite_wq_benchmark_stratified.csv')
ac    = pd.read_csv('data/outputs/baseline_ac_failure_results.csv')
ct    = pd.read_csv('data/outputs/baseline_conformal_results.csv')
cm    = pd.read_csv('data/outputs/multimodal_conformal_results.csv')

def fr2(v):
    if not np.isfinite(v): return '---'
    return (r'$-$' + f'{abs(v):.3f}') if v < 0 else f'{v:.3f}'

def frmse(v, tgt='chla'):
    if not np.isfinite(v): return '---'
    return f'{v:.2f}' if tgt == 'secchi_m' else f'{v:.1f}'

def B(s): return r'\textbf{' + s + '}'


# ════════════════════════════════════════════════════════════════════════════
# TABLE 1 — Rolling-origin temporal CV
# ════════════════════════════════════════════════════════════════════════════
tgt_tex   = {'chla_ug_l': r'Chl-\textit{a}', 'tp_ug_l': 'TP', 'secchi_m': 'Secchi'}
tgt_order = ['chla_ug_l', 'tp_ug_l', 'secchi_m']
periods   = ['2011-15', '2016-20', '2021-25']

roll_m = roll[roll.model.isin(['StationMean', 'LightGBM'])]

L = []
L.append(r'\begin{table}[ht]')
L.append(r'\centering')
L.append(
    r'\caption{Rolling-origin temporal cross-validation — LightGBM vs.\ StationMean '
    r'(\texttt{full} feature tier). Ridge excluded: multicollinearity under restricted '
    r'training data causes RMSE\,$\to\infty$ in the 2016--20 window. '
    r'RMSE units: \textmu g\,L$^{-1}$ (Chl-\textit{a}, TP), m (Secchi).}'
)
L.append(r'\label{tab:rolling}')
L.append(r'\resizebox{\columnwidth}{!}{%')
L.append(r'\begin{tabular}{@{}ll rr rr rr@{}}')
L.append(r'\toprule')
L.append(r'\textbf{Target} & \textbf{Model}')
L.append(r'  & \multicolumn{2}{c}{\textbf{2011--15}}')
L.append(r'  & \multicolumn{2}{c}{\textbf{2016--20}}')
header_end = r'  & \multicolumn{2}{c}{\textbf{2021--25}} \\'
L.append(header_end)
L.append(r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}')
sub_hdr = r'  & & $R^{2}$ & RMSE & $R^{2}$ & RMSE & $R^{2}$ & RMSE \\'
L.append(sub_hdr)
L.append(r'\midrule')

for ti, tgt in enumerate(tgt_order):
    sub = roll_m[roll_m.target == tgt]
    for mi, mod in enumerate(['StationMean', 'LightGBM']):
        if ti > 0 and mi == 0:
            L.append(r'\midrule')
        mr = (r'\multirow{2}{*}{' + tgt_tex[tgt] + '}') if mi == 0 else ''
        row_s = sub[sub.model == mod]
        bold = (mod == 'LightGBM')
        cells = []
        for p in periods:
            r = row_s[row_s.test_period == p]
            if r.empty:
                cells += ['---', '---']
                continue
            rv, rm = r['r2'].values[0], r['rmse'].values[0]
            r2s   = fr2(rv)
            rmses = frmse(rm, tgt)
            if bold:
                r2s   = B(r2s)
                rmses = B(rmses)
            cells += [r2s, rmses]
        mname = B('LightGBM') if bold else mod
        L.append(mr + ' & ' + mname + ' & ' + ' & '.join(cells) + r' \\')

L.append(r'\bottomrule')
L.append(r'\end{tabular}%')
L.append(r'}')
L.append(r'\end{table}')
tab_rolling = '\n'.join(L)


# ════════════════════════════════════════════════════════════════════════════
# TABLE 2 — Trophic stratification + calibration cascade (Chl-a, LightGBM)
# ════════════════════════════════════════════════════════════════════════════
tc_order = ['oligotrophic', 'mesotrophic', 'eutrophic', 'hypereutrophic']
tsi_map  = {'oligotrophic': r'${<}40$', 'mesotrophic': '40--50',
            'eutrophic': '50--60', 'hypereutrophic': r'${>}60$'}

troph_c = (troph[(troph.target == 'chla_ug_l') & (troph.model == 'LightGBM')]
           .set_index('trophic_class'))
quant_c = (quant[(quant.target == 'chla_ug_l') & (quant.nominal_coverage == 0.95)]
           .set_index('trophic_class'))

L = []
L.append(r'\begin{table}[ht]')
L.append(r'\centering')
L.append(
    r'\caption{Trophic stratification of LightGBM Chl-\textit{a} performance and '
    r'conformal calibration quality (\texttt{full} tier, 95\,\% nominal coverage). '
    r'Trophic class assigned from station-median TP via Carlson TSI\@. '
    r'RMSE in \textmu g\,L$^{-1}$; Winkler score in \textmu g\,L$^{-1}$ '
    r'(proper scoring rule~\citep{gneiting2007strictly}).}'
)
L.append(r'\label{tab:trophic}')
L.append(r'\resizebox{\columnwidth}{!}{%')
L.append(r'\begin{tabular}{@{}l r r r r r r@{}}')
L.append(r'\toprule')
L.append(r'\textbf{Trophic class} & \textbf{TSI} & $n$ & $R^{2}$ & '
         r'\textbf{RMSE} & \textbf{ECE\,@\,95\%} & \textbf{Winkler\,@\,95\%} \\')
L.append(r'\midrule')

for tc in tc_order:
    if tc not in troph_c.index:
        continue
    tr = troph_c.loc[tc]
    qr = quant_c.loc[tc] if tc in quant_c.index else None
    n    = f'{int(tr["n"]):,}'.replace(',', r'{,}')
    r2s  = fr2(tr['r2'])
    rmse = f'{tr["rmse"]:.1f}'
    ece  = f'{qr["ece"]:.3f}' if qr is not None else '---'
    wink = f'{qr["winkler_score"]:.1f}' if qr is not None else '---'
    if tc == 'hypereutrophic':
        ece  = B(ece)
        wink = B(wink)
        r2s  = B(r2s)
    tsi  = tsi_map[tc]
    name = tc.capitalize()
    L.append(name + ' & ' + tsi + ' & ' + n + ' & ' +
             r2s + ' & ' + rmse + ' & ' + ece + ' & ' + wink + r' \\')

L.append(r'\bottomrule')
L.append(r'\end{tabular}%')
L.append(r'}')
L.append(r'\end{table}')
tab_trophic = '\n'.join(L)


# ════════════════════════════════════════════════════════════════════════════
# TABLE 3 — Satellite–in-situ matchup
# ════════════════════════════════════════════════════════════════════════════
# Select key rows
rows_wanted = [
    ('MODIS',   'chla',     'ac_ok'),
    ('MODIS',   'chla',     'ac_failure'),
    ('MODIS',   'secchi',   'ac_ok'),
    ('MODIS',   'secchi',   'ac_failure'),
    ('Landsat', 'chla',     'ac_ok'),
    ('Landsat', 'chla',     'ac_failure'),
    ('S2',      'chla_ndci','ac_ok'),
    ('S2',      'secchi',   'ac_ok'),
]

var_tex = {'chla': r'Chl-\textit{a} (µg\,L$^{-1}$)',
           'chla_ndci': r'Chl-\textit{a} NDCI (µg\,L$^{-1}$)',
           'secchi': 'Secchi (m)',
           'turbidity': 'Turbidity'}
ac_tex = {'ac_ok': 'AC OK', 'ac_failure': r'\textbf{AC failure}', 'all': 'All'}

L = []
L.append(r'\begin{table}[ht]')
L.append(r'\centering')
L.append(
    r'\caption{Satellite--in-situ matchup stratified by atmospheric-correction (AC) '
    r'outcome ($\pm$3-day window). AC failure defined by '
    r'\texttt{rw\_blue\_negative\_flag}\,=\,1 (bias A1). '
    r'S2 has no per-pixel AC flag; a single row is shown.}'
)
L.append(r'\label{tab:matchup}')
L.append(r'\resizebox{\columnwidth}{!}{%')
L.append(r'\begin{tabular}{@{}l l l r r r r@{}}')
L.append(r'\toprule')
L.append(r'\textbf{Sensor} & \textbf{Variable} & \textbf{AC} & '
         r'\textbf{RMSE} & \textbf{MBE} & \textbf{Pearson\,$r$} & $n$ \\')
L.append(r'\midrule')

prev_sensor = None
for sensor, var, acf in rows_wanted:
    r = sat[(sat.sensor == sensor) & (sat.variable == var) & (sat.ac_flag == acf)]
    if r.empty:
        continue
    row = r.iloc[0]
    rmse = f'{row["rmse"]:.1f}' if var != 'secchi' else f'{row["rmse"]:.2f}'
    mbe  = f'{row["mbe"]:+.1f}' if var != 'secchi' else f'{row["mbe"]:+.2f}'
    pr   = f'{row["pearson_r"]:+.3f}'
    n    = f'{int(row["n"]):,}'.replace(',', r'{,}')

    if sensor != prev_sensor and prev_sensor is not None:
        L.append(r'\midrule')
    sc = sensor if sensor != prev_sensor else ''
    prev_sensor = sensor

    ac_s = ac_tex.get(acf, acf)
    var_s = var_tex.get(var, var)

    if acf == 'ac_failure':
        rmse = B(rmse); pr = B(pr)

    L.append(sc + ' & ' + var_s + ' & ' + ac_s + ' & ' +
             rmse + ' & ' + mbe + ' & ' + pr + ' & ' + n + r' \\')

L.append(r'\bottomrule')
L.append(r'\end{tabular}%')
L.append(r'}')
L.append(r'\end{table}')
tab_matchup = '\n'.join(L)


# ════════════════════════════════════════════════════════════════════════════
# TABLE 4 — Split conformal PICP / ECE at 90 % nominal
# ════════════════════════════════════════════════════════════════════════════
# Tabular models from CSV; multimodal Secchi from 04_results.md (not in CSV)
multi_secchi = {
    'MOSAIKS':           (0.866, 0.034),
    'ERA5-LSTM-w30':     (0.952, 0.052),
    'CrossModalNet':     (0.904, 0.004),
    'TemporalTrans-ERA5':(0.901, 0.002),
}
multi_tp_picp = {   # TemporalTrans tp not in CSV → use MD value
    'TemporalTrans-ERA5': (0.905, 0.005),
}

model_order = ['Ridge', 'RF', 'XGBoost', 'LightGBM', 'MLP', 'FT-Transformer',
               'MOSAIKS', 'ERA5-LSTM-w30', 'CrossModalNet', 'TemporalTrans-ERA5']
model_cite = {
    'RF':              r'~\citep{breiman2001random}',
    'XGBoost':         r'~\citep{chen2016xgboost}',
    'LightGBM':        r'~\citep{ke2017lightgbm}',
    'FT-Transformer':  r'~\citep{gorishniy2021revisiting}',
    'MOSAIKS':         r'~\citep{rolf2021generalizable}',
    'ERA5-LSTM-w30':   r'~\citep{kratzert2018rainfall}',
    'TemporalTrans-ERA5': r'~\citep{tseng2023lightweight}',
}

def get_conf(model, tgt, nominal=0.90):
    # tabular first
    r = ct[(ct.model == model) & (ct.target == tgt) &
           (ct.nominal_coverage == nominal)]
    if not r.empty:
        return r['picp'].values[0], r['ece'].values[0]
    # multimodal
    r = cm[(cm.model == model) & (cm.target == tgt) &
           (cm.nominal_coverage == nominal)]
    if not r.empty:
        return r['picp'].values[0], r['ece'].values[0]
    # fallback for TemporalTrans TP and all multimodal secchi
    if tgt == 'secchi_m' and model in multi_secchi:
        return multi_secchi[model]
    if tgt == 'tp_ug_l' and model in multi_tp_picp:
        return multi_tp_picp[model]
    return None, None

L = []
L.append(r'\begin{table}[ht]')
L.append(r'\centering')
L.append(
    r'\caption{Split conformal prediction interval coverage (PICP) and expected '
    r'calibration error (ECE) at 90\,\% nominal coverage. '
    r'GroupKFold($k{=}5$) spatial CV; calibration split 80\,/\,20\,\% per fold. '
    r'Nominal = 0.90 dashed reference. '
    r'$^{*}$ERA5-LSTM and TemporalTrans-ERA5 operate on 31.4\,\% of stations.}'
)
L.append(r'\label{tab:conformal}')
L.append(r'\resizebox{\columnwidth}{!}{%')
L.append(r'\begin{tabular}{@{}l rr rr rr@{}}')
L.append(r'\toprule')
L.append(r'\textbf{Model}')
L.append(r'  & \multicolumn{2}{c}{\textbf{Chl-\textit{a}}}')
L.append(r'  & \multicolumn{2}{c}{\textbf{TP}}')
conf_hdr = r'  & \multicolumn{2}{c}{\textbf{Secchi}} \\'
L.append(conf_hdr)
L.append(r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}')
sub_hdr2 = r'  & PICP & ECE & PICP & ECE & PICP & ECE \\'
L.append(sub_hdr2)
L.append(r'\midrule')

prev_family = None
tabular_set = {'Ridge', 'RF', 'XGBoost', 'LightGBM', 'MLP', 'FT-Transformer'}

for mod in model_order:
    family = 'tabular' if mod in tabular_set else 'multimodal'
    if prev_family == 'tabular' and family == 'multimodal':
        L.append(r'\midrule')
    prev_family = family

    cite  = model_cite.get(mod, '')
    star  = r'$^{*}$' if mod in ('ERA5-LSTM-w30', 'TemporalTrans-ERA5') else ''
    mname = mod + star + cite
    if mod == 'LightGBM':
        mname = r'\rowcolor{gray!8}' + r'\textbf{' + mod + r'}' + cite

    cells = []
    for tgt in ['chla_ug_l', 'tp_ug_l', 'secchi_m']:
        picp, ece = get_conf(mod, tgt)
        if picp is None:
            cells += ['---', '---']
        else:
            cells += [f'{picp:.3f}', f'{ece:.3f}']
    L.append(mname + ' & ' + ' & '.join(cells) + r' \\')

L.append(r'\bottomrule')
L.append(r'\end{tabular}%')
L.append(r'}')
L.append(r'\end{table}')
tab_conformal = '\n'.join(L)


# ════════════════════════════════════════════════════════════════════════════
# AC failure propagation — small inline table
# ════════════════════════════════════════════════════════════════════════════
ac_c = ac[ac.target == 'chla_ug_l'].sort_values(['feature_set', 'ac_flag'])

L = []
L.append(r'\begin{table}[ht]')
L.append(r'\centering')
L.append(
    r'\caption{AC failure propagation into LightGBM Chl-\textit{a} predictions. '
    r'Under \texttt{published\_alg}, AC-failure observations yield counterintuitively '
    r'higher $R^{2}$ because corrupted satellite values encode a lake-type signal '
    r'correlated with Chl-\textit{a}. The \texttt{full} tier absorbs this artefact '
    r'through static catchment attribute redundancy.}'
)
L.append(r'\label{tab:acprop}')
L.append(r'\begin{tabular}{@{}l l r r@{}}')
L.append(r'\toprule')
L.append(r'\textbf{Tier} & \textbf{AC condition} & $R^{2}$ & \textbf{RMSE} \\')
L.append(r'\midrule')
tier_tex = {'published_alg': r'\texttt{pub.alg.}', 'full': r'\texttt{full}'}
for _, row in ac_c.iterrows():
    ac_s = 'AC OK' if row['ac_flag'] == 'ac_ok' else r'\textbf{AC failure}'
    r2s  = fr2(row['r2'])
    rm   = f'{row["rmse"]:.1f}'
    if row['ac_flag'] == 'ac_failure':
        r2s = B(r2s); rm = B(rm)
    L.append(tier_tex.get(row['feature_set'], row['feature_set']) +
             ' & ' + ac_s + ' & ' + r2s + ' & ' + rm + r' \\')
L.append(r'\bottomrule')
L.append(r'\end{tabular}')
L.append(r'\end{table}')
tab_acprop = '\n'.join(L)


# ════════════════════════════════════════════════════════════════════════════
# Write output
# ════════════════════════════════════════════════════════════════════════════
all_tables = {
    'tab_rolling.tex':   tab_rolling,
    'tab_trophic.tex':   tab_trophic,
    'tab_matchup.tex':   tab_matchup,
    'tab_acprop.tex':    tab_acprop,
    'tab_conformal.tex': tab_conformal,
}
for fname, content in all_tables.items():
    path = OUT / fname
    path.write_text(content, encoding='utf-8')
    print(f'Saved {path}')

print('\n\n=== tab_rolling ===\n', tab_rolling)
print('\n\n=== tab_trophic ===\n', tab_trophic)
print('\n\n=== tab_matchup ===\n', tab_matchup)
print('\n\n=== tab_acprop ===\n', tab_acprop)
print('\n\n=== tab_conformal ===\n', tab_conformal)
