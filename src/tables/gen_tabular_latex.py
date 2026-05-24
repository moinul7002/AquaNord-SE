#!/usr/bin/env python3
"""Generate the full tabular-baseline LaTeX table from baseline_model_results.csv."""
import pandas as pd

df = pd.read_csv('data/outputs/baseline_model_results.csv')

tier_order  = ['static', 'met', 'met_mesan', 'published_alg', 'full']
model_order = ['StationMean', 'Ridge', 'RF', 'XGBoost', 'LightGBM', 'MLP', 'FT-Transformer']
tier_label  = {
    'static':        r'\texttt{static}',
    'met':           r'\texttt{met}',
    'met_mesan':     r'\texttt{met\_mesan}',
    'published_alg': r'\texttt{pub.alg.}',
    'full':          r'\texttt{full}',
}

df['feature_set'] = pd.Categorical(df['feature_set'], tier_order, ordered=True)
df['model']       = pd.Categorical(df['model'],       model_order, ordered=True)
df = df.sort_values(['model', 'feature_set'])

tgts = ['chla_ug_l', 'tp_ug_l', 'secchi_m']
best_r2   = {t: df[df.target == t]['r2'].max()   for t in tgts}
best_rmse = {t: df[df.target == t]['rmse'].min() for t in tgts}
best_mae  = {t: df[df.target == t]['mae'].min()  for t in tgts}

def fmt_r2(v):
    return (r'$-$' + f'{abs(v):.3f}') if v < 0 else f'{v:.3f}'

def fmt_rmse(v, tgt):
    return f'{v:.2f}' if tgt == 'secchi_m' else f'{v:.1f}'

def fmt_mae(v, tgt):
    return f'{v:.2f}' if tgt in ('chla_ug_l', 'secchi_m') else f'{v:.1f}'

def B(s, flag):
    return r'\textbf{' + s + '}' if flag else s

# Half the display-rounding unit per metric so values that print identically
# are always both bolded (avoids splits like 99.861 vs 99.856 → both "99.9").
TOL_R2        = 0.0005   # 3 decimal places
TOL_RMSE_TP   = 0.05     # 1 decimal place  (µg/L)
TOL_RMSE_CHLA = 0.05     # 1 decimal place  (µg/L)
TOL_RMSE_SEC  = 0.005    # 2 decimal places (m)
TOL_MAE_CHLA  = 0.005    # 2 decimal places
TOL_MAE_TP    = 0.05     # 1 decimal place
TOL_MAE_SEC   = 0.005    # 2 decimal places

model_refs = {
    'RF':             r'~\citep{breiman2001random}',
    'XGBoost':        r'~\citep{chen2016xgboost}',
    'LightGBM':       r'~\citep{ke2017lightgbm}',
    'FT-Transformer': r'~\citep{gorishniy2021revisiting}',
}

out = []
out.append(r'\begin{table*}[t]')
out.append(r'\centering')
caption = (
    r'\caption{Tabular baseline results across all five feature tiers. '
    r'GroupKFold($k{=}5$) spatial CV\@. '
    r'$n$: Chl-\textit{a}\,=\,49{,}300; TP\,=\,502{,}638; Secchi\,=\,16{,}163. '
    r'RMSE units: \textmu g\,L$^{-1}$ (Chl-\textit{a}, TP), m (Secchi). '
    r'$^{\star}$StationMean is feature-independent; one row shown. '
    r'\textbf{Bold}\,=\,best per metric column.}'
)
out.append(caption)
out.append(r'\label{tab:tabular_full}')
out.append(r'\resizebox{\textwidth}{!}{%')
out.append(r'\begin{tabular}{@{}ll rrr rrr rrr@{}}')
out.append(r'\toprule')
out.append(r'\textbf{Model} & \textbf{Tier}')
out.append(r'  & \multicolumn{3}{c}{\textbf{Chl-\textit{a}}}')
out.append(r'  & \multicolumn{3}{c}{\textbf{TP}}')
header_secchi = r'  & \multicolumn{3}{c}{\textbf{Secchi}} \\'
out.append(header_secchi)
out.append(r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}')
subheader = r'  & & $R^{2}$ & RMSE & MAE & $R^{2}$ & RMSE & MAE & $R^{2}$ & RMSE & MAE \\'
out.append(subheader)
out.append(r'\midrule')

prev_model = None
seen = set()
for _, row in df.iterrows():
    m = str(row['model'])
    t = str(row['feature_set'])
    if (m, t) in seen:
        continue
    seen.add((m, t))

    if m == 'StationMean' and t != 'static':
        continue

    if prev_model is not None and m != prev_model:
        out.append(r'\midrule')

    is_first = (prev_model != m)

    if m == 'StationMean':
        model_col = r'\textit{StationMean}$^{\star}$'
        tier_col  = r'\textit{---}'
    else:
        cite = model_refs.get(m, '')
        model_col = (r'\multirow{5}{*}{' + m + cite + '}') if is_first else ''
        tier_col  = tier_label[t]

    cells = []
    for tgt in tgts:
        sub = df[(df.model == m) & (df.feature_set == t) & (df.target == tgt)]
        r2, rmse, mae = sub['r2'].values[0], sub['rmse'].values[0], sub['mae'].values[0]

        tol_r2   = TOL_R2
        tol_rmse = TOL_RMSE_SEC  if tgt == 'secchi_m' else (TOL_RMSE_TP if tgt == 'tp_ug_l' else TOL_RMSE_CHLA)
        tol_mae  = TOL_MAE_SEC   if tgt == 'secchi_m' else (TOL_MAE_TP  if tgt == 'tp_ug_l' else TOL_MAE_CHLA)
        r2_s   = B(fmt_r2(r2),           abs(r2   - best_r2[tgt])   < tol_r2)
        rmse_s = B(fmt_rmse(rmse, tgt),  abs(rmse - best_rmse[tgt]) < tol_rmse)
        mae_s  = B(fmt_mae(mae, tgt),    abs(mae  - best_mae[tgt])  < tol_mae)
        cells += [r2_s, rmse_s, mae_s]

    rc = r'\rowcolor{gray!8}' if m == 'LightGBM' else ''
    line = rc + model_col + ' & ' + tier_col + ' & ' + ' & '.join(cells) + r' \\'
    out.append(line)
    prev_model = m

out.append(r'\bottomrule')
out.append(r'\end{tabular}%')
out.append(r'}')
out.append(r'\end{table*}')

result = '\n'.join(out)
print(result)

with open('Reports/tab_tabular_full.tex', 'w', encoding='utf-8') as f:
    f.write(result)
print('\n% Saved to Reports/tab_tabular_full.tex', flush=True)
