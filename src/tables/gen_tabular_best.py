#!/usr/bin/env python3
"""Generate best-tier-per-model tabular baseline LaTeX table."""
import pandas as pd

df = pd.read_csv('data/outputs/baseline_model_results.csv')

tier_order  = ['static','met','met_mesan','published_alg','full']
model_order = ['StationMean','Ridge','RF','XGBoost','LightGBM','MLP','FT-Transformer']
tier_tex    = {
    'static':        r'\texttt{static}',
    'met':           r'\texttt{met}',
    'met_mesan':     r'\texttt{met\_mesan}',
    'published_alg': r'\texttt{pub.alg.}',
    'full':          r'\texttt{full}',
}

tgts = ['chla_ug_l', 'tp_ug_l', 'secchi_m']

df['feature_set'] = pd.Categorical(df['feature_set'], tier_order, ordered=True)
df['model']       = pd.Categorical(df['model'], model_order, ordered=True)
df = df.sort_values(['model', 'feature_set'])

# Best tier per model = tier that maximises sum of R² across all three targets
r2_wide = (df.pivot_table(index=['model', 'feature_set'], columns='target',
                          values='r2', observed=True)
             .reset_index())
r2_wide['sum_r2'] = r2_wide[tgts].sum(axis=1)
r2_wide['model']  = pd.Categorical(r2_wide['model'],  model_order, ordered=True)
r2_wide['feature_set'] = pd.Categorical(r2_wide['feature_set'], tier_order, ordered=True)
best_tier = (r2_wide
             .loc[r2_wide.groupby('model', observed=True)['sum_r2'].idxmax(),
                  ['model', 'feature_set']]
             .set_index('model')['feature_set'])

# Collect best-tier rows
best_rows = []
for m in model_order:
    tier = best_tier[m]
    for tgt in tgts:
        sub = df[(df.model == m) & (df.feature_set == tier) & (df.target == tgt)]
        best_rows.append({
            'model': m, 'tier': str(tier), 'target': tgt,
            'r2':   sub['r2'].values[0],
            'rmse': sub['rmse'].values[0],
            'mae':  sub['mae'].values[0],
        })
bdf = pd.DataFrame(best_rows)

TOL_R2        = 0.0005
TOL_RMSE_TP   = 0.05
TOL_RMSE_CHLA = 0.05
TOL_RMSE_SEC  = 0.005
TOL_MAE_CHLA  = 0.005
TOL_MAE_TP    = 0.05
TOL_MAE_SEC   = 0.005

# Dagger: flag when a target's global-best R² is NOT achieved at the selected tier
dagger_models = {}
for m in model_order:
    shown_tier = str(best_tier[m])
    for tgt in tgts:
        global_best = df[df.target == tgt]['r2'].max()
        shown_r2    = df[(df.model == m) & (df.feature_set == shown_tier) &
                         (df.target == tgt)]['r2'].values[0]
        if global_best - shown_r2 > TOL_R2:
            sub     = df[(df.model == m) & (df.target == tgt)]
            peak_r2 = sub['r2'].max()
            if peak_r2 - shown_r2 > TOL_R2:
                peak_tier = str(sub.loc[sub['r2'].idxmax(), 'feature_set'])
                dagger_models.setdefault(m, {})[tgt] = (peak_tier, round(peak_r2, 3))

# Bold thresholds from GLOBAL best (all model × tier)
col_best_r2   = {t: df[df.target == t]['r2'].max()   for t in tgts}
col_best_rmse = {t: df[df.target == t]['rmse'].min() for t in tgts}
col_best_mae  = {t: df[df.target == t]['mae'].min()  for t in tgts}

def fmt_r2(v):
    return (r'$-$' + f'{abs(v):.3f}') if v < 0 else f'{v:.3f}'

def fmt_rmse(v, tgt):
    return f'{v:.2f}' if tgt == 'secchi_m' else f'{v:.1f}'

def fmt_mae(v, tgt):
    return f'{v:.2f}' if tgt in ('chla_ug_l', 'secchi_m') else f'{v:.1f}'

def B(s, flag):
    return r'\textbf{' + s + '}' if flag else s

model_refs = {
    'RF':             r'~\citep{breiman2001random}',
    'XGBoost':        r'~\citep{chen2016xgboost}',
    'LightGBM':       r'~\citep{ke2017lightgbm}',
    'FT-Transformer': r'~\citep{gorishniy2021revisiting}',
}

# Dagger footnote: group by target for readability
tgt_label = {'chla_ug_l': 'Chl-\\textit{a}', 'tp_ug_l': 'TP', 'secchi_m': 'Secchi'}
dag_by_tgt = {}
for m, tgt_dict in dagger_models.items():
    for tgt, (tier, val) in tgt_dict.items():
        dag_by_tgt.setdefault(tgt, []).append(
            m + r'\,=\,' + str(val) + r' (\texttt{' + tier.replace('_', r'\_') + r'})'
        )
dag_parts = []
for tgt in tgts:
    if tgt in dag_by_tgt:
        dag_parts.append(tgt_label[tgt] + ': ' + ', '.join(dag_by_tgt[tgt]))
dag_str = '; '.join(dag_parts) if dag_parts else 'none'

out = []
L = out.append

L(r'\begin{table}[t]')
L(r'\centering')
L(
    r'\caption{Tabular baseline results at each model\'s best feature tier '
    r'(maximising sum of $R^{2}$ across all three targets). '
    r'GroupKFold($k{=}5$) spatial CV\@. '
    r'$n$: Chl-\textit{a}\,=\,49{,}300; TP\,=\,502{,}638; Secchi\,=\,16{,}163. '
    r'RMSE: \textmu g\,L$^{-1}$ (Chl-\textit{a}, TP); m (Secchi). '
    r'$\dagger$\,TP peak at a different tier: ' + dag_str + r'. '
    r'\textbf{Bold}\,=\,best per metric column.}'
)
L(r'\label{tab:tabular}')
L(r'\resizebox{\columnwidth}{!}{%')
L(r'\begin{tabular}{@{}l l r r r r r r r r r@{}}')
L(r'\toprule')
L(r'\textbf{Model} & \textbf{Tier}')
L(r'  & \multicolumn{3}{c}{\textbf{Chl-\textit{a}}}')
L(r'  & \multicolumn{3}{c}{\textbf{TP}}')
sec_header = r'  & \multicolumn{3}{c}{\textbf{Secchi}} \\'
L(sec_header)
L(r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}')
subhdr = r'  & & $R^{2}$ & RMSE & MAE & $R^{2}$ & RMSE & MAE & $R^{2}$ & RMSE & MAE \\'
L(subhdr)
L(r'\midrule')

for m in model_order:
    tier  = str(best_tier[m])
    cite  = model_refs.get(m, '')
    rc    = r'\rowcolor{gray!8}' if m == 'LightGBM' else ''
    mname = (r'\textbf{' + m + '}' + cite) if m == 'LightGBM' else (m + cite)

    cells = []
    for tgt in tgts:
        row  = bdf[(bdf.model == m) & (bdf.target == tgt)].iloc[0]
        r2, rmse, mae = row['r2'], row['rmse'], row['mae']
        dag = r'$^{\dagger}$' if (m in dagger_models and tgt in dagger_models[m]) else ''

        tol_r2   = TOL_R2
        tol_rmse = TOL_RMSE_SEC  if tgt == 'secchi_m' else (TOL_RMSE_TP if tgt == 'tp_ug_l' else TOL_RMSE_CHLA)
        tol_mae  = TOL_MAE_SEC   if tgt == 'secchi_m' else (TOL_MAE_TP  if tgt == 'tp_ug_l' else TOL_MAE_CHLA)
        r2_s   = B(fmt_r2(r2) + dag,     abs(r2   - col_best_r2[tgt])   < tol_r2)
        rmse_s = B(fmt_rmse(rmse, tgt),  abs(rmse - col_best_rmse[tgt]) < tol_rmse)
        mae_s  = B(fmt_mae(mae, tgt),    abs(mae  - col_best_mae[tgt])  < tol_mae)
        cells += [r2_s, rmse_s, mae_s]

    row_tex = rc + mname + ' & ' + tier_tex[tier] + ' & ' + ' & '.join(cells) + r' \\'
    L(row_tex)

L(r'\bottomrule')
L(r'\end{tabular}%')
L(r'}')
L(r'\end{table}')

result = '\n'.join(out)
print(result)
with open('Reports/tab_tabular_best.tex', 'w', encoding='utf-8') as f:
    f.write(result)
print('\n% Saved to Reports/tab_tabular_best.tex')
