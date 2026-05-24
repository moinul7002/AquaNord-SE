#!/usr/bin/env python3
"""Generate combined spatial/temporal/multimodal baseline LaTeX table."""
import pandas as pd
import numpy as np

patch = pd.read_csv('data/outputs/patch_baseline_results.csv')
multi = pd.read_csv('data/outputs/multimodal_baseline_results.csv')
base  = pd.read_csv('data/outputs/baseline_model_results.csv')

tgts = ['chla_ug_l', 'tp_ug_l', 'secchi_m']

# Groups and model order
groups = {
    'Spatial patch':  ['Patch-XGBoost', 'CNN-patch', 'CNN+Tabular'],
    'Temporal':       ['ERA5-LSTM-w30', 'TemporalTrans-ERA5'],
    'Multimodal':     ['MOSAIKS', 'CrossModalNet'],
}

# Merge patch + multimodal into one lookup
df = pd.concat([patch, multi], ignore_index=True)

# Special cases
DIVERGED  = ('CNN+Tabular', 'tp_ug_l')    # training diverged
ARTEFACT  = ('MOSAIKS',     'secchi_m')   # confirmed modelling artefact
RESTRICTED = {'ERA5-LSTM-w30', 'TemporalTrans-ERA5'}  # limited-station models

# LightGBM static reference
ref = (base[(base.model == 'LightGBM') & (base.feature_set == 'static')]
       .set_index('target')[['r2', 'rmse', 'mae', 'n']])

# Bold = best within patch/temporal/multimodal rows only (exclude reference)
valid = df[~df.apply(lambda r: (r.model, r.target) == DIVERGED, axis=1)]
best_r2   = {t: valid[valid.target == t]['r2'].max()   for t in tgts}
best_rmse = {t: valid[valid.target == t]['rmse'].min() for t in tgts}
best_mae  = {t: valid[valid.target == t]['mae'].min()  for t in tgts}

TOL_R2 = 0.0005; TOL_RMSE_TP = 0.05; TOL_RMSE_CHLA = 0.05
TOL_RMSE_SEC = 0.005; TOL_MAE_CHLA = 0.005; TOL_MAE_TP = 0.05; TOL_MAE_SEC = 0.005

def fmt_r2(v):
    return (r'$-$' + f'{abs(v):.3f}') if v < 0 else f'{v:.3f}'
def fmt_rmse(v, tgt):
    return f'{v:.2f}' if tgt == 'secchi_m' else f'{v:.1f}'
def fmt_mae(v, tgt):
    return f'{v:.2f}' if tgt in ('chla_ug_l', 'secchi_m') else f'{v:.1f}'
def B(s, flag):
    return r'\textbf{' + s + '}' if flag else s

def tol_rmse(tgt):
    return TOL_RMSE_SEC if tgt == 'secchi_m' else (TOL_RMSE_TP if tgt == 'tp_ug_l' else TOL_RMSE_CHLA)
def tol_mae(tgt):
    return TOL_MAE_SEC  if tgt == 'secchi_m' else (TOL_MAE_TP  if tgt == 'tp_ug_l' else TOL_MAE_CHLA)

model_refs = {
    'Patch-XGBoost':      r'~\citep{chen2016xgboost}',
    'ERA5-LSTM-w30':      r'~\citep{kratzert2018rainfall}',
    'TemporalTrans-ERA5': r'~\citep{tseng2023lightweight}',
    'MOSAIKS':            r'~\citep{rolf2021generalizable}',
}

out = []
L = out.append

L(r'\begin{table}[t]')
L(r'\centering')
L(
    r'\caption{Spatial patch, temporal, and multimodal baseline results. '
    r'GroupKFold($k{=}5$) spatial CV\@. '
    r'$n$: Chl-\textit{a}\,=\,49{,}300; TP\,=\,502{,}638; Secchi\,=\,16{,}163 '
    r'(full-dataset models). '
    r'$^{*}$ERA5-LSTM and TemporalTrans operate on 31.4\% of stations '
    r'($n_{\mathrm{Chl\text{-}a}}{=}27{,}398$; $n_{\mathrm{Secchi}}{=}6{,}911$). '
    r'$^{\dagger}$CNN+Tabular TP training diverged ($R^{2}{=}{-}348$); excluded. '
    r'$^{\ddagger}$Confirmed modelling artefact. '
    r'RMSE: \textmu g\,L$^{-1}$ (Chl-\textit{a}, TP); m (Secchi). '
    r'\textbf{Bold}\,=\,best within non-tabular models. '
    r'\textit{Italic}\,=\,LightGBM \texttt{static} reference floor.}'
)
L(r'\label{tab:patch}')
L(r'\resizebox{\columnwidth}{!}{%')
NCOLS = 10  # Model + 9 metric columns

L(r'\begin{tabular}{@{}l rrr rrr rrr@{}}')
L(r'\toprule')
L(r'\textbf{Model}')
L(r'  & \multicolumn{3}{c}{\textbf{Chl-\textit{a}}}')
L(r'  & \multicolumn{3}{c}{\textbf{TP}}')
sec_hdr = r'  & \multicolumn{3}{c}{\textbf{Secchi}} \\'
L(sec_hdr)
L(r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}')
subhdr = r'  & $R^{2}$ & RMSE & MAE & $R^{2}$ & RMSE & MAE & $R^{2}$ & RMSE & MAE \\'
L(subhdr)

for grp, models in groups.items():
    L(r'\midrule')
    # Group header spanning all columns
    grp_hdr = (r'\multicolumn{' + str(NCOLS) + r'}{l}{\textit{' + grp + r'}} \\')
    L(grp_hdr)

    for m in models:
        cite  = model_refs.get(m, '')
        star  = r'$^{*}$' if m in RESTRICTED else ''
        mname = r'\quad ' + m + star + cite

        cells = []
        for tgt in tgts:
            row = df[(df.model == m) & (df.target == tgt)]
            if row.empty:
                cells += ['---', '---', '---']
                continue

            r2, rmse, mae = row['r2'].values[0], row['rmse'].values[0], row['mae'].values[0]

            if (m, tgt) == DIVERGED:
                cells += [r'---$^{\dagger}$', '---', '---']
                continue
            if (m, tgt) == ARTEFACT:
                # Round to 1 decimal for display consistency
                cells += [r'$^{\ddagger}$$-$15.6', fmt_rmse(rmse, tgt), fmt_mae(mae, tgt)]
                continue

            r2_s   = B(fmt_r2(r2),           abs(r2   - best_r2[tgt])   < TOL_R2)
            rmse_s = B(fmt_rmse(rmse, tgt),   abs(rmse - best_rmse[tgt]) < tol_rmse(tgt))
            mae_s  = B(fmt_mae(mae, tgt),     abs(mae  - best_mae[tgt])  < tol_mae(tgt))
            cells += [r2_s, rmse_s, mae_s]

        L(mname + ' & ' + ' & '.join(cells) + r' \\')

# LightGBM static reference row
cells = []
for tgt in tgts:
    r2, rmse, mae = ref.loc[tgt, 'r2'], ref.loc[tgt, 'rmse'], ref.loc[tgt, 'mae']
    cells += [r'\textit{' + f'{r2:.3f}' + '}',
              r'\textit{' + fmt_rmse(rmse, tgt) + '}',
              r'\textit{' + fmt_mae(mae, tgt) + '}']
L(r'\midrule')
L(r'\multicolumn{' + str(NCOLS) + r'}{l}{\textit{Reference floor}} \\')
L(r'\quad\textit{LightGBM \texttt{static}} & ' + ' & '.join(cells) + r' \\')

L(r'\bottomrule')
L(r'\end{tabular}%')
L(r'}')
L(r'\end{table}')

result = '\n'.join(out)
print(result)
with open('Reports/tab_patch.tex', 'w', encoding='utf-8') as f:
    f.write(result)
print('\n% Saved to Reports/tab_patch.tex')
