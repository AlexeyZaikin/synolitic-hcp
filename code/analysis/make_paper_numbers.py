"""Generate every number, table and traceability entry of the manuscript.

The manuscript never contains a hand-typed result. All values are emitted here
from the results files as LaTeX macros (generated/numbers.tex) and tables
(generated/tab_*.tex), and the same pass writes output/list_of_results/list_of_results.tex
so that each reported value carries its source file and the script that made it.

Outputs
    output/generated/numbers.tex
    output/generated/tab_*.tex
    output/list_of_results/list_of_results.tex

Usage
    python code/analysis/make_paper_numbers.py
"""

import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr, wilcoxon

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from code.utils.data_loader import TASKS, TASK_DISPLAY, MMP_REGIONS  # noqa: E402
from code.analysis.region_groups import GROUPS  # noqa: E402

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'metrics')
# Parcellated HCP time series (not distributed with this repository; see README).
# Override with the environment variable HCP_DATA_DIR.
DATA_DIR = os.environ.get(
    'HCP_DATA_DIR', os.path.join(PROJECT_ROOT, 'data', 'HCP_v4', 'split_data', 'split'))
GEN_DIR = os.path.join(PROJECT_ROOT, 'output', 'generated')
LOR_DIR = os.path.join(PROJECT_ROOT, 'output', 'list_of_results')

CODE = {'WM': 'WM', 'GAMBLING': 'Gam', 'MOTOR': 'Mot', 'LANGUAGE': 'Lan',
        'SOCIAL': 'Soc', 'RELATIONAL': 'Rel', 'EMOTION': 'Emo'}
MODELS = [('all_direct', 'AllDir', 'all 379 regions, direct features'),
          ('weak_direct', 'LpDir', 'weak regions, direct features'),
          ('weak_corr', 'LpCorr', 'weak regions, correlation graph'),
          ('weak_ensemble', 'LpEns', 'weak regions, synolitic graph'),
          ('weak_marginal', 'LpMarg', 'weak regions, marginal-only synolitic'),
          ('all_ensemble', 'AllEns', 'all 379 regions, synolitic graph')]
SHORT = {'WM': 'Working Memory', 'GAMBLING': 'Gambling', 'MOTOR': 'Motor',
         'LANGUAGE': 'Language', 'SOCIAL': 'Social', 'RELATIONAL': 'Relational',
         'EMOTION': 'Emotion'}

macros = []
trace = []


def mac(name, value, fmt='{:.2f}'):
    txt = fmt.format(value) if not isinstance(value, str) else value
    if txt in ('-0.00', '+-0.00', '-0.000'):
        txt = txt.replace('-0.00', '0.00').replace('-0.000', '0.000')
    macros.append((name, txt))
    return txt


def rec(value, description, source, script):
    trace.append((value, description, source, script))


def load(name):
    with open(os.path.join(RESULTS_DIR, name)) as f:
        return json.load(f)


def main():
    os.makedirs(GEN_DIR, exist_ok=True)
    os.makedirs(LOR_DIR, exist_ok=True)
    R = load('all_results.json')
    P = load('preprocessing_check.json')
    G = load('gsr_sensitivity.json') if os.path.exists(
        os.path.join(RESULTS_DIR, 'gsr_sensitivity.json')) else None
    I = load('interpretability.json') if os.path.exists(
        os.path.join(RESULTS_DIR, 'interpretability.json')) else None
    M = load('multiclass_results.json') if os.path.exists(
        os.path.join(RESULTS_DIR, 'multiclass_results.json')) else None

    src_main = 'results/metrics/all_results.json'
    scr_main = 'code/models/run\\_pipeline.py'

    # ---------------------------------------------------------------- accuracy
    for task in TASKS:
        r = R[task]
        c = CODE[task]
        mac(f'vN{c}', r['n_samples'], '{:d}')
        mac(f'vNweak{c}', r['n_weak'], '{:d}')
        mac(f'vNweakLo{c}', r['n_weak_range'][0], '{:d}')
        mac(f'vNweakHi{c}', r['n_weak_range'][1], '{:d}')
        nw = r['n_weak']
        mac(f'vEdges{c}', nw * (nw - 1) // 2, '{:d}')
        rec(str(r['n_weak']), f'{SHORT[task]}: median number of weak regions '
            f'across folds (60 per cent threshold)', src_main, scr_main)
        for key, code, desc in MODELS:
            d = r[key]
            a = mac(f'v{code}{c}Acc', d['accuracy_mean'] * 100)
            s = mac(f'v{code}{c}AccSd', d['accuracy_std'] * 100)
            u = mac(f'v{code}{c}Auc', d['auc_mean'], '{:.3f}')
            us = mac(f'v{code}{c}AucSd', d['auc_std'], '{:.3f}')
            rec(f'{a} +/- {s}', f'{SHORT[task]}: accuracy (per cent), {desc}',
                src_main, scr_main)
            rec(f'{u} +/- {us}', f'{SHORT[task]}: ROC AUC, {desc}', src_main, scr_main)

    def col(key, metric='accuracy_mean'):
        return np.array([R[t][key][metric] for t in TASKS]) * (
            100 if metric == 'accuracy_mean' else 1)

    ens, corr, dir_, marg = (col('weak_ensemble'), col('weak_corr'),
                             col('weak_direct'), col('weak_marginal'))
    diffs = {'EnsVsCorr': ens - corr, 'EnsVsDir': ens - dir_,
             'EnsVsMarg': ens - marg}
    for name, d in diffs.items():
        mac(f'vMean{name}', float(d.mean()), '{:+.2f}')
        mac(f'vMin{name}', float(d.min()), '{:+.2f}')
        mac(f'vMax{name}', float(d.max()), '{:+.2f}')
        mac(f'vNpos{name}', int((d > 0).sum()), '{:d}')
        mac(f'vNneg{name}', int((d < 0).sum()), '{:d}')
        rec(f'{d.mean():+.2f}', f'Mean accuracy difference across the seven '
            f'tasks, {name}', src_main, 'code/analysis/make\\_paper\\_numbers.py')
    for nm, v in (('MeanLpEns', ens), ('MeanLpCorr', corr), ('MeanLpDir', dir_),
                  ('MeanLpMarg', marg), ('MeanAllDir', col('all_direct')),
                  ('MeanAllEns', col('all_ensemble'))):
        mac(f'v{nm}', float(v.mean()))
        rec(f'{v.mean():.2f}', f'Mean accuracy across the seven tasks, {nm}',
            src_main, 'code/analysis/make\\_paper\\_numbers.py')
    # paired tests across the seven tasks (one paired observation per task)
    for name, d in diffs.items():
        try:
            stat, pv = wilcoxon(d)
            mac(f'vW{name}', float(stat), '{:.1f}')
            mac(f'vP{name}', float(pv), '{:.4f}')
            rec(f'{pv:.4f}', f'Wilcoxon signed-rank p across the seven tasks, '
                f'{name}', src_main, 'code/analysis/make\\_paper\\_numbers.py')
        except ValueError:
            mac(f'vP{name}', 'n/a')

    # cohort size (counts subject files, so it needs the data directory)
    if os.path.isdir(DATA_DIR):
        n_subj = {}
        for task in TASKS:
            subs = {f.split('_')[0] for f in os.listdir(os.path.join(DATA_DIR, task))
                    if f.endswith('.pickle') and f.split('_')[1] == 'LR'}
            n_subj[task] = len(subs)
            mac(f'vSubj{CODE[task]}', len(subs), '{:d}')
        mac('vSubjMin', min(n_subj.values()), '{:d}')
        mac('vSubjMax', max(n_subj.values()), '{:d}')
        rec(f'{min(n_subj.values())}--{max(n_subj.values())}',
            'Number of subjects contributing LR-encoded samples, range over tasks',
            'data/HCP\\_v4/split\\_data/split', 'code/analysis/make\\_paper\\_numbers.py')
    else:
        print(f'NOTE: {DATA_DIR} not found; the subject-count macros (vSubj*) '
              'are not written. Every other value comes from results/.')

    # feature-to-sample ratios: n_train is three of the four folds
    ntr = int(round(R[list(TASKS)[0]]['n_samples'] * 0.75))
    nw_max = max(R[t]['n_weak'] for t in TASKS)
    mac('vNtrain', ntr, '{:d}')
    mac('vPNallDir', 2 * 379 / ntr, '{:.2f}')
    mac('vPNlpDirMax', 2 * nw_max / ntr, '{:.2f}')
    mac('vPNlpNodeMax', nw_max / ntr, '{:.2f}')
    rec(f'{2*379/ntr:.2f}', 'Largest feature-to-sample ratio, all-region direct '
        'model', src_main, 'code/analysis/make\\_paper\\_numbers.py')

    # does the advantage over the correlation graph track how poorly that graph
    # does? (seven paradigms, so a tendency at best)
    rg, pg = spearmanr(corr, ens - corr)
    mac('vRhoGapCorr', float(rg), '{:+.2f}')
    mac('vPGapCorr', float(pg), '{:.2f}')
    rec(f'{rg:+.2f}', 'Spearman correlation across tasks between correlation-graph '
        'accuracy and the synolitic advantage over it', src_main,
        'code/analysis/make\\_paper\\_numbers.py')

    alld = col('all_direct')
    alle = col('all_ensemble')
    mac('vMeanAllEnsVsDir', float((alle - alld).mean()), '{:+.2f}')
    mac('vNposAllEnsVsDir', int((alle > alld).sum()), '{:d}')
    rec(f'{(alle - alld).mean():+.2f}', 'Mean accuracy difference, all-region '
        'synolitic minus all-region direct', src_main,
        'code/analysis/make\\_paper\\_numbers.py')

    mac('vMinLpEns', float(ens.min()))
    mac('vMaxLpEns', float(ens.max()))
    mac('vMinLpCorr', float(corr.min()))
    mac('vMaxLpCorr', float(corr.max()))

    # ------------------------------------------------------------- thresholds
    for thr, tc in ((55, 'Lo'), (60, 'Mid'), (65, 'Hi')):
        e = np.array([R[t]['threshold_results'][f'thr_{thr}']['weak_ensemble']
                      ['accuracy_mean'] for t in TASKS]) * 100
        c_ = np.array([R[t]['threshold_results'][f'thr_{thr}']['weak_corr']
                       ['accuracy_mean'] for t in TASKS]) * 100
        d_ = np.array([R[t]['threshold_results'][f'thr_{thr}']['weak_direct']
                       ['accuracy_mean'] for t in TASKS]) * 100
        mac(f'vThr{tc}EnsMean', float(e.mean()))
        mac(f'vThr{tc}EnsMin', float(e.min()))
        mac(f'vThr{tc}EnsMax', float(e.max()))
        mac(f'vThr{tc}EnsVsCorr', float((e - c_).mean()), '{:+.2f}')
        mac(f'vThr{tc}EnsVsDir', float((e - d_).mean()), '{:+.2f}')
        mac(f'vThr{tc}NposEnsVsCorr', int((e > c_).sum()), '{:d}')
        rec(f'{e.mean():.2f}', f'Mean synolitic accuracy over tasks at the '
            f'{thr} per cent threshold', src_main, scr_main)

    # ---------------------------------------------------------- preprocessing
    pp = P['pooled']
    src_pp = 'results/metrics/preprocessing\\_check.json'
    scr_pp = 'code/analysis/preprocessing\\_check.py'
    mac('vGsCorrMedian', pp['gs_corr_median'], '{:.3f}')
    mac('vEdgeCorrMean', pp['edge_corr_mean'], '{:.3f}')
    mac('vFracNegEdges', pp['edge_frac_negative'] * 100, '{:.1f}')
    mac('vPCone', pp['pc1_variance_share_median'] * 100, '{:.1f}')
    mac('vPConeGs', pp['pc1_global_signal_corr_median'], '{:.2f}')
    mac('vPConeSameSign', pp['pc1_same_sign_loading_frac_median'] * 100, '{:.0f}')
    mac('vRegionSd', pp['region_sd_median'], '{:.3f}')
    mac('vRegionMean', pp['region_mean_median'], '{:+.3f}')
    mac('vPostGsrEdgeCorr', pp['posthoc_gsr_edge_corr_mean'], '{:.3f}')
    mac('vPostGsrFracNeg', pp['posthoc_gsr_edge_frac_negative'] * 100, '{:.1f}')
    mac('vPostGsrGsCorr', pp['posthoc_gsr_gs_corr_abs_median'], '{:.0e}')
    mac('vNsubjPrep', P['n_subjects_per_task'], '{:d}')
    for nm, val, desc in (
            ('median correlation of a parcel with the global signal',
             pp['gs_corr_median'], 'GSR detector, pooled over tasks'),
            ('mean off-diagonal correlation', pp['edge_corr_mean'], 'pooled'),
            ('share of negative edges (per cent)',
             pp['edge_frac_negative'] * 100, 'pooled'),
            ('variance share of the first principal component (per cent)',
             pp['pc1_variance_share_median'] * 100, 'pooled median')):
        rec(f'{val:.3f}', f'{nm}, {desc}', src_pp, scr_pp)

    # ------------------------------------------------------------------- GSR
    if G:
        src_g = 'results/metrics/gsr\\_sensitivity.json'
        scr_g = 'code/analysis/gsr\\_sensitivity.py'
        ge = np.array([G[t]['weak_ensemble']['accuracy_mean'] for t in TASKS]) * 100
        gc = np.array([G[t]['weak_corr']['accuracy_mean'] for t in TASKS]) * 100
        gd = np.array([G[t]['weak_direct']['accuracy_mean'] for t in TASKS]) * 100
        gn = np.array([G[t]['n_weak_median'] for t in TASKS])
        nn = np.array([R[t]['n_weak'] for t in TASKS])
        mac('vGsrMeanEns', float(ge.mean()))
        mac('vGsrMeanCorr', float(gc.mean()))
        mac('vGsrMeanDir', float(gd.mean()))
        gm = np.array([G[t]['weak_marginal']['accuracy_mean'] for t in TASKS]) * 100
        mac('vGsrMeanMarg', float(gm.mean()))
        gdiffs = {'EnsVsCorr': ge - gc, 'EnsVsDir': ge - gd, 'EnsVsMarg': ge - gm}
        for name, dd in gdiffs.items():
            mac(f'vGsrMean{name}', float(dd.mean()), '{:+.2f}')
            mac(f'vGsrMin{name}', float(dd.min()), '{:+.2f}')
            mac(f'vGsrMax{name}', float(dd.max()), '{:+.2f}')
            mac(f'vGsrNpos{name}', int((dd > 0).sum()), '{:d}')
            try:
                mac(f'vGsrP{name}', float(wilcoxon(dd)[1]), '{:.4f}')
            except ValueError:
                mac(f'vGsrP{name}', 'n/a')
            rec(f'{dd.mean():+.2f}', f'Mean accuracy difference after GSR across '
                f'the seven tasks, {name}', src_g,
                'code/analysis/make\\_paper\\_numbers.py')
        mac('vGsrNweakMin', int(gn.min()), '{:d}')
        mac('vGsrNweakMax', int(gn.max()), '{:d}')
        mac('vGsrMeanNweak', float(gn.mean()), '{:.0f}')
        mac('vNoGsrMeanNweak', float(nn.mean()), '{:.0f}')
        mac('vGsrMeanDeltaEns', float((ge - ens).mean()), '{:+.2f}')
        for task in TASKS:
            c = CODE[task]
            mac(f'vGsrNweak{c}', G[task]['n_weak_median'], '{:d}')
            for key, code in (('weak_direct', 'LpDir'), ('weak_corr', 'LpCorr'),
                              ('weak_ensemble', 'LpEns'), ('weak_marginal', 'LpMarg')):
                v = mac(f'vGsr{code}{c}Acc', G[task][key]['accuracy_mean'] * 100)
                rec(v, f'{SHORT[task]}: accuracy after GSR, {key}', src_g, scr_g)
        rec(f'{ge.mean():.2f}', 'Mean synolitic accuracy over tasks after GSR',
            src_g, scr_g)

    # -------------------------------------------------------- interpretability
    if I:
        src_i = 'results/metrics/interpretability.json'
        scr_i = 'code/analysis/interpretability.py'
        rho = np.array([t['spearman_importance_vs_individual_accuracy']
                        for t in I['tasks']])
        mac('vRhoMean', float(rho.mean()), '{:+.2f}')
        mac('vRhoMin', float(rho.min()), '{:+.2f}')
        mac('vRhoMax', float(rho.max()), '{:+.2f}')
        mac('vRhoNpos', int((rho > 0).sum()), '{:d}')
        rec(f'{rho.mean():+.2f}', 'Mean Spearman correlation between model '
            'weight and individual region accuracy', src_i, scr_i)
        mac('vRhoNsig', int(sum(1 for t in I['tasks'] if t['spearman_p'] < 0.05)), '{:d}')
        for t, task in zip(I['tasks'], TASKS):
            c = CODE[task]
            best = max(((g, v['enrichment']) for g, v in t['systems'].items()
                        if v['enrichment'] == v['enrichment']),
                       key=lambda kv: kv[1])
            mac(f'vEnrichSystem{c}', best[0].lower())
            mac(f'vEnrich{c}', best[1], '{:.2f}')
            rec(f'{best[1]:.2f}', f'{SHORT[task]}: enrichment of the most '
                f'weighted system ({best[0]})', src_i, scr_i)
            mac(f'vRho{c}', t['spearman_importance_vs_individual_accuracy'], '{:+.2f}')
            mac(f'vTopRegion{c}', t['top_regions'][0]['region'].replace('_', '\\_'))
            mac(f'vTopSystem{c}', t['top_regions'][0]['system'])
            mac(f'vNstable{c}', t['n_stable_weak_regions'], '{:d}')
            rec(t['top_regions'][0]['region'].replace('_', '\\_'),
                f'{SHORT[task]}: highest-weighted weak region', src_i, scr_i)

    # ------------------------------------------------------------- multiclass
    if M:
        src_m = 'results/metrics/multiclass_results.json'
        scr_m = 'code/analysis/multiclass.py'
        mac('vMcK', M['n_classes'], '{:d}')
        mac('vMcChance', M['chance'] * 100)
        mac('vMcThreshold', M['threshold'] * 100)
        mac('vMcMargin', M['margin'] * 100)
        mac('vMcNsamples', M['n_samples'], '{:d}')
        mac('vMcNsubj', M['n_subjects'], '{:d}')
        mac('vMcNweak', M['n_weak_median'], '{:d}')
        mac('vMcNweakLo', M['n_weak_range'][0], '{:d}')
        mac('vMcNweakHi', M['n_weak_range'][1], '{:d}')
        mc_keys = [('all_direct', 'AllDir', 'all 379 regions, direct features'),
                   ('weak_direct', 'LpDir', 'weak regions, direct features'),
                   ('weak_corr', 'LpCorr', 'weak regions, correlation graph'),
                   ('weak_ensemble', 'LpEns', 'weak regions, synolitic graph'),
                   ('weak_marginal', 'LpMarg', 'weak regions, marginal-only')]
        for key, code, desc in mc_keys:
            d = M[key]
            a = mac(f'vMc{code}', d['accuracy_mean'] * 100)
            mac(f'vMc{code}Sd', d['accuracy_std'] * 100)
            mac(f'vMc{code}Auc', d['auc_mean'], '{:.3f}')
            rec(a, f'{M["n_classes"]}-class: accuracy (per cent), {desc}',
                src_m, scr_m)
        ra = np.asarray(M['region_accs_per_fold'], dtype=float).mean(axis=0)
        mac('vMcRegionMax', float(ra.max()) * 100)
        mac('vMcRegionMin', float(ra.min()) * 100)
        rec(f'{ra.max()*100:.2f}', f'{M["n_classes"]}-class: highest individual '
            'region accuracy (mean over folds)', src_m, scr_m)
        me = M['weak_ensemble']['accuracy_mean'] * 100
        for nm, key in (('EnsVsCorr', 'weak_corr'), ('EnsVsDir', 'weak_direct'),
                        ('EnsVsMarg', 'weak_marginal'),
                        ('EnsVsAllDir', 'all_direct')):
            v = me - M[key]['accuracy_mean'] * 100
            mac(f'vMc{nm}', float(v), '{:+.2f}')
            rec(f'{v:+.2f}', f'{M["n_classes"]}-class: accuracy difference, {nm}',
                src_m, scr_m)

        with open(os.path.join(GEN_DIR, 'tab_multiclass.tex'), 'w') as f:
            f.write('\\begin{tabular}{lcc}\n\\hline\nCondition & Accuracy (\\%) '
                    '& AUC (macro OvR) \\\\\n\\hline\n')
            labels = {'all_direct': 'All 379 regions, direct',
                      'weak_direct': 'Weak regions, direct',
                      'weak_corr': 'Weak regions, correlation graph',
                      'weak_ensemble': 'Weak regions, synolitic graph',
                      'weak_marginal': 'Weak regions, marginal-only synolitic'}
            for key, _, _ in mc_keys:
                d = M[key]
                f.write(f"{labels[key]} & "
                        f"${d['accuracy_mean']*100:.2f} \\pm "
                        f"{d['accuracy_std']*100:.2f}$ & "
                        f"${d['auc_mean']:.3f} \\pm {d['auc_std']:.3f}$ \\\\\n")
            f.write(f"\\hline\nChance & ${M['chance']*100:.2f}$ & $0.500$ "
                    f"\\\\\n\\hline\n\\end{{tabular}}\n")

    # ------------------------------------------------------------------ tables
    def acc_cell(task, key):
        d = R[task][key]
        return f"${d['accuracy_mean']*100:.2f} \\pm {d['accuracy_std']*100:.2f}$"

    def auc_cell(task, key):
        d = R[task][key]
        return f"${d['auc_mean']:.3f} \\pm {d['auc_std']:.3f}$"

    hdr = ('Task & All regions, & Weak regions, & Weak regions, & Weak regions, '
           '& Weak regions, & All regions, \\\\\n & direct & direct & correlation '
           '& synolitic & marginal & synolitic \\\\')
    order = ['all_direct', 'weak_direct', 'weak_corr', 'weak_ensemble',
             'weak_marginal', 'all_ensemble']

    with open(os.path.join(GEN_DIR, 'tab_main.tex'), 'w') as f:
        f.write('\\begin{tabular}{lcccccc}\n\\hline\n' + hdr + '\n\\hline\n')
        for t in TASKS:
            f.write(SHORT[t] + ' & ' + ' & '.join(acc_cell(t, k) for k in order)
                    + ' \\\\\n')
        f.write('\\hline\nMean & ' + ' & '.join(
            f"${np.mean([R[t][k]['accuracy_mean'] for t in TASKS])*100:.2f}$"
            for k in order) + ' \\\\\n\\hline\n\\end{tabular}\n')

    with open(os.path.join(GEN_DIR, 'tab_auc.tex'), 'w') as f:
        f.write('\\begin{tabular}{lcccccc}\n\\hline\n' + hdr + '\n\\hline\n')
        for t in TASKS:
            f.write(SHORT[t] + ' & ' + ' & '.join(auc_cell(t, k) for k in order)
                    + ' \\\\\n')
        f.write('\\hline\nMean & ' + ' & '.join(
            f"${np.mean([R[t][k]['auc_mean'] for t in TASKS]):.3f}$"
            for k in order) + ' \\\\\n\\hline\n\\end{tabular}\n')

    with open(os.path.join(GEN_DIR, 'tab_regions.tex'), 'w') as f:
        f.write('\\begin{tabular}{lccc}\n\\hline\nTask & Samples & Weak regions, '
                'median [range] & Edges at the median \\\\\n\\hline\n')
        for t in TASKS:
            r = R[t]
            nw = r['n_weak']
            f.write(f"{SHORT[t]} & {r['n_samples']} & {nw} "
                    f"[{r['n_weak_range'][0]}--{r['n_weak_range'][1]}] & "
                    f"{nw*(nw-1)//2} \\\\\n")
        f.write('\\hline\n\\end{tabular}\n')

    with open(os.path.join(GEN_DIR, 'tab_improvement.tex'), 'w') as f:
        f.write('\\begin{tabular}{lccc}\n\\hline\nTask & Synolitic $-$ correlation '
                '& Synolitic $-$ direct & Synolitic $-$ marginal \\\\\n\\hline\n')
        for i, t in enumerate(TASKS):
            def d3(x):
                return '\\pm 0.00' if abs(x) < 0.005 else f'{x:+.2f}'
            f.write(f"{SHORT[t]} & ${d3(diffs['EnsVsCorr'][i])}$ & "
                    f"${d3(diffs['EnsVsDir'][i])}$ & "
                    f"${d3(diffs['EnsVsMarg'][i])}$ \\\\\n")
        f.write('\\hline\nMean & ' + ' & '.join(
            f"$\\mathbf{{{diffs[k].mean():+.2f}}}$"
            for k in ('EnsVsCorr', 'EnsVsDir', 'EnsVsMarg'))
            + ' \\\\\n\\hline\n\\end{tabular}\n')

    with open(os.path.join(GEN_DIR, 'tab_sensitivity.tex'), 'w') as f:
        f.write('\\begin{tabular}{lcccccc}\n\\hline\n & \\multicolumn{2}{c}{55\\%} '
                '& \\multicolumn{2}{c}{60\\%} & \\multicolumn{2}{c}{65\\%} \\\\\n'
                'Task & $N$ & Synolitic & $N$ & Synolitic & $N$ & Synolitic \\\\\n'
                '\\hline\n')
        for t in TASKS:
            cells = []
            for thr in (55, 60, 65):
                d = R[t]['threshold_results'][f'thr_{thr}']
                cells += [str(d['n_weak_median']),
                          f"{d['weak_ensemble']['accuracy_mean']*100:.2f}"]
            f.write(SHORT[t] + ' & ' + ' & '.join(cells) + ' \\\\\n')
        f.write('\\hline\n\\end{tabular}\n')

    with open(os.path.join(GEN_DIR, 'tab_preprocessing.tex'), 'w') as f:
        f.write('\\begin{tabular}{lcccccc}\n\\hline\nTask & Parcel mean & Parcel '
                'SD & $r$(parcel, GS) & Mean edge $r$ & Negative edges (\\%) & '
                'PC1 variance (\\%) \\\\\n\\hline\n')
        for t in P['tasks']:
            f.write(f"{t['display']} & {t['region_mean_median']:+.3f} & "
                    f"{t['region_sd_median']:.3f} & {t['gs_corr_median']:+.3f} & "
                    f"{t['edge_corr_mean']:+.3f} & "
                    f"{t['edge_frac_negative']*100:.1f} & "
                    f"{t['pc1_variance_share_median']*100:.1f} \\\\\n")
        f.write('\\hline\nAfter post-hoc GSR & --- & --- & '
                f"{pp['posthoc_gsr_gs_corr_abs_median']:.0e} & "
                f"{pp['posthoc_gsr_edge_corr_mean']:+.3f} & "
                f"{pp['posthoc_gsr_edge_frac_negative']*100:.1f} & --- \\\\\n"
                '\\hline\n\\end{tabular}\n')

    if G:
        with open(os.path.join(GEN_DIR, 'tab_gsr.tex'), 'w') as f:
            f.write('\\begin{tabular}{lccccccccc}\n\\hline\n'
                    ' & \\multicolumn{4}{c}{As distributed} & '
                    '\\multicolumn{4}{c}{After GSR} \\\\\n'
                    'Task & $N$ & Corr. & Direct & Syn. & $N$ & Corr. & Direct '
                    '& Syn. \\\\\n\\hline\n')
            for t in TASKS:
                f.write(f"{SHORT[t]} & {R[t]['n_weak']} & "
                        f"{R[t]['weak_corr']['accuracy_mean']*100:.2f} & "
                        f"{R[t]['weak_direct']['accuracy_mean']*100:.2f} & "
                        f"{R[t]['weak_ensemble']['accuracy_mean']*100:.2f} & "
                        f"{G[t]['n_weak_median']} & "
                        f"{G[t]['weak_corr']['accuracy_mean']*100:.2f} & "
                        f"{G[t]['weak_direct']['accuracy_mean']*100:.2f} & "
                        f"{G[t]['weak_ensemble']['accuracy_mean']*100:.2f} \\\\\n")
            f.write(f"\\hline\nMean & {nn.mean():.0f} & {corr.mean():.2f} & "
                    f"{dir_.mean():.2f} & {ens.mean():.2f} & {gn.mean():.0f} & "
                    f"{gc.mean():.2f} & {gd.mean():.2f} & {ge.mean():.2f} "
                    f"\\\\\n\\hline\n\\end{{tabular}}\n")

        with open(os.path.join(GEN_DIR, 'tab_gsr_diff.tex'), 'w') as f:
            f.write('\\begin{tabular}{lcccccc}\n\\hline\n'
                    ' & \\multicolumn{3}{c}{As distributed} & '
                    '\\multicolumn{3}{c}{After GSR} \\\\\n'
                    'Task & $-$Corr. & $-$Direct & $-$Marginal & $-$Corr. & '
                    '$-$Direct & $-$Marginal \\\\\n\\hline\n')
            for i, t in enumerate(TASKS):
                f.write(f"{SHORT[t]} & "
                        f"${diffs['EnsVsCorr'][i]:+.2f}$ & "
                        f"${diffs['EnsVsDir'][i]:+.2f}$ & "
                        f"${diffs['EnsVsMarg'][i]:+.2f}$ & "
                        f"${gdiffs['EnsVsCorr'][i]:+.2f}$ & "
                        f"${gdiffs['EnsVsDir'][i]:+.2f}$ & "
                        f"${gdiffs['EnsVsMarg'][i]:+.2f}$ \\\\\n")
            f.write('\\hline\nMean & ' + ' & '.join(
                f"$\\mathbf{{{v.mean():+.2f}}}$" for v in
                [diffs['EnsVsCorr'], diffs['EnsVsDir'], diffs['EnsVsMarg'],
                 gdiffs['EnsVsCorr'], gdiffs['EnsVsDir'], gdiffs['EnsVsMarg']])
                + ' \\\\\n\\hline\n\\end{tabular}\n')

    if I:
        with open(os.path.join(GEN_DIR, 'tab_interpret.tex'), 'w') as f:
            f.write('\\begin{tabular}{llll}\n\\hline\nTask & Five '
                    'highest-weighted weak regions & System of the first & '
                    '$\\rho$ \\\\\n\\hline\n')
            for t in I['tasks']:
                names = ', '.join(x['region'].replace('_', '\\_')
                                  for x in t['top_regions'][:5])
                f.write(f"{t['display']} & {names} & "
                        f"{t['top_regions'][0]['system']} & "
                        f"{t['spearman_importance_vs_individual_accuracy']:+.2f}"
                        " \\\\\n")
            f.write('\\hline\n\\end{tabular}\n')

    with open(os.path.join(GEN_DIR, 'tab_groups.tex'), 'w') as f:
        f.write('\\begin{tabular}{p{4.2cm}p{10.5cm}}\n\\hline\nSystem & '
                'HCP-MMP1.0 parcel labels (each occurs in both hemispheres) '
                '\\\\\n\\hline\n')
        for g, labels in GROUPS.items():
            txt = ', '.join(l.replace('_', '\\_') for l in labels)
            f.write(f'{g} & \\footnotesize {txt} \\\\\n')
        f.write('\\hline\n\\end{tabular}\n')

    # ------------------------------------------------------------- numbers.tex
    with open(os.path.join(GEN_DIR, 'numbers.tex'), 'w') as f:
        f.write('% Generated by code/analysis/make_paper_numbers.py - do not edit.\n')
        for name, val in macros:
            f.write(f'\\newcommand{{\\{name}}}{{{val}}}\n')

    # ------------------------------------------------------ list_of_results.tex
    with open(os.path.join(LOR_DIR, 'list_of_results.tex'), 'w') as f:
        f.write('\\documentclass[10pt]{article}\n'
                '\\usepackage[margin=2cm]{geometry}\n\\usepackage{enumitem}\n'
                '\\usepackage{longtable}\n'
                '\\begin{document}\n'
                '\\section*{List of Reported Results}\n'
                'Generated by \\texttt{code/analysis/make\\_paper\\_numbers.py}. '
                'Every number that appears in the manuscript is emitted from the '
                'results files listed below as a LaTeX macro, so the manuscript '
                'contains no hand-typed result.\\\\[6pt]\n'
                '\\begin{longtable}{p{2.6cm}p{6.8cm}p{4.2cm}p{4.0cm}}\n'
                '\\textbf{Value} & \\textbf{Description} & \\textbf{Source file} '
                '& \\textbf{Computing script} \\\\ \\hline\n\\endhead\n')
        def esc(text):
            # idempotent: normalise then escape every underscore for LaTeX text mode
            return str(text).replace('\\_', '_').replace('_', '\\_')

        for value, desc, source, script in trace:
            f.write(f'{esc(value)} & {esc(desc)} & \\texttt{{{esc(source)}}} & '
                    f'\\texttt{{{esc(script)}}} \\\\\n')
        f.write('\\end{longtable}\n\\end{document}\n')

    print(f'{len(macros)} macros, {len(trace)} traceability entries')
    print('generated ->', GEN_DIR)


if __name__ == '__main__':
    main()
