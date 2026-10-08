"""What the synolitic representation of the weak regions actually uses.

Reads the per-fold artefacts written by code/models/run_pipeline.py for the
primary (60 per cent) threshold and answers three questions:

  1. Which regions carry the discriminative weight of the meta-model?
  2. How is that weight distributed over anatomical systems, relative to how
     the weak regions themselves are distributed?
  3. Does the weight simply track the individual discriminability of a region,
     or does the graph reweight regions that are individually uninformative?

Importance of region i is |beta_i|, the meta-model coefficient on the
standardised node summary d_i, averaged over the folds in which region i was
selected. Only regions selected in all four folds enter the ranking.

Outputs
    results/metrics/interpretability.json
    figures/fig8_top_regions.{pdf,png}
    figures/fig9_system_importance.{pdf,png}

Usage
    python code/analysis/interpretability.py
"""

import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from code.utils.data_loader import MMP_REGIONS, TASKS, TASK_DISPLAY  # noqa: E402
from code.analysis.region_groups import GROUPS, group_of, hemisphere_of  # noqa: E402

ART_DIR = os.path.join(PROJECT_ROOT, 'results', 'interpretability')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'metrics')
FIG_DIR = os.path.join(PROJECT_ROOT, 'figures')
N_FOLDS = 4
TOP_K = 10


def analyse_task(task, region_accs_per_fold):
    n_regions = len(MMP_REGIONS)
    coef_sum = np.zeros(n_regions)
    abscoef_sum = np.zeros(n_regions)
    effect_sum = np.zeros(n_regions)
    sel_count = np.zeros(n_regions, dtype=int)
    edge_abs_sum = np.zeros((n_regions, n_regions))
    edge_count = np.zeros((n_regions, n_regions), dtype=int)

    for fold in range(N_FOLDS):
        path = os.path.join(ART_DIR, f'{task}_fold{fold}.npz')
        z = np.load(path)
        idx = z['weak_idx']
        coef = z['meta_coef'].astype(float)
        eff = z['node_effect'].astype(float)
        coef_sum[idx] += coef
        abscoef_sum[idx] += np.abs(coef)
        effect_sum[idx] += eff
        sel_count[idx] += 1
        ee = np.abs(z['edge_effect'].astype(float))
        edge_abs_sum[np.ix_(idx, idx)] += ee
        edge_count[np.ix_(idx, idx)] += 1

    stable = np.where(sel_count == N_FOLDS)[0]
    importance = np.zeros(n_regions)
    importance[stable] = abscoef_sum[stable] / N_FOLDS
    signed = np.zeros(n_regions)
    signed[stable] = coef_sum[stable] / N_FOLDS
    effect = np.zeros(n_regions)
    effect[stable] = effect_sum[stable] / N_FOLDS

    # mean individual accuracy of each region across the four folds
    ind_acc = np.mean(np.asarray(region_accs_per_fold, dtype=float), axis=0)

    order = stable[np.argsort(-importance[stable])]
    top = [{
        'region': MMP_REGIONS[i],
        'system': group_of(MMP_REGIONS[i]),
        'hemisphere': hemisphere_of(MMP_REGIONS[i]),
        'importance': float(importance[i]),
        'signed_coefficient': float(signed[i]),
        'node_effect': float(effect[i]),
        'individual_accuracy': float(ind_acc[i]),
    } for i in order[:TOP_K]]

    # system-level distribution of importance versus of selected regions
    sys_imp, sys_n = {}, {}
    for g in GROUPS:
        members = [i for i in stable if group_of(MMP_REGIONS[i]) == g]
        sys_imp[g] = float(importance[members].sum())
        sys_n[g] = len(members)
    tot_imp = sum(sys_imp.values())
    tot_n = sum(sys_n.values())
    systems = {g: {
        'n_stable_weak_regions': sys_n[g],
        'share_of_regions': (sys_n[g] / tot_n) if tot_n else 0.0,
        'share_of_importance': (sys_imp[g] / tot_imp) if tot_imp else 0.0,
        'enrichment': ((sys_imp[g] / tot_imp) / (sys_n[g] / tot_n))
        if (tot_imp and sys_n[g]) else float('nan'),
    } for g in GROUPS}

    # does the graph weight simply follow individual discriminability?
    rho, pval = spearmanr(ind_acc[stable], importance[stable])

    # most discriminative region pairs (mean |class difference| of the edge weight)
    with np.errstate(invalid='ignore', divide='ignore'):
        edge_mean = np.where(edge_count == N_FOLDS, edge_abs_sum / N_FOLDS, np.nan)
    iu = np.triu_indices(n_regions, k=1)
    vals = edge_mean[iu]
    finite = np.where(np.isfinite(vals))[0]
    top_e = finite[np.argsort(-vals[finite])][:TOP_K]
    top_edges = [{
        'region_i': MMP_REGIONS[iu[0][k]],
        'region_j': MMP_REGIONS[iu[1][k]],
        'system_i': group_of(MMP_REGIONS[iu[0][k]]),
        'system_j': group_of(MMP_REGIONS[iu[1][k]]),
        'edge_effect_abs': float(vals[k]),
    } for k in top_e]

    return {
        'task': task,
        'display': TASK_DISPLAY[task],
        'n_stable_weak_regions': int(len(stable)),
        'top_regions': top,
        'systems': systems,
        'spearman_importance_vs_individual_accuracy': float(rho),
        'spearman_p': float(pval),
        'top_edges': top_edges,
        'edge_effect_abs_median': float(np.nanmedian(vals)),
        '_arrays': {'importance': importance, 'stable': stable, 'ind_acc': ind_acc},
    }


def make_figures(per_task):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(FIG_DIR, exist_ok=True)

    plt.rcParams.update({'font.size': 8, 'axes.spines.top': False,
                         'axes.spines.right': False})

    # fig8: top regions per task
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for ax, r in zip(axes.ravel(), per_task):
        names = [t['region'] for t in r['top_regions']][::-1]
        vals = [t['importance'] for t in r['top_regions']][::-1]
        accs = [t['individual_accuracy'] * 100 for t in r['top_regions']][::-1]
        ypos = np.arange(len(names))
        ax.barh(ypos, vals, color='#3b6ea5')
        ax.set_yticks(ypos)
        ax.set_yticklabels([f'{n}  ({a:.0f}%)' for n, a in zip(names, accs)],
                           fontsize=7)
        ax.set_title(r['display'], fontsize=9, fontweight='bold')
        ax.set_xlabel(r'$|\beta_i|$')
    axes.ravel()[-1].axis('off')
    axes.ravel()[-1].text(
        0.0, 0.5,
        'Ten highest-weighted regions of the\nweak-region synolitic meta-model,\n'
        'per task. Brackets give the individual\naccuracy of that region '
        '(all below the\n60 per cent selection threshold).',
        fontsize=8, va='center')
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG_DIR, f'fig8_top_regions.{ext}'), dpi=300,
                    bbox_inches='tight')
    plt.close(fig)

    # fig9: system enrichment heatmap + importance vs individual accuracy
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 4.6), gridspec_kw={'width_ratios': [1.45, 1]})
    gl = list(GROUPS)
    M = np.array([[per_task[t]['systems'][g]['enrichment'] for g in gl]
                  for t in range(len(per_task))])
    im = ax1.imshow(M, cmap='RdBu_r', vmin=0.5, vmax=1.5, aspect='auto')
    ax1.set_xticks(range(len(gl)))
    ax1.set_xticklabels(gl, rotation=35, ha='right', fontsize=7)
    ax1.set_yticks(range(len(per_task)))
    ax1.set_yticklabels([r['display'] for r in per_task], fontsize=7)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax1.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center',
                         fontsize=6.5,
                         color='white' if abs(M[i, j] - 1) > 0.35 else 'black')
    ax1.set_title('A  Share of model weight per system, divided by share of '
                  'selected regions', loc='left', fontweight='bold', fontsize=9)
    fig.colorbar(im, ax=ax1, fraction=0.03, pad=0.02, label='enrichment')

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(per_task)))
    for c, r in zip(colors, per_task):
        a = r['_arrays']
        ax2.scatter(a['ind_acc'][a['stable']] * 100, a['importance'][a['stable']],
                    s=4, alpha=0.45, color=c,
                    label=f"{r['display']} "
                          f"($\\rho$={r['spearman_importance_vs_individual_accuracy']:.2f})")
    ax2.set_xlabel('Individual accuracy of the region (%)')
    ax2.set_ylabel(r'Model weight $|\beta_i|$')
    ax2.set_title('B  Model weight against individual discriminability',
                  loc='left', fontweight='bold', fontsize=9)
    ax2.legend(fontsize=6, frameon=False, loc='upper left')
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG_DIR, f'fig9_system_importance.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    with open(os.path.join(RESULTS_DIR, 'all_results.json')) as f:
        all_results = json.load(f)

    per_task = []
    for task in TASKS:
        r = analyse_task(task, all_results[task]['region_accs_per_fold'])
        per_task.append(r)
        print(f"{r['display']:<22} stable weak regions={r['n_stable_weak_regions']:>4}  "
              f"top={r['top_regions'][0]['region']:<10} "
              f"rho(importance, individual acc)={r['spearman_importance_vs_individual_accuracy']:+.2f}")

    make_figures(per_task)

    out = {'top_k': TOP_K, 'threshold': 0.60,
           'tasks': [{k: v for k, v in r.items() if k != '_arrays'}
                     for r in per_task]}
    path = os.path.join(RESULTS_DIR, 'interpretability.json')
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print('Written', path)


if __name__ == '__main__':
    main()
