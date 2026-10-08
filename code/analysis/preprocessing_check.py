"""Empirical verification of the state of the HCP parcellated time series.

The distributed data set (data/HCP_v4/split_data/split) contains parcellated,
condition-split BOLD time series. Its ReadMe documents the atlas and the
parcellation library but not the temporal preprocessing that preceded it.
Two properties matter for the present study and are therefore verified from
the data itself rather than assumed:

  1. Whether regional time courses were standardised (z-scored) over the run.
  2. Whether global signal regression (GSR) was applied.

GSR is algebraically detectable: if every grayordinate (and hence every
parcel) time course has had the whole-brain mean signal regressed out, then
the residual time courses are orthogonal to that mean by construction, so
  - the correlation of every parcel with the global signal is ~0,
  - the mean of the off-diagonal parcel-by-parcel correlation matrix is
    slightly negative (it is exactly -1/(N-1) for an idealised full-rank case),
  - close to half of all edges carry a negative correlation.
Without GSR, all three quantities are markedly positive.

Outputs
    results/metrics/preprocessing_check.json
    figures/fig7_preprocessing_check.{pdf,png}

Usage
    python code/analysis/preprocessing_check.py
"""

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from code.utils.data_loader import TASKS, TASK_DISPLAY  # noqa: E402

# Parcellated HCP time series (not distributed with this repository; see README).
# Override with the environment variable HCP_DATA_DIR.
DATA_DIR = os.environ.get(
    'HCP_DATA_DIR', os.path.join(PROJECT_ROOT, 'data', 'HCP_v4', 'split_data', 'split'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'metrics')
FIG_DIR = os.path.join(PROJECT_ROOT, 'figures')
ENCODING = 'LR'
N_SUBJECTS = 60          # subjects sampled per task
RANDOM_SEED = 42


def load_run(task, subject):
    """Concatenate both condition files of one subject into a single matrix.

    Returns (n_regions, T) or None when a condition file is missing.
    """
    label_0, label_1 = TASKS[task]
    blocks = []
    for lab in (label_0, label_1):
        path = os.path.join(DATA_DIR, task, f'{subject}_{ENCODING}_{lab}.pickle')
        if not os.path.exists(path):
            return None
        with open(path, 'rb') as f:
            blocks.extend(pickle.load(f))
    ts = pd.concat(blocks, axis=1).values.astype(np.float64)
    return ts


def subjects_for_task(task):
    task_dir = os.path.join(DATA_DIR, task)
    subs = sorted({f.split('_')[0] for f in os.listdir(task_dir)
                   if f.endswith('.pickle') and f.split('_')[1] == ENCODING})
    return subs


def analyse_task(task):
    rng = np.random.default_rng(RANDOM_SEED)
    subs = subjects_for_task(task)
    sel = rng.choice(len(subs), size=min(N_SUBJECTS, len(subs)), replace=False)
    sel = [subs[i] for i in sorted(sel)]

    region_means, region_stds = [], []
    gs_corr, edge_corr_mean, edge_frac_neg = [], [], []
    gs_sd_ratio, pc1_var = [], []
    pc1_same_sign, pc1_gs = [], []
    gs_corr_post, edge_corr_mean_post, edge_frac_neg_post = [], [], []
    n_used, n_timepoints = 0, []

    for sub in sel:
        ts = load_run(task, sub)
        if ts is None:
            continue
        n_used += 1
        n_timepoints.append(ts.shape[1])

        region_means.append(ts.mean(axis=1))
        sd = ts.std(axis=1)
        region_stds.append(sd)

        # global signal and its coupling with each parcel
        gs = ts.mean(axis=0)
        tsc = ts - ts.mean(axis=1, keepdims=True)
        gsc = gs - gs.mean()
        denom = np.sqrt((tsc ** 2).sum(axis=1) * (gsc ** 2).sum())
        with np.errstate(invalid='ignore', divide='ignore'):
            r_gs = (tsc @ gsc) / denom
        gs_corr.append(np.nan_to_num(r_gs))
        gs_sd_ratio.append(gs.std() / np.median(sd))

        # off-diagonal correlation structure
        c = np.corrcoef(ts)
        iu = np.triu_indices(c.shape[0], k=1)
        vals = np.nan_to_num(c[iu])
        edge_corr_mean.append(vals.mean())
        edge_frac_neg.append(float((vals < 0).mean()))

        # share of temporal variance in the leading principal component, and
        # whether that component is a global mode (uniform-sign loadings, time
        # course aligned with the global signal). GSR removes exactly this mode.
        x = tsc / np.sqrt(max(ts.shape[1] - 1, 1))
        u, sv, vt = np.linalg.svd(x, full_matrices=False)
        pc1_var.append(float(sv[0] ** 2 / (sv ** 2).sum()))
        load = u[:, 0]
        pc1_same_sign.append(float(max((load > 0).mean(), (load < 0).mean())))
        pc1_gs.append(float(abs(np.corrcoef(vt[0], gsc)[0, 1])))

        # positive control for the detector: apply GSR here and re-measure
        beta = (tsc @ gsc) / (gsc @ gsc)
        res = tsc - np.outer(beta, gsc)
        r_gs_post = np.nan_to_num(
            (res @ gsc) / np.sqrt(np.maximum((res ** 2).sum(axis=1), 1e-12) * (gsc @ gsc)))
        gs_corr_post.append(np.median(np.abs(r_gs_post)))
        c_post = np.corrcoef(res)
        vals_post = np.nan_to_num(c_post[np.triu_indices(c_post.shape[0], k=1)])
        edge_corr_mean_post.append(vals_post.mean())
        edge_frac_neg_post.append(float((vals_post < 0).mean()))

    region_means = np.concatenate(region_means)
    region_stds = np.concatenate(region_stds)
    gs_corr = np.concatenate(gs_corr)

    return {
        'task': task,
        'display': TASK_DISPLAY[task],
        'n_subjects_analysed': n_used,
        'timepoints_median': float(np.median(n_timepoints)),
        'timepoints_range': [int(np.min(n_timepoints)), int(np.max(n_timepoints))],
        'region_mean_median': float(np.median(region_means)),
        'region_mean_abs_p95': float(np.percentile(np.abs(region_means), 95)),
        'region_sd_median': float(np.median(region_stds)),
        'region_sd_iqr': [float(np.percentile(region_stds, 25)),
                          float(np.percentile(region_stds, 75))],
        'gs_corr_median': float(np.median(gs_corr)),
        'gs_corr_p5': float(np.percentile(gs_corr, 5)),
        'gs_corr_p95': float(np.percentile(gs_corr, 95)),
        'gs_corr_frac_above_0p3': float((gs_corr > 0.3).mean()),
        'gs_sd_over_region_sd_median': float(np.median(gs_sd_ratio)),
        'edge_corr_mean': float(np.mean(edge_corr_mean)),
        'edge_corr_mean_sd': float(np.std(edge_corr_mean)),
        'edge_frac_negative': float(np.mean(edge_frac_neg)),
        'pc1_variance_share_median': float(np.median(pc1_var)),
        'pc1_same_sign_loading_frac_median': float(np.median(pc1_same_sign)),
        'pc1_global_signal_corr_median': float(np.median(pc1_gs)),
        'after_posthoc_gsr': {
            'gs_corr_abs_median': float(np.median(gs_corr_post)),
            'edge_corr_mean': float(np.mean(edge_corr_mean_post)),
            'edge_frac_negative': float(np.mean(edge_frac_neg_post)),
        },
        '_dist': {'gs_corr': gs_corr, 'region_mean': region_means,
                  'region_sd': region_stds},
    }


def make_figure(per_task):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(per_task)))

    for c, r in zip(colors, per_task):
        axes[0].hist(r['_dist']['region_mean'], bins=80, range=(-0.6, 0.6),
                     histtype='step', density=True, color=c, label=r['display'])
        axes[1].hist(r['_dist']['region_sd'], bins=80, range=(0, 2.0),
                     histtype='step', density=True, color=c)
        axes[2].hist(r['_dist']['gs_corr'], bins=80, range=(-1, 1),
                     histtype='step', density=True, color=c)

    axes[0].set_xlabel('Parcel mean over the task blocks (a.u.)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('A  Parcel means', loc='left', fontweight='bold')
    axes[0].axvline(0, color='0.4', lw=0.8, ls='--')

    axes[1].set_xlabel('Parcel standard deviation (a.u.)')
    axes[1].set_title('B  Parcel standard deviations', loc='left', fontweight='bold')
    axes[1].axvline(1.0, color='0.4', lw=0.8, ls='--')

    axes[2].set_xlabel('Correlation of parcel with global signal')
    axes[2].set_title('C  Global signal coupling', loc='left', fontweight='bold')
    axes[2].axvline(0, color='crimson', lw=1.0, ls='--')
    axes[2].text(0.02, 0.92, 'value expected under GSR', color='crimson',
                 transform=axes[2].transAxes, fontsize=8)

    axes[0].legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(FIG_DIR, f'fig7_preprocessing_check.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    per_task = []
    for task in TASKS:
        r = analyse_task(task)
        per_task.append(r)
        print(f"{r['display']:<22} parcel mean median={r['region_mean_median']:+.4f} "
              f"sd median={r['region_sd_median']:.3f} "
              f"r(parcel,GS) median={r['gs_corr_median']:+.3f} "
              f"mean edge r={r['edge_corr_mean']:+.3f} "
              f"neg edges={r['edge_frac_negative']*100:.1f}% "
              f"PC1={r['pc1_variance_share_median']*100:.1f}% "
              f"(|r| with GS={r['pc1_global_signal_corr_median']:.2f}, "
              f"same-sign loadings={r['pc1_same_sign_loading_frac_median']*100:.0f}%) | "
              f"post-hoc GSR: r(parcel,GS)={r['after_posthoc_gsr']['gs_corr_abs_median']:.1e}, "
              f"mean edge r={r['after_posthoc_gsr']['edge_corr_mean']:+.3f}, "
              f"neg={r['after_posthoc_gsr']['edge_frac_negative']*100:.1f}%")

    make_figure(per_task)

    summary = {
        'n_subjects_per_task': N_SUBJECTS,
        'encoding': ENCODING,
        'pooled': {
            'gs_corr_median': float(np.median([r['gs_corr_median'] for r in per_task])),
            'edge_corr_mean': float(np.mean([r['edge_corr_mean'] for r in per_task])),
            'edge_frac_negative': float(np.mean([r['edge_frac_negative'] for r in per_task])),
            'region_sd_median': float(np.median([r['region_sd_median'] for r in per_task])),
            'region_mean_median': float(np.median([r['region_mean_median'] for r in per_task])),
            'pc1_variance_share_median': float(np.median(
                [r['pc1_variance_share_median'] for r in per_task])),
            'pc1_global_signal_corr_median': float(np.median(
                [r['pc1_global_signal_corr_median'] for r in per_task])),
            'pc1_same_sign_loading_frac_median': float(np.median(
                [r['pc1_same_sign_loading_frac_median'] for r in per_task])),
            'posthoc_gsr_gs_corr_abs_median': float(np.median(
                [r['after_posthoc_gsr']['gs_corr_abs_median'] for r in per_task])),
            'posthoc_gsr_edge_corr_mean': float(np.mean(
                [r['after_posthoc_gsr']['edge_corr_mean'] for r in per_task])),
            'posthoc_gsr_edge_frac_negative': float(np.mean(
                [r['after_posthoc_gsr']['edge_frac_negative'] for r in per_task])),
        },
        'tasks': [{k: v for k, v in r.items() if k != '_dist'} for r in per_task],
    }
    out = os.path.join(RESULTS_DIR, 'preprocessing_check.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)
    print('\nPooled:', json.dumps(summary['pooled'], indent=2))
    print('Written', out)


if __name__ == '__main__':
    main()
