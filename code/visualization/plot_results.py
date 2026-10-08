"""
Generate figures for the HPC ensemble graph paper.

Figures:
1. Bar chart comparing classification accuracy across methods and tasks
2. Improvement of ensemble over baselines (direct and correlation)
3. Region-level accuracy distribution per task (from one representative fold)
4. Stacked bar showing weak/strong region counts per task
5. Threshold sensitivity analysis
6. Marginal-only vs full ensemble comparison
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

TASK_DISPLAY = {
    'WM': 'Working Memory',
    'GAMBLING': 'Gambling',
    'MOTOR': 'Motor Activity',
    'LANGUAGE': 'Language Processing',
    'SOCIAL': 'Social Cognition',
    'RELATIONAL': 'Relational Processing',
    'EMOTION': 'Emotion Processing',
}

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'metrics')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures')

TASK_ORDER = ['WM', 'GAMBLING', 'MOTOR', 'LANGUAGE', 'SOCIAL', 'RELATIONAL', 'EMOTION']
TASK_SHORT = {
    'WM': 'Working\nMemory',
    'GAMBLING': 'Gambling',
    'MOTOR': 'Motor',
    'LANGUAGE': 'Language',
    'SOCIAL': 'Social\nCognition',
    'RELATIONAL': 'Relational',
    'EMOTION': 'Emotion',
}


def load_results() -> dict:
    results_file = os.path.join(RESULTS_DIR, 'all_results.json')
    with open(results_file, 'r') as f:
        return json.load(f)


def save_fig(fig, name):
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(FIGURES_DIR, f'{name}.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {name}')


def plot_main_comparison(results: dict) -> None:
    """Figure 1: Bar chart comparing methods across tasks (including marginal)."""
    fig, ax = plt.subplots(figsize=(16, 6))

    tasks = [t for t in TASK_ORDER if t in results]
    n_tasks = len(tasks)
    x = np.arange(n_tasks)
    width = 0.14

    methods = [
        ('all_direct', 'All regions\n(direct)', '#2196F3'),
        ('weak_direct', 'LP regions\n(direct)', '#FF9800'),
        ('weak_corr', 'LP regions\n(correlation)', '#F44336'),
        ('weak_ensemble', 'LP regions\n(ensemble)', '#4CAF50'),
        ('weak_marginal', 'LP regions\n(marginal-only)', '#8BC34A'),
        ('all_ensemble', 'All regions\n(ensemble)', '#9C27B0'),
    ]

    for i, (key, label, color) in enumerate(methods):
        means = []
        stds = []
        for task in tasks:
            r = results[task].get(key)
            if r and 'accuracy_mean' in r:
                means.append(r['accuracy_mean'] * 100)
                stds.append(r['accuracy_std'] * 100)
            else:
                means.append(0)
                stds.append(0)

        ax.bar(x + i * width - 2.5 * width, means, width,
               yerr=stds, label=label, color=color, alpha=0.85,
               capsize=2, edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Task', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Classification Accuracy Across Methods and Tasks', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_SHORT[t] for t in tasks], fontsize=10)
    ax.legend(fontsize=8, loc='lower left', ncol=3)
    ax.set_ylim(40, 105)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_fig(fig, 'fig1_main_comparison')


def plot_ensemble_improvement(results: dict) -> None:
    """Figure 2: Improvement of ensemble over baselines."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    tasks = [t for t in TASK_ORDER if t in results
             and results[t].get('weak_ensemble')]
    task_labels = [TASK_SHORT[t].replace('\n', ' ') for t in tasks]

    # Improvement over direct
    imp_direct = []
    for t in tasks:
        r = results[t]
        ens = r['weak_ensemble']['accuracy_mean'] * 100
        direct = r['weak_direct']['accuracy_mean'] * 100
        imp_direct.append(ens - direct)

    colors_direct = ['#4CAF50' if v >= 0 else '#F44336' for v in imp_direct]
    axes[0].barh(task_labels, imp_direct, color=colors_direct, alpha=0.8)
    axes[0].set_xlabel('Accuracy improvement (pp)', fontsize=11)
    axes[0].set_title('Ensemble vs. Direct\n(low-performing regions)', fontsize=12)
    axes[0].axvline(x=0, color='black', linewidth=0.5)
    axes[0].grid(axis='x', alpha=0.3)

    # Improvement over correlation
    imp_corr = []
    for t in tasks:
        r = results[t]
        ens = r['weak_ensemble']['accuracy_mean'] * 100
        corr = r['weak_corr']['accuracy_mean'] * 100
        imp_corr.append(ens - corr)

    colors_corr = ['#4CAF50' if v >= 0 else '#F44336' for v in imp_corr]
    axes[1].barh(task_labels, imp_corr, color=colors_corr, alpha=0.8)
    axes[1].set_xlabel('Accuracy improvement (pp)', fontsize=11)
    axes[1].set_title('Ensemble vs. Correlation\n(low-performing regions)', fontsize=12)
    axes[1].axvline(x=0, color='black', linewidth=0.5)
    axes[1].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    save_fig(fig, 'fig2_improvement')


def plot_region_assessment(results: dict) -> None:
    """Figure 3: distribution of individual region accuracy per task.

    Accuracies come from the in-fold region assessment of the first outer fold
    (training subjects only), which is the quantity the selection acts on.
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    tasks = [t for t in TASK_ORDER if t in results]

    for idx, task in enumerate(tasks):
        ax = axes[idx]
        accs = np.asarray(results[task]['region_accs_per_fold'][0]) * 100
        n_weak = int((accs < 60).sum())
        ax.hist(accs, bins=40, color='#4C72B0', alpha=0.85)
        ax.axvline(60, color='#C44E52', ls='--', lw=1.5)
        ax.set_title(TASK_DISPLAY[task], fontsize=11)
        ax.set_xlabel('Individual region accuracy (%)', fontsize=9)
        ax.set_ylabel('Number of regions', fontsize=9)
        ax.text(0.03, 0.95, f'{n_weak} of {len(accs)} below 60%',
                transform=ax.transAxes, va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    if len(tasks) < 8:
        axes[7].set_visible(False)

    plt.suptitle('Individual classification accuracy of each parcel '
                 '(outer fold 1, training subjects only)', fontsize=14, y=1.02)
    plt.tight_layout()
    save_fig(fig, 'fig3_region_assessment')


def plot_weak_vs_strong_count(results: dict) -> None:
    """Figure 4: Stacked bar showing weak/strong region counts per task."""
    fig, ax = plt.subplots(figsize=(10, 5))

    tasks = [t for t in TASK_ORDER if t in results]
    n_weak = []
    n_strong = []
    for t in tasks:
        thr_data = results[t].get('threshold_results', {}).get('thr_60', {})
        nw = thr_data.get('n_weak_median', results[t].get('n_weak', 0))
        n_weak.append(nw)
        n_strong.append(379 - nw)
    labels = [TASK_SHORT[t] for t in tasks]

    x = np.arange(len(tasks))
    ax.bar(x, n_weak, label='Low-performing (<60%)', color='#FF9800', alpha=0.85)
    ax.bar(x, n_strong, bottom=n_weak, label='High-performing (>=60%)',
           color='#2196F3', alpha=0.85)

    ax.set_xlabel('Task', fontsize=12)
    ax.set_ylabel('Number of regions', fontsize=12)
    ax.set_title('Low-performing vs. High-performing Regions per Task (out of 379)',
                 fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(fontsize=10)
    ax.axhline(y=379, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylim(0, 420)

    for i, (w, s) in enumerate(zip(n_weak, n_strong)):
        ax.text(i, w / 2, str(w), ha='center', va='center', fontsize=9,
                fontweight='bold', color='white')

    plt.tight_layout()
    save_fig(fig, 'fig4_region_counts')


def plot_threshold_sensitivity(results: dict) -> None:
    """Figure 5: Threshold sensitivity analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    tasks = [t for t in TASK_ORDER if t in results]
    thresholds = ['thr_55', 'thr_60', 'thr_65']
    thr_labels = ['55%', '60%', '65%']

    # Left panel: ensemble accuracy at each threshold
    x = np.arange(len(tasks))
    width = 0.25
    colors = ['#42A5F5', '#4CAF50', '#FF7043']

    for i, (thr_key, thr_label, color) in enumerate(zip(thresholds, thr_labels, colors)):
        accs = []
        for t in tasks:
            thr_data = results[t].get('threshold_results', {}).get(thr_key, {})
            we = thr_data.get('weak_ensemble', {})
            accs.append(we.get('accuracy_mean', 0) * 100)
        axes[0].bar(x + i * width - width, accs, width,
                    label=f'Threshold {thr_label}', color=color, alpha=0.85)

    axes[0].set_xlabel('Task', fontsize=11)
    axes[0].set_ylabel('Ensemble Accuracy (%)', fontsize=11)
    axes[0].set_title('Ensemble Accuracy at Different Thresholds', fontsize=13)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([TASK_SHORT[t] for t in tasks], fontsize=9)
    axes[0].legend(fontsize=9)
    axes[0].set_ylim(40, 105)
    axes[0].grid(axis='y', alpha=0.3)

    # Right panel: number of selected regions
    for i, (thr_key, thr_label, color) in enumerate(zip(thresholds, thr_labels, colors)):
        ns = []
        for t in tasks:
            thr_data = results[t].get('threshold_results', {}).get(thr_key, {})
            ns.append(thr_data.get('n_weak_median', 0))
        axes[1].bar(x + i * width - width, ns, width,
                    label=f'Threshold {thr_label}', color=color, alpha=0.85)

    axes[1].set_xlabel('Task', fontsize=11)
    axes[1].set_ylabel('Number of regions selected', fontsize=11)
    axes[1].set_title('Low-performing Regions at Different Thresholds', fontsize=13)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([TASK_SHORT[t] for t in tasks], fontsize=9)
    axes[1].legend(fontsize=9)
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_fig(fig, 'fig5_threshold_sensitivity')


def plot_marginal_comparison(results: dict) -> None:
    """Figure 6: Full ensemble vs marginal-only ensemble."""
    fig, ax = plt.subplots(figsize=(12, 5))

    tasks = [t for t in TASK_ORDER if t in results
             and results[t].get('weak_marginal')]
    task_labels = [TASK_SHORT[t] for t in tasks]
    x = np.arange(len(tasks))
    width = 0.35

    full_accs = [results[t]['weak_ensemble']['accuracy_mean'] * 100 for t in tasks]
    marg_accs = [results[t]['weak_marginal']['accuracy_mean'] * 100 for t in tasks]

    ax.bar(x - width/2, full_accs, width, label='Full ensemble (5 features)',
           color='#4CAF50', alpha=0.85)
    ax.bar(x + width/2, marg_accs, width, label='Marginal-only (4 features)',
           color='#8BC34A', alpha=0.85)

    ax.set_xlabel('Task', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Full Ensemble vs. Marginal-only Ensemble\n(low-performing regions, 60% threshold)',
                 fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(40, 105)
    ax.grid(axis='y', alpha=0.3)

    # Add difference annotations
    for i, (f, m) in enumerate(zip(full_accs, marg_accs)):
        diff = f - m
        color = '#4CAF50' if diff > 0 else '#F44336'
        ax.annotate(f'{diff:+.1f}pp', xy=(i, max(f, m) + 1),
                    ha='center', fontsize=8, color=color, fontweight='bold')

    plt.tight_layout()
    save_fig(fig, 'fig6_marginal_comparison')


def plot_gsr_comparison(results: dict) -> None:
    """Figure 10: what global signal regression does to the three comparisons."""
    gsr_path = os.path.join(RESULTS_DIR, 'gsr_sensitivity.json')
    if not os.path.exists(gsr_path):
        print('Skipping fig10: gsr_sensitivity.json not found')
        return
    with open(gsr_path) as f:
        gsr = json.load(f)

    tasks = [t for t in TASK_ORDER if t in results and t in gsr]
    x = np.arange(len(tasks))
    width = 0.26

    def acc(d, t, key):
        return d[t][key]['accuracy_mean'] * 100

    panels = [
        ('A  As distributed', results, 'n_weak'),
        ('B  After global signal regression', gsr, 'n_weak_median'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4),
                             gridspec_kw={'width_ratios': [1.35, 1.35, 1]})

    series = [('vs. correlation graph', 'weak_corr', '#8D6E63'),
              ('vs. direct classification', 'weak_direct', '#1565C0'),
              ('vs. marginal-only edges', 'weak_marginal', '#EF6C00')]

    for ax, (title, data, _) in zip(axes[:2], panels):
        for k, (label, key, colour) in enumerate(series):
            vals = [acc(data, t, 'weak_ensemble') - acc(data, t, key) for t in tasks]
            ax.bar(x + (k - 1) * width, vals, width, label=label, color=colour)
        ax.axhline(0, color='0.2', lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([TASK_SHORT[t] for t in tasks], fontsize=8)
        ax.set_ylim(-8, 34)
        ax.set_ylabel('Synolitic advantage (percentage points)', fontsize=10)
        ax.set_title(title, loc='left', fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    axes[0].legend(fontsize=8, loc='upper right')

    ax = axes[2]
    n_plain = [results[t]['n_weak'] for t in tasks]
    n_gsr = [gsr[t]['n_weak_median'] for t in tasks]
    ax.bar(x - 0.2, n_plain, 0.4, label='As distributed', color='#B0BEC5')
    ax.bar(x + 0.2, n_gsr, 0.4, label='After GSR', color='#37474F')
    ax.axhline(379, color='0.3', ls='--', lw=1)
    ax.text(len(tasks) - 0.5, 388, 'all 379 parcels', fontsize=7, ha='right')
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace('\n', ' ') for t in
                        (TASK_SHORT[t] for t in tasks)], fontsize=7.5,
                       rotation=40, ha='right')
    ax.set_ylim(0, 430)
    ax.set_ylabel('Weak parcels (median over folds)', fontsize=10)
    ax.set_title('C  Individually weak parcels', loc='left', fontsize=11,
                 fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_fig(fig, 'fig10_gsr_comparison')


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    results = load_results()

    plot_main_comparison(results)
    plot_ensemble_improvement(results)
    plot_region_assessment(results)
    plot_weak_vs_strong_count(results)
    plot_threshold_sensitivity(results)
    plot_marginal_comparison(results)
    plot_gsr_comparison(results)

    print('\nAll figures generated successfully.')


if __name__ == '__main__':
    main()
