"""
Main pipeline: region assessment, ensemble graph construction, classification.

CRITICAL FIX (2026-08-21): Region selection is now performed INSIDE the outer
CV loop using only training data, preventing information leakage from test folds.

Experiments per task:
1. All regions, direct classification (mean+std features)
2. Weak regions, direct classification (threshold-dependent, nested)
3. Weak regions, correlation graph node summaries (nested)
4. Weak regions, ensemble graph (5 features: marginal + correlation) (nested)
5. Weak regions, marginal-only ensemble (4 features: no correlation) (nested)
6. All regions, ensemble graph (nested, threshold-independent)

Threshold sensitivity: runs weak-region experiments at 55%, 60%, 65%.

Uses 4-fold subject-level cross-validation as in Algorithm 1 of the paper.
"""

import os
import sys
import json
import logging
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from joblib import Parallel, delayed

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from code.utils.data_loader import load_task_timeseries, TASKS, TASK_DISPLAY
from code.models.ensemble_graph import (
    precompute_all_features, build_ensemble_graphs_fast,
    build_correlation_graphs, compute_node_summaries
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
N_FOLDS = 4
N_JOBS = 12
ENCODING = 'LR'
THRESHOLDS = [0.55, 0.60, 0.65]
# Parcellated HCP time series (not distributed with this repository; see README).
# Override with the environment variable HCP_DATA_DIR.
DATA_DIR = os.environ.get(
    'HCP_DATA_DIR', os.path.join(PROJECT_ROOT, 'data', 'HCP_v4', 'split_data', 'split'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'metrics')

np.random.seed(RANDOM_SEED)


def assess_regions_on_subset(
    timeseries: list,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    subset_idx: np.ndarray,
    n_regions: int
) -> np.ndarray:
    """Assess classification accuracy of each region using only a subset of samples.

    Uses internal 3-fold CV on the subset to get unbiased accuracy estimates.
    """
    ts_sub = [timeseries[i] for i in subset_idx]
    lab_sub = labels[subset_idx]
    sid_sub = subject_ids[subset_idx]
    n_sub = len(ts_sub)

    means = np.zeros((n_sub, n_regions), dtype=np.float32)
    stds_arr = np.zeros((n_sub, n_regions), dtype=np.float32)
    for s in range(n_sub):
        means[s] = ts_sub[s].mean(axis=1)
        stds_arr[s] = ts_sub[s].std(axis=1)

    sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    folds = list(sgkf.split(means, lab_sub, groups=sid_sub))

    def assess_region(r):
        X = np.column_stack([means[:, r], stds_arr[:, r]])
        accs = []
        for tr, te in folds:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[tr])
            X_te = scaler.transform(X[te])
            clf = LogisticRegression(C=1.0, solver='lbfgs',
                                     max_iter=100, random_state=RANDOM_SEED)
            clf.fit(X_tr, lab_sub[tr])
            pred = clf.predict(X_te)
            accs.append(accuracy_score(lab_sub[te], pred))
        return np.mean(accs)

    region_accs = Parallel(n_jobs=N_JOBS)(
        delayed(assess_region)(r) for r in range(n_regions)
    )
    return np.array(region_accs, dtype=np.float32)


def run_ensemble_fold(
    means, stds, corrs, labels, subject_ids,
    meta_train_idx, test_idx,
    use_correlation=True, return_artifacts=False
):
    """Run ensemble graph construction for one outer fold.

    Inner CV for graph construction, then meta-model classification.
    """
    n_regions = means.shape[1]
    labels_meta_train = labels[meta_train_idx]
    labels_test = labels[test_idx]

    # Unique subject-level grouping for inner CV
    # Use index as proxy for subject grouping (already grouped at outer level)
    inner_sgkf = StratifiedGroupKFold(
        n_splits=N_FOLDS - 1, shuffle=True, random_state=RANDOM_SEED
    )
    # Subject-level grouping inside the inner loop as well: both samples of a
    # subject (one per brain state) must stay in the same inner fold, otherwise
    # the edge-level base models see the same subject they later score.
    inner_groups = subject_ids[meta_train_idx]
    inner_folds = list(inner_sgkf.split(
        np.zeros(len(meta_train_idx)),
        labels_meta_train,
        groups=inner_groups
    ))

    train_node_features = np.zeros(
        (len(meta_train_idx), n_regions), dtype=np.float32
    )

    for base_train_rel, graph_rel in inner_folds:
        base_train_idx = meta_train_idx[base_train_rel]
        graph_idx = meta_train_idx[graph_rel]

        W = build_ensemble_graphs_fast(
            means[base_train_idx], stds[base_train_idx],
            corrs[base_train_idx], labels[base_train_idx],
            means[graph_idx], stds[graph_idx], corrs[graph_idx],
            use_correlation=use_correlation
        )
        node_feats = compute_node_summaries(W)
        train_node_features[graph_rel] = node_feats

    # Build test graphs using full meta-train
    W_test = build_ensemble_graphs_fast(
        means[meta_train_idx], stds[meta_train_idx],
        corrs[meta_train_idx], labels_meta_train,
        means[test_idx], stds[test_idx], corrs[test_idx],
        use_correlation=use_correlation
    )
    test_node_features = compute_node_summaries(W_test)

    # Meta-model
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_node_features)
    X_test = scaler.transform(test_node_features)

    meta_clf = LogisticRegression(C=1.0, solver='lbfgs',
                                   max_iter=100, random_state=RANDOM_SEED)
    meta_clf.fit(X_train, labels_meta_train)

    pred = meta_clf.predict(X_test)
    prob = meta_clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(labels_test, pred)
    auc = roc_auc_score(labels_test, prob)

    if not return_artifacts:
        return acc, auc

    # Interpretability artefacts, all computed on the held-out test fold.
    # meta_coef: logistic-regression weight on the standardised node summary of
    # each region. edge_effect: difference between the class-conditional means
    # of each edge weight, i.e. how far that edge separates the two states.
    W1 = W_test[labels_test == 1].mean(axis=0)
    W0 = W_test[labels_test == 0].mean(axis=0)
    artifacts = {
        'meta_coef': meta_clf.coef_[0].astype(float).tolist(),
        'meta_intercept': float(meta_clf.intercept_[0]),
        'node_feature_std_train': scaler.scale_.astype(float).tolist(),
        'edge_effect': (W1 - W0).astype(np.float32),
        'node_effect': (W1 - W0).sum(axis=1).astype(float) / (n_regions - 1),
    }
    return acc, auc, artifacts


def save_interpretability_artifacts(task, fold_idx, weak_idx, art):
    """Persist per-fold interpretability artefacts of the weak-region synolitic
    model at the primary 60 per cent threshold (see code/analysis/interpretability.py)."""
    out_dir = os.path.join(PROJECT_ROOT, 'results', 'interpretability')
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(out_dir, f'{task}_fold{fold_idx}.npz'),
        weak_idx=np.asarray(weak_idx, dtype=np.int32),
        meta_coef=np.asarray(art['meta_coef'], dtype=np.float32),
        node_feature_std_train=np.asarray(art['node_feature_std_train'], dtype=np.float32),
        node_effect=np.asarray(art['node_effect'], dtype=np.float32),
        edge_effect=art['edge_effect'],
    )


def make_result_dict(accuracies, aucs):
    """Create a standardized result dictionary."""
    return {
        'accuracy': accuracies,
        'auc': aucs,
        'accuracy_mean': float(np.mean(accuracies)),
        'accuracy_std': float(np.std(accuracies)),
        'auc_mean': float(np.mean(aucs)),
        'auc_std': float(np.std(aucs)),
    }


def run_task_experiment(task, timeseries, labels, subject_ids):
    """Run all experiments for one task with proper nested CV.

    Region selection is performed INSIDE each outer fold using only training data.
    """
    n_samples = len(timeseries)
    n_regions = timeseries[0].shape[0]
    all_indices = np.arange(n_regions)

    # Create shared outer CV splits
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True,
                                 random_state=RANDOM_SEED)
    outer_folds = list(sgkf.split(np.zeros(n_samples), labels,
                                   groups=subject_ids))

    # Pre-compute features for ALL regions once (used across experiments)
    logger.info("  Pre-computing features for all 379 regions...")
    t0 = time.time()
    means_all, stds_all, corrs_all = precompute_all_features(
        timeseries, all_indices
    )
    logger.info(f"  Features computed in {time.time()-t0:.1f}s")

    # Also build direct feature matrix for all regions
    X_all_direct = np.column_stack([means_all, stds_all])  # (n, 758)

    task_results = {
        'task': task,
        'display': TASK_DISPLAY[task],
        'n_samples': n_samples,
        'n_regions': n_regions,
    }

    # ====================================================================
    # Experiment 1: All Direct (threshold-independent)
    # ====================================================================
    logger.info("  All Direct classification...")
    accs, aucs = [], []
    for train_idx, test_idx in outer_folds:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_all_direct[train_idx])
        X_te = scaler.transform(X_all_direct[test_idx])
        clf = LogisticRegression(C=1.0, solver='lbfgs',
                                  max_iter=100, random_state=RANDOM_SEED)
        clf.fit(X_tr, labels[train_idx])
        pred = clf.predict(X_te)
        prob = clf.predict_proba(X_te)[:, 1]
        accs.append(accuracy_score(labels[test_idx], pred))
        aucs.append(roc_auc_score(labels[test_idx], prob))
    task_results['all_direct'] = make_result_dict(accs, aucs)
    logger.info(f"    acc={np.mean(accs)*100:.2f}+/-{np.std(accs)*100:.2f}, "
                 f"auc={np.mean(aucs):.4f}")

    # ====================================================================
    # Experiment 6: All Ensemble (threshold-independent)
    # ====================================================================
    logger.info(f"  All Ensemble ({n_regions} regions)...")
    accs, aucs = [], []
    for fold_idx, (train_idx, test_idx) in enumerate(outer_folds):
        logger.info(f"    Outer fold {fold_idx+1}/{N_FOLDS} (all ensemble)")
        t0 = time.time()
        acc, auc = run_ensemble_fold(
            means_all, stds_all, corrs_all, labels, subject_ids,
            train_idx, test_idx, use_correlation=True
        )
        accs.append(acc)
        aucs.append(auc)
        logger.info(f"      acc={acc:.4f}, auc={auc:.4f} ({time.time()-t0:.1f}s)")
    task_results['all_ensemble'] = make_result_dict(accs, aucs)

    # ====================================================================
    # Experiments 2-5: Weak-region experiments at multiple thresholds
    # Region selection INSIDE outer CV (no leakage)
    # ====================================================================
    task_results['threshold_results'] = {}

    # Region assessment depends only on the outer training fold, not on the
    # threshold, so it is computed once per fold and reused for all thresholds.
    region_accs_per_fold = []
    for fold_idx, (train_idx, test_idx) in enumerate(outer_folds):
        t0 = time.time()
        region_accs_per_fold.append(assess_regions_on_subset(
            timeseries, labels, subject_ids, train_idx, n_regions))
        logger.info(f"  Region assessment fold {fold_idx+1}/{N_FOLDS} "
                    f"({time.time()-t0:.1f}s)")
    task_results['region_accs_per_fold'] = [
        a.tolist() for a in region_accs_per_fold]

    for threshold in THRESHOLDS:
        thr_key = f'thr_{int(threshold*100)}'
        logger.info(f"\n  === Threshold {threshold*100:.0f}% ===")

        thr_results = {
            'threshold': threshold,
            'weak_direct': {'accs': [], 'aucs': []},
            'weak_corr': {'accs': [], 'aucs': []},
            'weak_ensemble': {'accs': [], 'aucs': []},
            'weak_marginal': {'accs': [], 'aucs': []},
            'n_weak_per_fold': [],
            'weak_indices_per_fold': [],
        }

        for fold_idx, (train_idx, test_idx) in enumerate(outer_folds):
            logger.info(f"    Outer fold {fold_idx+1}/{N_FOLDS}")

            # Region accuracies from training data only (computed once above)
            region_accs = region_accs_per_fold[fold_idx]
            weak_idx = np.where(region_accs < threshold)[0]
            n_weak = len(weak_idx)
            thr_results['n_weak_per_fold'].append(n_weak)
            thr_results['weak_indices_per_fold'].append(weak_idx.tolist())
            logger.info(f"      {n_weak} weak regions identified")

            if n_weak < 2:
                logger.info("      Too few weak regions, skipping")
                for key in ['weak_direct', 'weak_corr', 'weak_ensemble',
                            'weak_marginal']:
                    thr_results[key]['accs'].append(np.nan)
                    thr_results[key]['aucs'].append(np.nan)
                continue

            # Pre-compute features for this fold's weak regions
            means_w, stds_w, corrs_w = precompute_all_features(
                timeseries, weak_idx
            )

            # --- Weak Direct ---
            X_w = np.column_stack([means_w, stds_w])
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_w[train_idx])
            X_te = scaler.transform(X_w[test_idx])
            clf = LogisticRegression(C=1.0, solver='lbfgs',
                                      max_iter=100, random_state=RANDOM_SEED)
            clf.fit(X_tr, labels[train_idx])
            pred = clf.predict(X_te)
            prob = clf.predict_proba(X_te)[:, 1]
            thr_results['weak_direct']['accs'].append(
                accuracy_score(labels[test_idx], pred))
            thr_results['weak_direct']['aucs'].append(
                roc_auc_score(labels[test_idx], prob))

            # --- Weak Correlation ---
            C_train = build_correlation_graphs(corrs_w[train_idx])
            train_node_f = compute_node_summaries(C_train)
            C_test = build_correlation_graphs(corrs_w[test_idx])
            test_node_f = compute_node_summaries(C_test)
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(train_node_f)
            X_te = scaler.transform(test_node_f)
            clf = LogisticRegression(C=1.0, solver='lbfgs',
                                      max_iter=100, random_state=RANDOM_SEED)
            clf.fit(X_tr, labels[train_idx])
            pred = clf.predict(X_te)
            prob = clf.predict_proba(X_te)[:, 1]
            thr_results['weak_corr']['accs'].append(
                accuracy_score(labels[test_idx], pred))
            thr_results['weak_corr']['aucs'].append(
                roc_auc_score(labels[test_idx], prob))

            # --- Weak Ensemble (full 5 features) ---
            t0 = time.time()
            acc, auc, art = run_ensemble_fold(
                means_w, stds_w, corrs_w, labels, subject_ids,
                train_idx, test_idx, use_correlation=True,
                return_artifacts=True
            )
            if abs(threshold - 0.60) < 1e-9:
                save_interpretability_artifacts(task, fold_idx, weak_idx, art)
            thr_results['weak_ensemble']['accs'].append(acc)
            thr_results['weak_ensemble']['aucs'].append(auc)
            logger.info(f"      Weak Ensemble: acc={acc:.4f} ({time.time()-t0:.1f}s)")

            # --- Weak Marginal-only (4 features, no correlation) ---
            t0 = time.time()
            acc, auc = run_ensemble_fold(
                means_w, stds_w, corrs_w, labels, subject_ids,
                train_idx, test_idx, use_correlation=False
            )
            thr_results['weak_marginal']['accs'].append(acc)
            thr_results['weak_marginal']['aucs'].append(auc)
            logger.info(f"      Weak Marginal: acc={acc:.4f} ({time.time()-t0:.1f}s)")

        # Summarize threshold results
        for key in ['weak_direct', 'weak_corr', 'weak_ensemble', 'weak_marginal']:
            valid = [a for a in thr_results[key]['accs'] if not np.isnan(a)]
            valid_auc = [a for a in thr_results[key]['aucs'] if not np.isnan(a)]
            thr_results[key] = make_result_dict(valid, valid_auc)

        thr_results['n_weak_median'] = int(np.median(thr_results['n_weak_per_fold']))
        thr_results['n_weak_range'] = [
            int(min(thr_results['n_weak_per_fold'])),
            int(max(thr_results['n_weak_per_fold']))
        ]

        task_results['threshold_results'][thr_key] = thr_results

        logger.info(f"  Threshold {threshold*100:.0f}% summary: "
                     f"n_weak={thr_results['n_weak_median']} "
                     f"[{thr_results['n_weak_range'][0]}-"
                     f"{thr_results['n_weak_range'][1]}]")

    # For backward compatibility: copy 60% threshold results to top level
    thr60 = task_results['threshold_results']['thr_60']
    task_results['n_weak'] = thr60['n_weak_median']
    task_results['n_weak_range'] = thr60['n_weak_range']
    task_results['weak_direct'] = thr60['weak_direct']
    task_results['weak_corr'] = thr60['weak_corr']
    task_results['weak_ensemble'] = thr60['weak_ensemble']
    task_results['weak_marginal'] = thr60['weak_marginal']

    return task_results


def main():
    """Run the full experimental pipeline for all 7 tasks."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = {}

    for task in TASKS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Task: {TASK_DISPLAY[task]}")
        logger.info(f"{'='*60}")

        # Load data
        timeseries, labels, subject_ids = load_task_timeseries(
            DATA_DIR, task, encoding=ENCODING
        )
        n_samples = len(timeseries)
        n_regions = timeseries[0].shape[0]
        logger.info(f"  {n_samples} samples, {n_regions} regions, "
                     f"classes: {np.bincount(labels)}")

        task_results = run_task_experiment(task, timeseries, labels, subject_ids)
        all_results[task] = task_results

        # Save per-task results
        with open(os.path.join(RESULTS_DIR, f'results_{task}.json'), 'w') as f:
            json.dump(task_results, f, indent=2, default=int)

    # Save all results
    with open(os.path.join(RESULTS_DIR, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=int)

    print_summary(all_results)


def print_summary(all_results: dict) -> None:
    """Print summary table."""
    print("\n" + "=" * 130)
    print(f"{'Task':<20} {'N_weak':>8} {'All Dir':>14} "
          f"{'Wk Dir':>14} {'Wk Corr':>14} {'Wk Ens':>14} "
          f"{'Wk Marg':>14} {'All Ens':>14}")
    print("-" * 130)
    for task in TASKS:
        if task not in all_results:
            continue
        r = all_results[task]
        nw = r.get('n_weak', '?')
        nw_range = r.get('n_weak_range', [nw, nw])
        parts = [f"{TASK_DISPLAY[task]:<20}",
                 f"{nw} [{nw_range[0]}-{nw_range[1]}]".rjust(8)]

        for key in ['all_direct', 'weak_direct', 'weak_corr',
                     'weak_ensemble', 'weak_marginal', 'all_ensemble']:
            d = r.get(key)
            if d and 'accuracy_mean' in d:
                parts.append(
                    f"{d['accuracy_mean']*100:>6.2f}({d['accuracy_std']*100:.2f})")
            else:
                parts.append(f"{'N/A':>14}")
        print(" ".join(parts))
    print("=" * 130)

    # Also print AUC
    print("\nAUC values:")
    print("-" * 100)
    for task in TASKS:
        if task not in all_results:
            continue
        r = all_results[task]
        parts = [f"{TASK_DISPLAY[task]:<20}"]
        for key in ['all_direct', 'weak_direct', 'weak_corr',
                     'weak_ensemble', 'weak_marginal', 'all_ensemble']:
            d = r.get(key)
            if d and 'auc_mean' in d:
                parts.append(f"{d['auc_mean']:.4f}")
            else:
                parts.append("N/A")
        print("  ".join(parts))

    # Threshold sensitivity
    print("\n\nThreshold Sensitivity (Weak Ensemble accuracy):")
    print("-" * 80)
    for task in TASKS:
        if task not in all_results:
            continue
        r = all_results[task]
        parts = [f"{TASK_DISPLAY[task]:<20}"]
        for thr_key in ['thr_55', 'thr_60', 'thr_65']:
            thr = r.get('threshold_results', {}).get(thr_key)
            if thr:
                we = thr['weak_ensemble']
                nw = thr['n_weak_median']
                parts.append(f"{we['accuracy_mean']*100:.2f}% (n={nw})")
            else:
                parts.append("N/A")
        print("  ".join(parts))


if __name__ == '__main__':
    main()
