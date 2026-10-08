"""Robustness of the main comparison to global signal regression (GSR).

The distributed parcellated time series retain a positive global mode
(code/analysis/preprocessing_check.py). Because GSR is one of the most
consequential and most contested choices in fMRI preprocessing, we repeat the
primary experiment on the same data after regressing the global signal out of
every parcel time course, and report both versions side by side.

For each sample the global signal is the mean over all 379 parcels of that
sample's own time points; each parcel time course is replaced by the residual
of its least-squares regression on that global signal. The operation is
strictly within-sample, so it cannot move information between subjects or
between cross-validation folds.

Everything else - the outer folds, the in-fold region assessment, the 60 per
cent threshold, the nested graph construction and the meta-model - is
identical to code/models/run_pipeline.py.

Outputs
    results/metrics/gsr_sensitivity.json

Usage
    python code/analysis/gsr_sensitivity.py [TASK ...]
"""

import json
import os
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import code.models.run_pipeline as rp  # noqa: E402
from code.utils.data_loader import load_task_timeseries, TASKS, TASK_DISPLAY  # noqa: E402
from code.models.ensemble_graph import (  # noqa: E402
    precompute_all_features, build_correlation_graphs, compute_node_summaries)

rp.N_JOBS = 4                      # leave cores for the main pipeline
THRESHOLD = 0.60
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'metrics')
OUT_PATH = os.path.join(RESULTS_DIR, 'gsr_sensitivity.json')


def apply_gsr(timeseries):
    """Regress each sample's own global signal out of every parcel."""
    out = []
    for ts in timeseries:
        x = ts.astype(np.float64)
        xc = x - x.mean(axis=1, keepdims=True)
        gs = xc.mean(axis=0)
        denom = gs @ gs
        if denom <= 0:
            out.append(x.astype(np.float32))
            continue
        beta = (xc @ gs) / denom
        out.append((xc - np.outer(beta, gs)).astype(np.float32))
    return out


def run_task(task):
    timeseries, labels, subject_ids = load_task_timeseries(
        rp.DATA_DIR, task, encoding=rp.ENCODING)
    timeseries = apply_gsr(timeseries)
    n_samples = len(timeseries)
    n_regions = timeseries[0].shape[0]

    sgkf = StratifiedGroupKFold(n_splits=rp.N_FOLDS, shuffle=True,
                                random_state=rp.RANDOM_SEED)
    outer_folds = list(sgkf.split(np.zeros(n_samples), labels, groups=subject_ids))

    res = {k: {'accs': [], 'aucs': []} for k in
           ['weak_direct', 'weak_corr', 'weak_ensemble', 'weak_marginal']}
    n_weak_per_fold, weak_idx_per_fold = [], []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_folds):
        t0 = time.time()
        region_accs = rp.assess_regions_on_subset(
            timeseries, labels, subject_ids, train_idx, n_regions)
        weak_idx = np.where(region_accs < THRESHOLD)[0]
        n_weak_per_fold.append(int(len(weak_idx)))
        weak_idx_per_fold.append(weak_idx.tolist())
        print(f"  [{task}] fold {fold_idx+1}: {len(weak_idx)} weak regions "
              f"({time.time()-t0:.0f}s)", flush=True)
        if len(weak_idx) < 2:
            for k in res:
                res[k]['accs'].append(np.nan)
                res[k]['aucs'].append(np.nan)
            continue

        means_w, stds_w, corrs_w = precompute_all_features(timeseries, weak_idx)

        X_w = np.column_stack([means_w, stds_w])
        scaler = StandardScaler()
        clf = LogisticRegression(C=1.0, solver='lbfgs', max_iter=100,
                                 random_state=rp.RANDOM_SEED)
        clf.fit(scaler.fit_transform(X_w[train_idx]), labels[train_idx])
        Xte = scaler.transform(X_w[test_idx])
        res['weak_direct']['accs'].append(
            accuracy_score(labels[test_idx], clf.predict(Xte)))
        res['weak_direct']['aucs'].append(
            roc_auc_score(labels[test_idx], clf.predict_proba(Xte)[:, 1]))

        tr_f = compute_node_summaries(build_correlation_graphs(corrs_w[train_idx]))
        te_f = compute_node_summaries(build_correlation_graphs(corrs_w[test_idx]))
        scaler = StandardScaler()
        clf = LogisticRegression(C=1.0, solver='lbfgs', max_iter=100,
                                 random_state=rp.RANDOM_SEED)
        clf.fit(scaler.fit_transform(tr_f), labels[train_idx])
        Xte = scaler.transform(te_f)
        res['weak_corr']['accs'].append(
            accuracy_score(labels[test_idx], clf.predict(Xte)))
        res['weak_corr']['aucs'].append(
            roc_auc_score(labels[test_idx], clf.predict_proba(Xte)[:, 1]))

        for key, use_corr in (('weak_ensemble', True), ('weak_marginal', False)):
            t1 = time.time()
            acc, auc = rp.run_ensemble_fold(
                means_w, stds_w, corrs_w, labels, subject_ids,
                train_idx, test_idx, use_correlation=use_corr)
            res[key]['accs'].append(acc)
            res[key]['aucs'].append(auc)
            print(f"  [{task}] fold {fold_idx+1} {key}: acc={acc:.4f} "
                  f"({time.time()-t1:.0f}s)", flush=True)

    out = {'task': task, 'display': TASK_DISPLAY[task], 'threshold': THRESHOLD,
           'n_weak_per_fold': n_weak_per_fold,
           'n_weak_median': int(np.median(n_weak_per_fold)),
           'n_weak_range': [int(min(n_weak_per_fold)), int(max(n_weak_per_fold))],
           'weak_indices_per_fold': weak_idx_per_fold}
    for k in res:
        valid = [a for a in res[k]['accs'] if not np.isnan(a)]
        valid_auc = [a for a in res[k]['aucs'] if not np.isnan(a)]
        out[k] = rp.make_result_dict(valid, valid_auc)
    return out


def main():
    tasks = sys.argv[1:] or list(TASKS)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_out = {}
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            all_out = json.load(f)
    for task in tasks:
        print(f"=== GSR sensitivity: {TASK_DISPLAY[task]} ===", flush=True)
        all_out[task] = run_task(task)
        with open(OUT_PATH, 'w') as f:
            json.dump(all_out, f, indent=2, default=float)
    print('Written', OUT_PATH)


if __name__ == '__main__':
    main()
