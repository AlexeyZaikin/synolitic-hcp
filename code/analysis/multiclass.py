"""Multiclass extension: does the result hold beyond binary classification?

Each HCP paradigm as distributed contains exactly two conditions, so a
multiclass problem has to be built across paradigms. We use the 14 classes
formed by the seven paradigms times their two conditions, over the subjects
that contribute all fourteen. Chance is 1/14.

The synolitic edge generalises naturally. For K classes the elementary model
on the pair (i, j) returns a posterior vector, and the edge carries the
centred posterior

    w_ij^k = P(k | features of the pair) - 1/K,      k = 1 ... K

which reduces to the signed binary weight, up to a factor of two, when K = 2.
The node summary of each region becomes a K-vector, the mean incident edge
weight per class, so a graph over N regions yields N*K features instead of N.

Region selection keeps the spirit of the binary experiment - discard the
regions that are individually informative - with the threshold placed the same
distance above chance: 1/K + 0.10 by default.

Outputs
    results/metrics/multiclass_results.json

Usage
    python code/analysis/multiclass.py [--margin 0.10]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from code.utils.data_loader import load_task_timeseries, TASKS, TASK_DISPLAY  # noqa: E402

RANDOM_SEED = 42
N_FOLDS = 4
N_JOBS = 12
ENCODING = 'LR'
# Parcellated HCP time series (not distributed with this repository; see README).
# Override with the environment variable HCP_DATA_DIR.
DATA_DIR = os.environ.get(
    'HCP_DATA_DIR', os.path.join(PROJECT_ROOT, 'data', 'HCP_v4', 'split_data', 'split'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'metrics')
OUT_PATH = os.path.join(RESULTS_DIR, 'multiclass_results.json')


def build_dataset():
    """One sample per subject per (paradigm, condition): 14 classes."""
    per_subject = {}
    class_names = []
    for task in TASKS:
        ts, lab, sid = load_task_timeseries(DATA_DIR, task, encoding=ENCODING)
        names = TASKS[task]
        for k in (0, 1):
            class_names.append(f'{task}:{names[k]}')
        base = len(class_names) - 2
        for series, y, s in zip(ts, lab, sid):
            per_subject.setdefault(s, {})[base + int(y)] = series
        print(f'  loaded {TASK_DISPLAY[task]}: {len(ts)} samples', flush=True)

    K = len(class_names)
    complete = sorted(s for s, d in per_subject.items() if len(d) == K)
    timeseries, labels, subjects = [], [], []
    for s in complete:
        for k in range(K):
            timeseries.append(per_subject[s][k])
            labels.append(k)
            subjects.append(s)
    print(f'  {len(complete)} subjects with all {K} classes, '
          f'{len(timeseries)} samples', flush=True)
    return timeseries, np.array(labels), np.array(subjects), class_names


def region_features(timeseries, n_regions):
    means = np.zeros((len(timeseries), n_regions), dtype=np.float32)
    stds = np.zeros((len(timeseries), n_regions), dtype=np.float32)
    for s, ts in enumerate(timeseries):
        means[s] = ts.mean(axis=1)
        stds[s] = ts.std(axis=1)
    return means, stds


def multinomial(seed=RANDOM_SEED, max_iter=200):
    return LogisticRegression(C=1.0, solver='lbfgs', max_iter=max_iter,
                              random_state=seed)


def assess_regions(means, stds, labels, subjects, train_idx):
    """Individual accuracy of every region, training subjects only."""
    sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    y = labels[train_idx]
    g = subjects[train_idx]
    folds = list(sgkf.split(np.zeros(len(train_idx)), y, groups=g))

    def one(r):
        X = np.column_stack([means[train_idx, r], stds[train_idx, r]])
        accs = []
        for tr, te in folds:
            sc = StandardScaler()
            clf = multinomial()
            clf.fit(sc.fit_transform(X[tr]), y[tr])
            accs.append(accuracy_score(y[te], clf.predict(sc.transform(X[te]))))
        return float(np.mean(accs))

    return np.array(Parallel(n_jobs=N_JOBS)(
        delayed(one)(r) for r in range(means.shape[1])), dtype=np.float32)


def _edge_block(i_list, means_tr, stds_tr, corrs_tr, y_tr,
                means_pr, stds_pr, corrs_pr, K, use_corr):
    """Node summaries contributed to regions in i_list by all their edges.

    Each region in i_list is paired with every other region, so edges are
    fitted twice overall; this keeps the returned array small enough to send
    back from a worker, which matters far more than the duplicated fits.
    """
    n_pred = means_pr.shape[0]
    n_regions = means_tr.shape[1]
    out = np.zeros((n_pred, len(i_list), K), dtype=np.float32)
    for pos, i in enumerate(i_list):
        acc = np.zeros((n_pred, K), dtype=np.float64)
        for j in range(n_regions):
            if j == i:
                continue
            cols_tr = [means_tr[:, i], stds_tr[:, i], means_tr[:, j], stds_tr[:, j]]
            cols_pr = [means_pr[:, i], stds_pr[:, i], means_pr[:, j], stds_pr[:, j]]
            if use_corr:
                cols_tr.append(corrs_tr[:, i, j])
                cols_pr.append(corrs_pr[:, i, j])
            X_tr = np.column_stack(cols_tr)
            X_pr = np.column_stack(cols_pr)
            sc = StandardScaler()
            clf = multinomial()
            clf.fit(sc.fit_transform(X_tr), y_tr)
            p = clf.predict_proba(sc.transform(X_pr))
            full = np.full((n_pred, K), 1.0 / K)
            full[:, clf.classes_] = p
            acc += full - 1.0 / K
        out[:, pos, :] = (acc / (n_regions - 1)).astype(np.float32)
    return out


def synolitic_node_features(means_tr, stds_tr, corrs_tr, y_tr,
                            means_pr, stds_pr, corrs_pr, K, use_corr=True):
    n_regions = means_tr.shape[1]
    blocks = [list(range(a, min(a + 8, n_regions))) for a in range(0, n_regions, 8)]
    parts = Parallel(n_jobs=N_JOBS, verbose=0)(
        delayed(_edge_block)(b, means_tr, stds_tr, corrs_tr, y_tr,
                             means_pr, stds_pr, corrs_pr, K, use_corr)
        for b in blocks)
    return np.concatenate(parts, axis=1).reshape(means_pr.shape[0], -1)


def corr_tensor(timeseries, idx, region_idx):
    n = len(idx)
    R = len(region_idx)
    out = np.zeros((n, R, R), dtype=np.float32)
    for pos, s in enumerate(idx):
        c = np.corrcoef(timeseries[s][region_idx])
        out[pos] = np.nan_to_num(c, nan=0.0)
    return out


def correlation_node_features(corrs):
    n, R, _ = corrs.shape
    out = np.zeros((n, R), dtype=np.float32)
    iu = np.triu_indices(R, k=1)
    for s in range(n):
        c = corrs[s].copy()
        v = c[iu]
        if v.std() > 0:
            c = (c - v.mean()) / v.std()
        np.fill_diagonal(c, 0.0)
        out[s] = c.sum(axis=1) / (R - 1)
    return out


def evaluate(X, y, train_idx, test_idx):
    sc = StandardScaler()
    clf = multinomial(max_iter=1000)
    clf.fit(sc.fit_transform(X[train_idx]), y[train_idx])
    Xte = sc.transform(X[test_idx])
    pred = clf.predict(Xte)
    prob = clf.predict_proba(Xte)
    acc = accuracy_score(y[test_idx], pred)
    try:
        auc = roc_auc_score(y[test_idx], prob, multi_class='ovr',
                            average='macro', labels=clf.classes_)
    except ValueError:
        auc = float('nan')
    return float(acc), float(auc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--margin', type=float, default=0.10,
                    help='threshold above chance for calling a region weak')
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print('Building the 14-class data set...', flush=True)
    timeseries, labels, subjects, class_names = build_dataset()
    K = len(class_names)
    n_regions = timeseries[0].shape[0]
    chance = 1.0 / K
    threshold = chance + args.margin
    print(f'  K={K}, chance={chance:.4f}, weak threshold={threshold:.4f}',
          flush=True)

    means, stds = region_features(timeseries, n_regions)

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True,
                                random_state=RANDOM_SEED)
    folds = list(sgkf.split(np.zeros(len(timeseries)), labels, groups=subjects))

    res = {k: {'accs': [], 'aucs': []} for k in
           ['all_direct', 'weak_direct', 'weak_corr', 'weak_ensemble',
            'weak_marginal']}
    n_weak_per_fold, weak_idx_per_fold, region_accs_per_fold = [], [], []

    X_all_direct = np.column_stack([means, stds])

    for fold, (train_idx, test_idx) in enumerate(folds):
        t0 = time.time()
        accs = assess_regions(means, stds, labels, subjects, train_idx)
        region_accs_per_fold.append(accs.tolist())
        weak = np.where(accs < threshold)[0]
        n_weak_per_fold.append(int(len(weak)))
        weak_idx_per_fold.append(weak.tolist())
        print(f'fold {fold+1}: {len(weak)} weak regions of {n_regions} '
              f'({time.time()-t0:.0f}s); individual accuracy '
              f'{accs.min():.3f} to {accs.max():.3f}', flush=True)

        a, u = evaluate(X_all_direct, labels, train_idx, test_idx)
        res['all_direct']['accs'].append(a)
        res['all_direct']['aucs'].append(u)
        print(f'  all direct        acc={a:.4f} auc={u:.4f}', flush=True)

        if len(weak) < 2:
            for k in ['weak_direct', 'weak_corr', 'weak_ensemble', 'weak_marginal']:
                res[k]['accs'].append(float('nan'))
                res[k]['aucs'].append(float('nan'))
            continue

        Xw = np.column_stack([means[:, weak], stds[:, weak]])
        a, u = evaluate(Xw, labels, train_idx, test_idx)
        res['weak_direct']['accs'].append(a)
        res['weak_direct']['aucs'].append(u)
        print(f'  weak direct       acc={a:.4f} auc={u:.4f}', flush=True)

        t0 = time.time()
        corrs_tr = corr_tensor(timeseries, train_idx, weak)
        corrs_te = corr_tensor(timeseries, test_idx, weak)
        print(f'  correlations in {time.time()-t0:.0f}s', flush=True)

        Xc = np.zeros((len(labels), len(weak)), dtype=np.float32)
        Xc[train_idx] = correlation_node_features(corrs_tr)
        Xc[test_idx] = correlation_node_features(corrs_te)
        a, u = evaluate(Xc, labels, train_idx, test_idx)
        res['weak_corr']['accs'].append(a)
        res['weak_corr']['aucs'].append(u)
        print(f'  weak correlation  acc={a:.4f} auc={u:.4f}', flush=True)

        # inner loop for out-of-fold training graphs, grouped by subject
        inner = StratifiedGroupKFold(n_splits=N_FOLDS - 1, shuffle=True,
                                     random_state=RANDOM_SEED)
        inner_folds = list(inner.split(np.zeros(len(train_idx)),
                                       labels[train_idx],
                                       groups=subjects[train_idx]))

        for key, use_corr in (('weak_ensemble', True), ('weak_marginal', False)):
            t0 = time.time()
            Xg = None
            for base_rel, graph_rel in inner_folds:
                b = train_idx[base_rel]
                g = train_idx[graph_rel]
                f = synolitic_node_features(
                    means[b][:, weak], stds[b][:, weak], corrs_tr[base_rel],
                    labels[b], means[g][:, weak], stds[g][:, weak],
                    corrs_tr[graph_rel], K, use_corr)
                if Xg is None:
                    Xg = np.zeros((len(labels), f.shape[1]), dtype=np.float32)
                Xg[g] = f
            Xg[test_idx] = synolitic_node_features(
                means[train_idx][:, weak], stds[train_idx][:, weak], corrs_tr,
                labels[train_idx], means[test_idx][:, weak],
                stds[test_idx][:, weak], corrs_te, K, use_corr)
            a, u = evaluate(Xg, labels, train_idx, test_idx)
            res[key]['accs'].append(a)
            res[key]['aucs'].append(u)
            print(f'  {key:<16}  acc={a:.4f} auc={u:.4f} '
                  f'({time.time()-t0:.0f}s)', flush=True)
            del Xg

        del corrs_tr, corrs_te

    out = {
        'n_classes': K,
        'class_names': class_names,
        'chance': chance,
        'margin': args.margin,
        'threshold': threshold,
        'n_samples': len(timeseries),
        'n_subjects': len(set(subjects.tolist())),
        'n_regions': n_regions,
        'n_weak_per_fold': n_weak_per_fold,
        'n_weak_median': int(np.median(n_weak_per_fold)),
        'n_weak_range': [int(min(n_weak_per_fold)), int(max(n_weak_per_fold))],
        'weak_indices_per_fold': weak_idx_per_fold,
        'region_accs_per_fold': region_accs_per_fold,
    }
    for k, v in res.items():
        a = [x for x in v['accs'] if not np.isnan(x)]
        u = [x for x in v['aucs'] if not np.isnan(x)]
        out[k] = {'accuracy': a, 'auc': u,
                  'accuracy_mean': float(np.mean(a)) if a else float('nan'),
                  'accuracy_std': float(np.std(a)) if a else float('nan'),
                  'auc_mean': float(np.mean(u)) if u else float('nan'),
                  'auc_std': float(np.std(u)) if u else float('nan')}
    with open(OUT_PATH, 'w') as f:
        json.dump(out, f, indent=2)
    print('Written', OUT_PATH)


if __name__ == '__main__':
    main()
