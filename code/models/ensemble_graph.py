"""
Ensemble (synolytic) graph construction for HCP fMRI data.

Implements the method from Vlasenko et al. (2025): edge weights encode
the difference between posterior probabilities of two brain states,
estimated by logistic regression on pairwise time-series features.

Optimized: pre-computes all features vectorized, then trains classifiers.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Tuple


def precompute_all_features(
    timeseries: list,
    region_indices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pre-compute mean, std, and correlation matrix for all samples.

    Args:
        timeseries: list of (n_total_regions, T) arrays.
        region_indices: indices of regions to use.

    Returns:
        means: (n_samples, n_regions) per-region means.
        stds: (n_samples, n_regions) per-region stds.
        corrs: (n_samples, n_regions, n_regions) pairwise correlations.
    """
    n_samples = len(timeseries)
    n_regions = len(region_indices)

    means = np.zeros((n_samples, n_regions), dtype=np.float32)
    stds = np.zeros((n_samples, n_regions), dtype=np.float32)
    corrs = np.zeros((n_samples, n_regions, n_regions), dtype=np.float32)

    for s in range(n_samples):
        ts = timeseries[s][region_indices]  # (n_regions, T)
        means[s] = ts.mean(axis=1)
        stds[s] = ts.std(axis=1)
        c = np.corrcoef(ts)
        corrs[s] = np.nan_to_num(c, nan=0.0).astype(np.float32)

    return means, stds, corrs


def build_ensemble_graphs_fast(
    means_train: np.ndarray,
    stds_train: np.ndarray,
    corrs_train: np.ndarray,
    labels_train: np.ndarray,
    means_pred: np.ndarray,
    stds_pred: np.ndarray,
    corrs_pred: np.ndarray,
    C: float = 1.0,
    use_correlation: bool = True
) -> np.ndarray:
    """Build ensemble graphs using pre-computed features. Vectorized.

    For each edge (i,j), trains a logistic regression on pairwise features
    and predicts w_ij = P(state2) - P(state1).

    Args:
        means_train, stds_train, corrs_train: training features.
        labels_train: (n_train,) binary labels.
        means_pred, stds_pred, corrs_pred: prediction features.
        C: regularization parameter.
        use_correlation: if True, use 5 features [mean_i, std_i, mean_j, std_j, corr_ij].
            If False, use only 4 marginal features [mean_i, std_i, mean_j, std_j].

    Returns:
        W: (n_pred, n_regions, n_regions) edge weight matrices.
    """
    n_train = means_train.shape[0]
    n_pred = means_pred.shape[0]
    n_regions = means_train.shape[1]

    W = np.zeros((n_pred, n_regions, n_regions), dtype=np.float32)

    for i in range(n_regions):
        for j in range(i + 1, n_regions):
            if use_correlation:
                X_train = np.column_stack([
                    means_train[:, i], stds_train[:, i],
                    means_train[:, j], stds_train[:, j],
                    corrs_train[:, i, j]
                ])
                X_pred = np.column_stack([
                    means_pred[:, i], stds_pred[:, i],
                    means_pred[:, j], stds_pred[:, j],
                    corrs_pred[:, i, j]
                ])
            else:
                X_train = np.column_stack([
                    means_train[:, i], stds_train[:, i],
                    means_train[:, j], stds_train[:, j],
                ])
                X_pred = np.column_stack([
                    means_pred[:, i], stds_pred[:, i],
                    means_pred[:, j], stds_pred[:, j],
                ])

            # Standardize
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_pred_s = scaler.transform(X_pred)

            # Train and predict
            clf = LogisticRegression(C=C, solver='lbfgs',
                                     max_iter=100, random_state=42)
            clf.fit(X_train_s, labels_train)
            probs = clf.predict_proba(X_pred_s)
            edge_w = probs[:, 1] - probs[:, 0]

            W[:, i, j] = edge_w
            W[:, j, i] = edge_w

    return W


def build_correlation_graphs(
    corrs: np.ndarray
) -> np.ndarray:
    """Build z-scored correlation graphs from pre-computed correlations.

    Args:
        corrs: (n_samples, n_regions, n_regions) correlation matrices.

    Returns:
        C_zscore: (n_samples, n_regions, n_regions) z-scored correlation matrices.
    """
    n_samples, n_regions, _ = corrs.shape
    C_out = np.zeros_like(corrs)

    for s in range(n_samples):
        c = corrs[s].copy()
        mask = np.triu_indices(n_regions, k=1)
        vals = c[mask]
        if vals.std() > 0:
            c = (c - vals.mean()) / vals.std()
        np.fill_diagonal(c, 0)
        C_out[s] = c

    return C_out


def compute_node_summaries(W: np.ndarray) -> np.ndarray:
    """Compute mean incident edge weight for each node.

    d_i = (1/(N-1)) * sum_{j!=i} w_ij

    Args:
        W: (n_samples, n_regions, n_regions) edge weight matrices.

    Returns:
        node_features: (n_samples, n_regions) node summary features.
    """
    n_regions = W.shape[1]
    node_sums = W.sum(axis=2)  # (n_samples, n_regions)
    return node_sums / (n_regions - 1)
