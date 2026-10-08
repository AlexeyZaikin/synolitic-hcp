"""
Data loading utilities for HCP fMRI parcellated time series.

Each pickle file contains a list of DataFrames (one per block),
where rows = brain regions and columns = time points within that block.
"""

import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# MMP atlas region names (379 regions: 180 L cortical + 180 R cortical + 19 subcortical)
MMP_REGIONS = [
    'L_V1', 'L_MST', 'L_V6', 'L_V2', 'L_V3', 'L_V4', 'L_V8', 'L_4', 'L_3b', 'L_FEF',
    'L_PEF', 'L_55b', 'L_V3A', 'L_RSC', 'L_POS2', 'L_V7', 'L_IPS1', 'L_FFC', 'L_V3B',
    'L_LO1', 'L_LO2', 'L_PIT', 'L_MT', 'L_A1', 'L_PSL', 'L_SFL', 'L_PCV', 'L_STV',
    'L_7Pm', 'L_7m', 'L_POS1', 'L_23d', 'L_v23ab', 'L_d23ab', 'L_31pv', 'L_5m', 'L_5mv',
    'L_23c', 'L_5L', 'L_24dd', 'L_24dv', 'L_7AL', 'L_SCEF', 'L_6ma', 'L_7Am', 'L_7PL',
    'L_7PC', 'L_LIPv', 'L_VIP', 'L_MIP', 'L_1', 'L_2', 'L_3a', 'L_6d', 'L_6mp', 'L_6v',
    'L_p24pr', 'L_33pr', 'L_a24pr', 'L_p32pr', 'L_a24', 'L_d32', 'L_8BM', 'L_p32',
    'L_10r', 'L_47m', 'L_8Av', 'L_8Ad', 'L_9m', 'L_8BL', 'L_9p', 'L_10d', 'L_8C',
    'L_44', 'L_45', 'L_47l', 'L_a47r', 'L_6r', 'L_IFJa', 'L_IFJp', 'L_IFSp', 'L_IFSa',
    'L_p9-46v', 'L_46', 'L_a9-46v', 'L_9-46d', 'L_9a', 'L_10v', 'L_a10p', 'L_10pp',
    'L_11l', 'L_13l', 'L_OFC', 'L_47s', 'L_LIPd', 'L_6a', 'L_i6-8', 'L_s6-8', 'L_43',
    'L_OP4', 'L_OP1', 'L_OP2-3', 'L_52', 'L_RI', 'L_PFcm', 'L_PoI2', 'L_TA2', 'L_FOP4',
    'L_MI', 'L_Pir', 'L_AVI', 'L_AAIC', 'L_FOP1', 'L_FOP3', 'L_FOP2', 'L_PFt', 'L_AIP',
    'L_EC', 'L_PreS', 'L_H', 'L_ProS', 'L_PeEc', 'L_STGa', 'L_PBelt', 'L_A5', 'L_PHA1',
    'L_PHA3', 'L_STSda', 'L_STSdp', 'L_STSvp', 'L_TGd', 'L_TE1a', 'L_TE1p', 'L_TE2a',
    'L_TF', 'L_TE2p', 'L_PHT', 'L_PH', 'L_TPOJ1', 'L_TPOJ2', 'L_TPOJ3', 'L_DVT',
    'L_PGp', 'L_IP2', 'L_IP1', 'L_IP0', 'L_PFop', 'L_PF', 'L_PFm', 'L_PGi', 'L_PGs',
    'L_V6A', 'L_VMV1', 'L_VMV3', 'L_PHA2', 'L_V4t', 'L_FST', 'L_V3CD', 'L_LO3',
    'L_VMV2', 'L_31pd', 'L_31a', 'L_VVC', 'L_25', 'L_s32', 'L_pOFC', 'L_PoI1', 'L_Ig',
    'L_FOP5', 'L_p10p', 'L_p47r', 'L_TGv', 'L_MBelt', 'L_LBelt', 'L_A4', 'L_STSva',
    'L_TE1m', 'L_PI', 'L_a32pr', 'L_p24',
    'R_V1', 'R_MST', 'R_V6', 'R_V2', 'R_V3', 'R_V4', 'R_V8', 'R_4', 'R_3b', 'R_FEF',
    'R_PEF', 'R_55b', 'R_V3A', 'R_RSC', 'R_POS2', 'R_V7', 'R_IPS1', 'R_FFC', 'R_V3B',
    'R_LO1', 'R_LO2', 'R_PIT', 'R_MT', 'R_A1', 'R_PSL', 'R_SFL', 'R_PCV', 'R_STV',
    'R_7Pm', 'R_7m', 'R_POS1', 'R_23d', 'R_v23ab', 'R_d23ab', 'R_31pv', 'R_5m', 'R_5mv',
    'R_23c', 'R_5L', 'R_24dd', 'R_24dv', 'R_7AL', 'R_SCEF', 'R_6ma', 'R_7Am', 'R_7PL',
    'R_7PC', 'R_LIPv', 'R_VIP', 'R_MIP', 'R_1', 'R_2', 'R_3a', 'R_6d', 'R_6mp', 'R_6v',
    'R_p24pr', 'R_33pr', 'R_a24pr', 'R_p32pr', 'R_a24', 'R_d32', 'R_8BM', 'R_p32',
    'R_10r', 'R_47m', 'R_8Av', 'R_8Ad', 'R_9m', 'R_8BL', 'R_9p', 'R_10d', 'R_8C',
    'R_44', 'R_45', 'R_47l', 'R_a47r', 'R_6r', 'R_IFJa', 'R_IFJp', 'R_IFSp', 'R_IFSa',
    'R_p9-46v', 'R_46', 'R_a9-46v', 'R_9-46d', 'R_9a', 'R_10v', 'R_a10p', 'R_10pp',
    'R_11l', 'R_13l', 'R_OFC', 'R_47s', 'R_LIPd', 'R_6a', 'R_i6-8', 'R_s6-8', 'R_43',
    'R_OP4', 'R_OP1', 'R_OP2-3', 'R_52', 'R_RI', 'R_PFcm', 'R_PoI2', 'R_TA2', 'R_FOP4',
    'R_MI', 'R_Pir', 'R_AVI', 'R_AAIC', 'R_FOP1', 'R_FOP3', 'R_FOP2', 'R_PFt', 'R_AIP',
    'R_EC', 'R_PreS', 'R_H', 'R_ProS', 'R_PeEc', 'R_STGa', 'R_PBelt', 'R_A5', 'R_PHA1',
    'R_PHA3', 'R_STSda', 'R_STSdp', 'R_STSvp', 'R_TGd', 'R_TE1a', 'R_TE1p', 'R_TE2a',
    'R_TF', 'R_TE2p', 'R_PHT', 'R_PH', 'R_TPOJ1', 'R_TPOJ2', 'R_TPOJ3', 'R_DVT',
    'R_PGp', 'R_IP2', 'R_IP1', 'R_IP0', 'R_PFop', 'R_PF', 'R_PFm', 'R_PGi', 'R_PGs',
    'R_V6A', 'R_VMV1', 'R_VMV3', 'R_PHA2', 'R_V4t', 'R_FST', 'R_V3CD', 'R_LO3',
    'R_VMV2', 'R_31pd', 'R_31a', 'R_VVC', 'R_25', 'R_s32', 'R_pOFC', 'R_PoI1', 'R_Ig',
    'R_FOP5', 'R_p10p', 'R_p47r', 'R_TGv', 'R_MBelt', 'R_LBelt', 'R_A4', 'R_STSva',
    'R_TE1m', 'R_PI', 'R_a32pr', 'R_p24',
    'accumbens_left', 'accumbens_right', 'amygdala_left', 'amygdala_right',
    'caudate_left', 'caudate_right', 'cerebellum_left', 'cerebellum_right',
    'diencephalon_left', 'diencephalon_right', 'hippocampus_left', 'hippocampus_right',
    'pallidum_left', 'pallidum_right', 'putamen_left', 'putamen_right',
    'thalamus_left', 'thalamus_right', 'brainStem'
]

# Yeo17 network names (34 regions: 17 networks x 2 hemispheres)
YEO17_REGIONS = [f'network_{i} {h}' for h in ['L', 'R'] for i in range(1, 18)]

# Task definitions: task_name -> (label_0, label_1)
TASKS = {
    'WM': ('0bk', '2bk'),
    'GAMBLING': ('loss', 'win'),
    'MOTOR': ('l', 'r'),
    'LANGUAGE': ('math', 'story'),
    'SOCIAL': ('mental', 'rnd'),
    'RELATIONAL': ('match', 'relation'),
    'EMOTION': ('fear', 'neut'),
}

# Task display names for paper
TASK_DISPLAY = {
    'WM': 'Working Memory',
    'GAMBLING': 'Gambling',
    'MOTOR': 'Motor Activity',
    'LANGUAGE': 'Language Processing',
    'SOCIAL': 'Social Cognition',
    'RELATIONAL': 'Relational Processing',
    'EMOTION': 'Emotion Processing',
}


def load_task_data(
    data_dir: str,
    task: str,
    encoding: str = 'LR',
    atlas: str = 'mmp'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load HCP task data and extract features.

    For each sample, concatenates all blocks and computes per-region
    mean and standard deviation of the time series.

    Args:
        data_dir: Path to the extracted split data directory.
        task: Task name (e.g., 'LANGUAGE', 'WM').
        encoding: Phase encoding direction ('LR' or 'RL').
        atlas: Atlas used ('mmp' for 379 regions, 'yeo17' for 34 regions).

    Returns:
        means: (n_samples, n_regions) mean of time series per region.
        stds: (n_samples, n_regions) std of time series per region.
        labels: (n_samples,) binary labels (0 or 1).
        subject_ids: (n_samples,) subject identifiers.
        region_names: list of region names.
    """
    label_0, label_1 = TASKS[task]
    task_dir = os.path.join(data_dir, task)

    if atlas == 'mmp':
        n_regions = 379
        region_names = MMP_REGIONS
    else:
        n_regions = 34
        region_names = YEO17_REGIONS

    means_list = []
    stds_list = []
    labels_list = []
    subjects_list = []

    for fname in sorted(os.listdir(task_dir)):
        if not fname.endswith('.pickle'):
            continue

        parts = fname.replace('.pickle', '').split('_')
        subject_id = parts[0]
        file_encoding = parts[1]
        label_str = parts[2]

        if file_encoding != encoding:
            continue

        if label_str == label_0:
            label = 0
        elif label_str == label_1:
            label = 1
        else:
            continue

        filepath = os.path.join(task_dir, fname)
        with open(filepath, 'rb') as f:
            blocks = pickle.load(f)

        # Concatenate all blocks along time axis
        # Each block is a DataFrame: (n_regions, n_timepoints)
        all_timepoints = pd.concat(blocks, axis=1)
        ts = all_timepoints.values.astype(np.float32)  # (n_regions, total_timepoints)

        means_list.append(ts.mean(axis=1))
        stds_list.append(ts.std(axis=1))
        labels_list.append(label)
        subjects_list.append(subject_id)

    means = np.array(means_list, dtype=np.float32)
    stds = np.array(stds_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int32)
    subject_ids = np.array(subjects_list)

    return means, stds, labels, subject_ids, region_names


def compute_pairwise_features(
    means: np.ndarray,
    stds: np.ndarray,
    blocks_data: List[np.ndarray],
    region_i: int,
    region_j: int
) -> np.ndarray:
    """Compute pairwise features for edge (i, j) across all samples.

    Features: [mean_i, std_i, mean_j, std_j, pearson_corr(i,j)]

    Args:
        means: (n_samples, n_regions) region means.
        stds: (n_samples, n_regions) region stds.
        blocks_data: list of (n_regions, total_timepoints) arrays per sample.
        region_i: Index of first region.
        region_j: Index of second region.

    Returns:
        features: (n_samples, 5) pairwise feature matrix.
    """
    n_samples = means.shape[0]
    features = np.zeros((n_samples, 5), dtype=np.float32)
    features[:, 0] = means[:, region_i]
    features[:, 1] = stds[:, region_i]
    features[:, 2] = means[:, region_j]
    features[:, 3] = stds[:, region_j]

    for s in range(n_samples):
        ts_i = blocks_data[s][region_i]
        ts_j = blocks_data[s][region_j]
        corr = np.corrcoef(ts_i, ts_j)[0, 1]
        features[s, 4] = corr if np.isfinite(corr) else 0.0

    return features


def load_task_timeseries(
    data_dir: str,
    task: str,
    encoding: str = 'LR'
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """Load raw concatenated time series for all samples.

    Returns:
        timeseries: list of (n_regions, total_timepoints) arrays.
        labels: (n_samples,) binary labels.
        subject_ids: (n_samples,) subject identifiers.
    """
    label_0, label_1 = TASKS[task]
    task_dir = os.path.join(data_dir, task)

    timeseries = []
    labels_list = []
    subjects_list = []

    for fname in sorted(os.listdir(task_dir)):
        if not fname.endswith('.pickle'):
            continue

        parts = fname.replace('.pickle', '').split('_')
        subject_id = parts[0]
        file_encoding = parts[1]
        label_str = parts[2]

        if file_encoding != encoding:
            continue

        if label_str == label_0:
            label = 0
        elif label_str == label_1:
            label = 1
        else:
            continue

        filepath = os.path.join(task_dir, fname)
        with open(filepath, 'rb') as f:
            blocks = pickle.load(f)

        all_timepoints = pd.concat(blocks, axis=1)
        ts = all_timepoints.values.astype(np.float32)
        timeseries.append(ts)
        labels_list.append(label)
        subjects_list.append(subject_id)

    labels = np.array(labels_list, dtype=np.int32)
    subject_ids = np.array(subjects_list)

    return timeseries, labels, subject_ids
