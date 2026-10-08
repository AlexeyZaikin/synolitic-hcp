"""Coarse anatomical grouping of the HCP-MMP1.0 parcels.

Nine descriptive systems, assigned by exact parcel label. The grouping follows
the anatomical divisions described by Glasser et al. (2016) but is deliberately
coarser, and it is used only to summarise where the discriminative weight of a
model falls; no result depends on it. The full assignment is printed by
running this module and is reproduced in the supplementary material.
"""

GROUPS = {
    'Visual': [
        'V1', 'V2', 'V3', 'V4', 'V8', 'V3A', 'V3B', 'V3CD', 'V4t', 'V6', 'V6A',
        'V7', 'VVC', 'VMV1', 'VMV2', 'VMV3', 'LO1', 'LO2', 'LO3', 'PIT', 'FFC',
        'MT', 'MST', 'FST', 'PH', 'DVT', 'IPS1',
    ],
    'Somatomotor and premotor': [
        '4', '3a', '3b', '1', '2', '5L', '5m', '5mv', '24dd', '24dv',
        '6a', '6d', '6ma', '6mp', '6r', '6v', '55b', 'FEF', 'PEF', 'SCEF',
        '43', 'OP1', 'OP2-3', 'OP4', 'PFcm',
    ],
    'Auditory and superior temporal': [
        'A1', 'A4', 'A5', 'MBelt', 'LBelt', 'PBelt', 'RI', 'TA2', '52', 'STGa',
        'STSda', 'STSdp', 'STSva', 'STSvp', 'PSL', 'STV', 'TPOJ1', 'TPOJ2',
        'TPOJ3',
    ],
    'Lateral and medial temporal': [
        'TE1a', 'TE1m', 'TE1p', 'TE2a', 'TE2p', 'TF', 'TGd', 'TGv', 'PHT',
        'PeEc', 'PreS', 'EC', 'H', 'PHA1', 'PHA2', 'PHA3', 'ProS',
    ],
    'Insular and opercular': [
        'AAIC', 'AVI', 'MI', 'Ig', 'Pir', 'PoI1', 'PoI2', 'PI',
        'FOP1', 'FOP2', 'FOP3', 'FOP4', 'FOP5',
    ],
    'Posterior parietal': [
        '7AL', '7Am', '7PC', '7PL', '7Pm', '7m', 'AIP', 'LIPd', 'LIPv', 'MIP',
        'VIP', 'IP0', 'IP1', 'IP2', 'PF', 'PFm', 'PFop', 'PFt', 'PGi', 'PGp',
        'PGs', 'PCV', 'POS1', 'POS2',
    ],
    'Cingulate and medial parietal': [
        '23c', '23d', 'v23ab', 'd23ab', '31a', '31pd', '31pv', '33pr',
        'a24', 'a24pr', 'p24', 'p24pr', 'a32pr', 'p32pr', 'd32', 'p32', 's32',
        '25', 'RSC',
    ],
    'Prefrontal': [
        '8Ad', '8Av', '8BL', '8BM', '8C', '9-46d', '9a', '9m', '9p', '10d',
        '10pp', '10r', '10v', 'a10p', 'p10p', '11l', '13l', '44', '45', '46',
        '47l', '47m', '47s', 'a47r', 'p47r', 'a9-46v', 'p9-46v', 'IFJa',
        'IFJp', 'IFSa', 'IFSp', 'OFC', 'pOFC', 'i6-8', 's6-8', 'SFL',
    ],
    'Subcortical and cerebellum': [
        'accumbens', 'amygdala', 'caudate', 'cerebellum', 'diencephalon',
        'hippocampus', 'pallidum', 'putamen', 'thalamus', 'brainStem',
    ],
}

_LOOKUP = {}
for _g, _labels in GROUPS.items():
    for _lab in _labels:
        assert _lab not in _LOOKUP, f'duplicate label {_lab}'
        _LOOKUP[_lab] = _g


def group_of(region_name):
    """Map a region name from data_loader.MMP_REGIONS to its system."""
    if region_name.startswith(('L_', 'R_')):
        return _LOOKUP[region_name[2:]]
    base = region_name.replace('_left', '').replace('_right', '')
    return _LOOKUP[base]


def hemisphere_of(region_name):
    if region_name.startswith('L_') or region_name.endswith('_left'):
        return 'L'
    if region_name.startswith('R_') or region_name.endswith('_right'):
        return 'R'
    return 'M'


if __name__ == '__main__':
    import os
    import sys
    from collections import Counter
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    from code.utils.data_loader import MMP_REGIONS
    counts = Counter(group_of(r) for r in MMP_REGIONS)
    total = 0
    for g in GROUPS:
        print(f'{g:<32} {counts[g]:>4}')
        total += counts[g]
    print(f'{"TOTAL":<32} {total:>4} of {len(MMP_REGIONS)}')
    assert total == len(MMP_REGIONS)
