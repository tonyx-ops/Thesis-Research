"""Match EEGViT's public ID split without assuming IDs are unique people."""
import math
import numpy as np


def make_splits(labels, caps=(0, 0, 0)):
    labels = np.asarray(labels)
    if labels.ndim != 2 or labels.shape[1] != 3 or not np.isfinite(labels).all():
        raise ValueError('Expected finite labels [record ID, x, y]')
    if len(caps) != 3 or any(cap < 0 for cap in caps):
        raise ValueError('Three nonnegative caps required; 0 means all samples')
    ids = np.unique(labels[:, 0])
    n_val = n_test = math.ceil(len(ids) * .15)
    n_train = len(ids) - n_val - n_test
    if n_train < 1:
        raise ValueError('Not enough IDs for three nonempty groups')
    groups = (ids[:n_train], ids[n_train:n_train + n_val], ids[n_train + n_val:])
    result = {}
    for name, group, cap in zip(('train', 'val', 'test'), groups, caps):
        eligible = np.flatnonzero(np.isin(labels[:, 0], group))
        if cap and cap < len(eligible):
            buckets = [np.flatnonzero(labels[:, 0] == identifier) for identifier in group]
            quotas = np.full(len(buckets), cap // len(buckets))
            quotas[:cap % len(buckets)] += 1
            picks = []
            for bucket, quota in zip(buckets, quotas):
                if quota > len(bucket):
                    raise ValueError('Cap exceeds a per-ID quota; use 0 for full data')
                picks.extend(bucket[np.linspace(0, len(bucket) - 1, quota, dtype=int)])
            eligible = np.sort(picks)
        result[name] = np.asarray(eligible, dtype=np.int64)
    sets = [set(labels[result[name], 0]) for name in ('train', 'val', 'test')]
    assert not any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3))
    return result


def split_report(labels, splits):
    return dict(
        reference='https://github.com/ruiqiRichard/EEGViT/blob/master/helper_functions.py',
        protocol='sorted unique labels[:,0] IDs; 70/15/15 by ID count, rounded as EEGViT',
        total_samples=len(labels), total_ids=len(np.unique(labels[:, 0])),
        sample_counts={name: len(ix) for name, ix in splits.items()},
        id_counts={name: len(np.unique(labels[ix, 0])) for name, ix in splits.items()},
        id_ranges={name: [float(labels[ix, 0].min()), float(labels[ix, 0].max())]
                   if len(ix) else [] for name, ix in splits.items()},
        id_overlap=False,
        participant_mapping='Unverified: ID counts are not established participant counts',
        paper_table_sample_counts={'train': 14706, 'val': 3277, 'test': 3481},
        selection='minimum validation MSE; test excluded from optimization and checkpoint selection')
