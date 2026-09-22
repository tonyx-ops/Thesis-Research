#!/usr/bin/env python3
"""Train published GACNet classes and a corrected model on identical data.

The published entry point is broken; this is an explicitly adapted baseline.
Stored .npz arrays are memory mapped, without loading 10 GiB into RAM.
Default uses all samples and the EEGViT public ID split.
This does not resolve the ID-to-participant mapping or reproduce paper scores.
"""
import argparse
import ast
import csv
import hashlib
import json
import math
import random
import struct
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from torch_geometric.nn import GATv2Conv, BatchNorm, global_add_pool
from einops import rearrange
from corrected_model import CorrectedEncoder, CorrectedRegressionHead, supervised_contrastive_loss
from data_protocol import make_splits, split_report

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'Position_task_with_dots_synchronised_min.npz'


class KMeans:
    """Small 2D NumPy Lloyd implementation avoiding conflicting OpenMP libraries.

    Train-only k-means++ initialization, 10 starts, squared Euclidean distances.
    This reconstructs missing cluster labels; it does not recover author labels.
    """
    def __init__(self, n_clusters, random_state, n_init=10):
        self.k, self.seed, self.n_init = n_clusters, random_state, n_init

    def fit(self, x):
        x = np.asarray(x, dtype=np.float64)
        rng = np.random.default_rng(self.seed)
        best = float('inf')
        for _ in range(self.n_init):
            centers = [x[rng.integers(len(x))]]
            for _ in range(1, self.k):
                d2 = ((x[:, None] - np.asarray(centers)[None]) ** 2).sum(2).min(1)
                centers.append(x[rng.choice(len(x), p=d2 / d2.sum())] if d2.sum() else x[rng.integers(len(x))])
            centers = np.asarray(centers)
            for _ in range(100):
                d2 = ((x[:, None] - centers[None]) ** 2).sum(2)
                labels = d2.argmin(1)
                new = np.array([x[labels == j].mean(0) if np.any(labels == j)
                                else x[d2.min(1).argmax()] for j in range(self.k)])
                if np.max(np.abs(new - centers)) < 1e-6:
                    centers = new
                    break
                centers = new
            inertia = ((x[:, None] - centers[None]) ** 2).sum(2).min(1).sum()
            if inertia < best:
                self.cluster_centers_, best = centers.copy(), inertia
        return self

    def predict(self, x):
        return ((np.asarray(x)[:, None] - self.cluster_centers_[None]) ** 2).sum(2).argmin(1)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(request):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if request == 'auto' else torch.device(request)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Use auto, cpu, cuda, or cuda:N')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable. Check NVIDIA driver and CUDA-enabled PyTorch.')
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise ValueError('Requested CUDA device does not exist')
        device = torch.device('cuda', index)
    return device


def device_report(device):
    info = dict(device=str(device), torch_version=str(torch.__version__),
                cuda_runtime=torch.version.cuda, cuda_available=torch.cuda.is_available())
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(device)
        info.update(gpu_name=props.name, gpu_memory_gib=props.total_memory / 2**30)
    return info


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def load_original():
    path = ROOT / 'original_snapshot/main.py'
    source = path.read_text()
    definitions = ast.Module(body=[n for n in ast.parse(source).body
                                  if isinstance(n, (ast.FunctionDef, ast.ClassDef))], type_ignores=[])
    ns = dict(torch=torch, nn=nn, F=F, np=np, rearrange=rearrange,
              GATv2Conv=GATv2Conv, BatchNorm=BatchNorm, global_add_pool=global_add_pool)
    exec(compile(definitions, str(path), 'exec'), ns)
    return ns


def mmap_eeg(path):
    with zipfile.ZipFile(path) as archive:
        member = archive.getinfo('EEG.npy')
        if member.compress_type != zipfile.ZIP_STORED:
            raise ValueError('This memory-map runner requires an uncompressed EEG.npy member')
        with path.open('rb') as f:
            f.seek(member.header_offset)
            header = f.read(30)
            if header[:4] != b'PK\x03\x04':
                raise ValueError('Invalid local ZIP header')
            name_len, extra_len = struct.unpack_from('<HH', header, 26)
            f.seek(name_len + extra_len, 1)
            version = np.lib.format.read_magic(f)
            reader = { (1, 0): np.lib.format.read_array_header_1_0,
                       (2, 0): np.lib.format.read_array_header_2_0 }[version]
            shape, fortran, dtype = reader(f)
            offset = f.tell()
        with archive.open('labels.npy') as f:
            labels = np.load(f, allow_pickle=False)
    array = np.memmap(path, dtype=dtype, mode='r', offset=offset, shape=shape,
                      order='F' if fortran else 'C')
    if shape != (len(labels), 500, 129) or labels.shape[1] != 3:
        raise ValueError(f'Unexpected shapes {shape}, {labels.shape}')
    return array, labels



def read_subset(eeg, indices, channels):
    # Bounded sequential reads preserve EEG-label alignment.
    out = np.empty((len(indices), len(channels), 500), dtype=np.float32)
    for i, idx in enumerate(indices):
        out[i] = np.asarray(eeg[idx])[:, channels].T
    if not np.isfinite(out).all():
        raise ValueError('Non-finite EEG in selected data')
    return torch.from_numpy(out)


def edge_batch(channels, count, corrected, device='cpu'):
    if corrected:
        pairs = [(i, j) for i in range(channels) for j in range(channels) if i != j]
    else:
        pairs = [(i, j) for i in range(channels) for j in range(i + 1, channels)]
    edges = torch.tensor(pairs, dtype=torch.long, device=device).T
    edges = torch.cat([edges + i * channels for i in range(count)], dim=1)
    batch = torch.arange(count, device=device).repeat_interleave(channels)
    return edges, batch


def batches(count, batch_size, shuffle=False):
    order = torch.randperm(count) if shuffle else torch.arange(count)
    chunks = list(order.split(batch_size))
    # BatchNorm needs more than one regression example; merge a singleton tail.
    if len(chunks) > 1 and len(chunks[-1]) == 1:
        chunks[-2] = torch.cat((chunks[-2], chunks[-1]))
        chunks.pop()
    return chunks


def metrics(prediction, target):
    errors = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    distance = np.linalg.norm(errors, axis=1)
    return dict(n=len(errors), mse_per_coordinate=float(np.mean(errors ** 2)),
                rmse_per_coordinate=float(np.sqrt(np.mean(errors ** 2))),
                rmse_2d=float(np.sqrt(np.mean(distance ** 2))),
                mean_euclidean_distance=float(distance.mean()))


def evaluate(model, x, y_raw, batch_size, corrected, target_mean, target_scale, edge_cache, order=None):
    model.eval()
    device = next(model.parameters()).device
    if order is None:
        order = torch.arange(len(x))
    prediction = np.empty((len(x), 2), dtype=np.float32)
    with torch.no_grad():
        for ix in order.split(batch_size):
            key = (corrected, len(ix))
            if key not in edge_cache:
                edge_cache[key] = edge_batch(x.shape[1], len(ix), corrected, device)
            out, _ = model(x[ix].to(device), *edge_cache[key])
            prediction[ix.numpy()] = out.detach().float().cpu().numpy() * target_scale + target_mean
    return metrics(prediction, y_raw), prediction


def make_model(name, ns, hidden):
    if name == 'corrected':
        return CorrectedRegressionHead(CorrectedEncoder(hidden=hidden))
    encoder = ns['BiLSTMEncoder']()
    model = ns['RegressionHead'](encoder)
    if name == 'joint_only':
        # Same modules, initialization, graph, loss and batch order as published.
        # The sole model change is removal of the encoder no_grad context.
        def joint_forward(self, x, edge_index, batch):
            features_ = self.encoder(x, edge_index, batch)
            return self.mlp(features_.view(features_.size(0), -1)), F.normalize(features_, dim=1)
        from types import MethodType
        model.forward = MethodType(joint_forward, model)
    return model


def verify_behavior(ns, hidden, outdir, device):
    seed_all(2718)
    x = torch.randn(4, 40, 500, device=device)
    target = torch.randn(4, 2, device=device)
    labels = torch.tensor([0, 0, 1, 1], device=device)
    results = {}
    reference = None
    for name in ('published', 'joint_only', 'corrected'):
        seed_all(2718)
        corrected = name == 'corrected'
        model = make_model(name, ns, hidden)
        if name == 'published':
            reference = {k: v.clone() for k, v in model.state_dict().items()}
        elif name == 'joint_only':
            assert all(torch.equal(v, reference[k]) for k, v in model.state_dict().items())
        model = model.to(device)
        encoder = model.encoder
        edges, batch = edge_batch(40, 4, corrected, device)
        model.train()
        out, features = model(x, edges, batch)
        cluster_fn = supervised_contrastive_loss if corrected else ns['supervised_contrastive_loss']
        cluster = cluster_fn(features, labels)
        (F.mse_loss(out, target) + cluster).backward()
        grad_count = sum(p.grad is not None for p in encoder.parameters())
        model.eval()
        changed = x.clone()
        changed[1:] = torch.randn_like(changed[1:]) * 3
        permutation = torch.tensor([2, 0, 3, 1], device=device)
        with torch.no_grad():
            original, _ = model(x, edges, batch)
            alternate, _ = model(changed, edges, batch)
            permuted, _ = model(x[permutation], edges, batch)
            solo, _ = model(x[:1], *edge_batch(40, 1, corrected, device))
        results[name] = dict(parameters=sum(p.numel() for p in model.parameters()),
                             encoder_parameters_with_grad=grad_count,
                             encoder_parameter_tensors=len(list(encoder.parameters())),
                             cluster_loss_requires_grad=cluster.requires_grad,
                             companion_max_change=float((original[0] - alternate[0]).abs().max()),
                             permutation_max_change=float((original - permuted[torch.argsort(permutation)]).abs().max()),
                             singleton_max_change=float((original[0] - solo[0]).abs().max()))
        if name == 'published':
            assert grad_count == 0 and not cluster.requires_grad
        else:
            assert grad_count > 0 and cluster.requires_grad
        if corrected:
            assert max(results[name][k] for k in ('companion_max_change', 'permutation_max_change', 'singleton_max_change')) < (1e-4 if device.type == 'cuda' else 1e-5)
    write_json(outdir / 'behavior_checks.json', results)
    return results


def train_one(name, ns, data, args, target_mean, target_scale, outdir):
    corrected = name == 'corrected'
    seed_all(args.seed)
    device = resolve_device(args.device)
    model = make_model(name, ns, args.hidden).to(device)
    encoder = model.encoder
    cluster_fn = supervised_contrastive_loss if corrected else ns['supervised_contrastive_loss']
    edge_cache = {}
    synchronize(device)
    start = time.monotonic()
    history = []
    model_dir = outdir / name
    model_dir.mkdir(exist_ok=True)
    csv_path = model_dir / 'history.csv'
    csv_file = csv_path.open('w', newline='')
    writer = csv.DictWriter(csv_file, fieldnames=['phase', 'epoch', 'train_loss', 'val_mse', 'val_rmse_2d', 'elapsed_seconds'])
    writer.writeheader()
    x, y, clusters = data['train']['x'], data['train']['y'], data['train']['cluster']
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.pretrain_lr)
    for epoch in range(args.pretrain_epochs):
        encoder.train()
        total, seen = 0., 0
        for ix in batches(len(x), args.batch_size, shuffle=corrected):
            key = (corrected, len(ix))
            if key not in edge_cache:
                edge_cache[key] = edge_batch(40, len(ix), corrected, device)
            optimizer.zero_grad(set_to_none=True)
            feature = encoder(x[ix].to(device), *edge_cache[key])
            loss = cluster_fn(feature, clusters[ix].to(device))
            if not torch.isfinite(loss):
                raise ValueError(f'{name} non-finite pretrain loss')
            loss.backward()
            optimizer.step()
            total += loss.item() * len(ix)
            seen += len(ix)
        row = dict(phase='pretrain', epoch=epoch + 1, train_loss=total / seen,
                   val_mse='', val_rmse_2d='', elapsed_seconds=time.monotonic() - start)
        history.append(row)
        writer.writerow(row)
        csv_file.flush()
        print(name, row, flush=True)
    # Clear pretrain gradients before checking joint-training gradients.
    model.zero_grad(set_to_none=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.regression_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=.1, patience=5, min_lr=args.min_lr)
    best_val, selected_epoch, encoder_grad = float('inf'), None, None
    for epoch in range(args.regression_epochs):
        if epoch and time.monotonic() - start >= args.model_time_budget:
            print(name, 'time budget reached after completed epoch', epoch, flush=True)
            break
        model.train()
        total, seen = 0., 0
        for ix in batches(len(x), args.batch_size, shuffle=corrected):
            key = (corrected, len(ix))
            if key not in edge_cache:
                edge_cache[key] = edge_batch(40, len(ix), corrected, device)
            optimizer.zero_grad(set_to_none=True)
            out, feature = model(x[ix].to(device), *edge_cache[key])
            loss = F.mse_loss(out, y[ix].to(device)) + args.contrastive_weight * cluster_fn(feature, clusters[ix].to(device))
            if not torch.isfinite(loss):
                raise ValueError(f'{name} non-finite regression loss')
            loss.backward()
            if encoder_grad is None:
                encoder_grad = sum(p.grad is not None for p in encoder.parameters())
            optimizer.step()
            total += loss.item() * len(ix)
            seen += len(ix)
        val, _ = evaluate(model, data['val']['x'], data['val']['raw_y'], args.batch_size,
                          corrected, target_mean, target_scale, edge_cache)
        # Both models use only validation data for selection. Test-selection
        # contamination in the original is documented, not used for this comparison.
        scheduler.step(val['mse_per_coordinate'] / target_scale ** 2)
        if val['mse_per_coordinate'] < best_val:
            best_val = val['mse_per_coordinate']
            selected_epoch = epoch + 1
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, model_dir / 'best_validation.pt')
        row = dict(phase='regression', epoch=epoch + 1, train_loss=total / seen,
                   val_mse=val['mse_per_coordinate'], val_rmse_2d=val['rmse_2d'],
                   elapsed_seconds=time.monotonic() - start)
        history.append(row)
        writer.writerow(row)
        csv_file.flush()
        print(name, row, flush=True)
    csv_file.close()
    model.load_state_dict(torch.load(model_dir / 'best_validation.pt', map_location=device, weights_only=True))
    test, pred = evaluate(model, data['test']['x'], data['test']['raw_y'], args.batch_size,
                          corrected, target_mean, target_scale, edge_cache)
    order = torch.from_numpy(np.random.default_rng(args.seed + 1).permutation(len(pred)))
    permuted_metrics, reordered_pred = evaluate(model, data['test']['x'], data['test']['raw_y'],
                                               args.batch_size, corrected, target_mean, target_scale, edge_cache, order)
    np.savez(model_dir / 'test_predictions.npz', indices=data['test']['indices'],
             truth=data['test']['raw_y'], prediction=pred, reordered_prediction=reordered_pred)
    synchronize(device)
    result = dict(device=str(device), model=name, parameters=sum(p.numel() for p in model.parameters()),
                  completed_pretrain_epochs=args.pretrain_epochs,
                  completed_regression_epochs=sum(row['phase'] == 'regression' for row in history),
                  regression_encoder_parameter_tensors_with_grad=encoder_grad,
                  selected_by='minimum_validation_mse', selected_epoch=selected_epoch,
                  test=test, reordered_test=permuted_metrics,
                  reordered_prediction_max_abs_change=float(np.max(np.abs(pred - reordered_pred))),
                  elapsed_seconds=time.monotonic() - start,
                  history_file=str(csv_path))
    write_json(model_dir / 'result.json', result)
    print('RESULT', json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:N; explicit CUDA never falls back to CPU')
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--out', type=Path, default=ROOT / 'runs' / 'full')
    parser.add_argument('--train-cap', type=int, default=0)
    parser.add_argument('--val-cap', type=int, default=0)
    parser.add_argument('--test-cap', type=int, default=0)
    parser.add_argument('--pretrain-epochs', type=int, default=100)
    parser.add_argument('--regression-epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--hidden', type=int, default=500)
    parser.add_argument('--pretrain-lr', type=float, default=1e-3)
    parser.add_argument('--regression-lr', type=float, default=1e-3)
    parser.add_argument('--min-lr', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--normalize-eeg', action='store_true')
    parser.add_argument('--normalize-target', action='store_true')
    parser.add_argument('--contrastive-weight', type=float, default=1.)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--prepare-only', action='store_true', help='Read labels and write ID split; no training or EEG sample reads')
    parser.add_argument('--variant', choices=['published', 'joint_only', 'corrected', 'ablation', 'comparison', 'all'], default='corrected')
    parser.add_argument('--model-time-budget', type=float, default=float('inf'))
    args = parser.parse_args()
    if args.batch_size < 2 or args.threads < 1 or args.hidden < 1:
        raise ValueError('Require batch_size >= 2, threads >= 1 and hidden >= 1')
    if args.pretrain_epochs < 0 or args.regression_epochs < 1:
        raise ValueError('Require nonnegative pretrain epochs and positive regression epochs')
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    if args.verify_only:
        device = resolve_device(args.device)
        print('DEVICE', json.dumps(device_report(device)), flush=True)
        behavior = verify_behavior(load_original(), args.hidden, args.out, device)
        print('BEHAVIOR', json.dumps(behavior), flush=True)
        return
    eeg, labels = mmap_eeg(args.data)
    split = make_splits(labels, (args.train_cap, args.val_cap, args.test_cap))
    report = split_report(labels, split)
    write_json(args.out / 'split_report.json', report)
    np.savez(args.out / 'split_indices.npz', **split)
    print('SPLIT', json.dumps(report), flush=True)
    if args.prepare_only:
        return
    if any(len(v) < 2 for v in split.values()):
        raise ValueError('Each split needs at least two selected examples')
    # Avoid silently overwriting prior selected checkpoints or results.
    if (args.out / 'summary.json').exists() or any((args.out / n / 'best_validation.pt').exists() for n in ('published', 'joint_only', 'corrected')):
        raise FileExistsError('Choose a fresh --out directory for each training run')
    device = resolve_device(args.device)
    print('DEVICE', json.dumps(device_report(device)), flush=True)
    ns = load_original()
    # The author-selected channels are unavailable. This common, explicit subset
    # is neither a recovered author list nor the broken attention selector.
    channels = np.linspace(0, 128, 40, dtype=int)
    data = {}
    for name, indices in split.items():
        print('LOAD', name, len(indices), flush=True)
        raw = labels[indices, 1:].astype(np.float32)
        data[name] = dict(indices=indices, x=read_subset(eeg, indices, channels), raw_y=raw)
    del eeg
    seed_all(args.seed)
    kmeans = KMeans(n_clusters=25, random_state=args.seed, n_init=10)
    kmeans.fit(data['train']['raw_y'])
    for name in data:
        # Only training labels are used for training SupCon. Validation/test
        # cluster labels are not needed or accessed during model training.
        data[name]['cluster'] = torch.from_numpy(kmeans.predict(data[name]['raw_y'])) if name == 'train' else None
    if args.normalize_eeg:
        mean = data['train']['x'].mean(dim=(0, 2), keepdim=True)
        std = data['train']['x'].std(dim=(0, 2), keepdim=True).clamp_min(1e-6)
        for name in data:
            data[name]['x'] = (data[name]['x'] - mean) / std
        np.savez(args.out / 'eeg_scaler.npz', mean=mean.numpy(), std=std.numpy())
    target_mean = data['train']['raw_y'].mean(0) if args.normalize_target else np.zeros(2, dtype=np.float32)
    target_scale = float(data['train']['raw_y'].std()) if args.normalize_target else 1.
    for name in data:
        data[name]['y'] = torch.from_numpy((data[name]['raw_y'] - target_mean) / target_scale)
    np.savez(args.out / 'data_manifest.npz', channels=channels, cluster_centers=kmeans.cluster_centers_,
             target_mean=target_mean, target_scale=target_scale, **split)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(source_sha256=hashlib.sha256((ROOT / 'original_snapshot/main.py').read_bytes()).hexdigest(),
                  dataset_bytes=args.data.stat().st_size, labels_shape=list(labels.shape),
                  eeg_shape=[len(labels), 500, 129],
                  total_ids=len(np.unique(labels[:, 0])),
                  selected_counts={k: len(v) for k, v in split.items()},
                  selected_id_counts={k: len(np.unique(labels[v, 0])) for k, v in split.items()},
                  units='original label coordinate units; no unverified mm conversion',
                  numpy_version=np.__version__, runtime=device_report(device),
                  baseline_adaptations=['replacement loader', 'common explicit 40-channel subset',
                    'train-only KMeans labels reconstructed', 'no unexplained hardcoded sample removal',
                    'both models selected on validation only', 'all selected tail samples retained',
                    'train-only normalization for both models if explicitly enabled',
                    'seed set before initialization', 'explicit CPU/CUDA device and per-batch transfer'],
                  corrected_changes=['independent per-sample/electrode temporal sequences',
                    '20 chronological patches of 25 time points', 'temporal hidden width 64 by default',
                    'channel attention preserves temporal value positions', 'bidirectional complete electrode graph',
                    'joint MSE/SupCon gradients', 'ignore anchors with no positives', 'shuffle training examples'],
                  split_protocol=report,
                  limitation='Public EEGViT ID split; participant mapping unverified. Corrected architecture differs. Published/joint_only provide a model-gradient ablation; no claim of paper reproduction.')
    write_json(args.out / 'config.json', config)
    mean_prediction = np.tile(data['train']['raw_y'].mean(0), (len(split['test']), 1))
    mean_baseline = metrics(mean_prediction, data['test']['raw_y'])
    write_json(args.out / 'training_mean_baseline.json', mean_baseline)
    names = {'comparison': ('published', 'corrected'),
             'ablation': ('published', 'joint_only'),
             'all': ('published', 'joint_only', 'corrected')}.get(args.variant, (args.variant,))
    results = {}
    for name in names:
        results[name] = train_one(name, ns, data, args, target_mean, target_scale, args.out)
        write_json(args.out / 'summary.json', dict(status='completed' if len(results) == len(names) else 'partial',
                   training_mean_baseline=mean_baseline, **results))



if __name__ == '__main__':
    main()
