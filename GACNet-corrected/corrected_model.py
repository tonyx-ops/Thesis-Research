"""Independent samples, chronological per-electrode encoding, and joint gradients.

This is a corrected implementation, not a recovered unpublished author model.
25-point chronological patches keep CPU memory use bounded. All 500 points are
used in 20 steps. The temporal hidden width is explicit in experiment metadata.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATv2Conv, BatchNorm, global_add_pool


class CorrectedEncoder(nn.Module):
    def __init__(self, channels=40, hidden=64, patch_size=25):
        super().__init__()
        if 500 % patch_size:
            raise ValueError('patch_size must divide the 500 time points')
        self.channels = channels
        self.patch_size = patch_size
        self.conv11 = nn.Conv1d(channels, 64, 3, padding=1)
        self.conv21 = nn.Conv1d(64, channels, 3, padding=1)
        self.bn11 = nn.BatchNorm1d(channels)
        # Channel attention mixes channels while keeping the chronological
        # time-point values intact, rather than applying a dense time MLP.
        self.query = nn.Linear(500, 64, bias=False)
        self.key = nn.Linear(500, 64, bias=False)
        self.lstm = nn.LSTM(patch_size, hidden, 2, batch_first=True,
                            bidirectional=True)
        self.node_projection = nn.Linear(2 * hidden, 1000)
        self.conv1 = GATv2Conv(1000, 128, heads=4, concat=True)
        self.conv2 = GATv2Conv(512, 250, heads=4, concat=True)
        self.conv2_bn = BatchNorm(1000)

    def forward(self, x, edge_index, batch):
        b, c, t = x.shape
        if c != self.channels or t != 500:
            raise ValueError(f'Expected [B,{self.channels},500], got {tuple(x.shape)}')
        x = F.leaky_relu(self.conv11(x))
        x = F.leaky_relu(self.bn11(self.conv21(x)))
        weights = (self.query(x) @ self.key(x).transpose(-1, -2) / 8).softmax(-1)
        x = x + weights @ x
        # Each (sample, electrode) is an independent recurrent sequence.
        sequences = x.reshape(b * c, t // self.patch_size, self.patch_size)
        _, (hidden, _) = self.lstm(sequences)
        nodes = self.node_projection(torch.cat((hidden[-2], hidden[-1]), -1))
        nodes = F.relu(self.conv1(nodes, edge_index))
        nodes = F.relu(self.conv2_bn(self.conv2(nodes, edge_index)))
        return F.normalize(global_add_pool(nodes, batch), dim=1)


class CorrectedRegressionHead(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.mlp = nn.Sequential(nn.Linear(1000, 512), nn.BatchNorm1d(512),
                                 nn.ReLU(), nn.Linear(512, 2))
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)

    def forward(self, x, edge_index, batch):
        features = self.encoder(x, edge_index, batch)
        return self.mlp(features), features


def supervised_contrastive_loss(features, labels, temperature=0.07):
    labels = labels.reshape(-1)
    logits = F.normalize(features, dim=1) @ F.normalize(features, dim=1).T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    diagonal = torch.eye(len(labels), dtype=torch.bool, device=features.device)
    positives = labels[:, None].eq(labels[None, :]) & ~diagonal
    counts = positives.sum(1)
    usable = counts > 0
    if not usable.any():
        return features.sum() * 0
    log_prob = logits - torch.logsumexp(logits.masked_fill(diagonal, -torch.inf), 1, keepdim=True)
    return -(log_prob.masked_fill(~positives, 0).sum(1)[usable] / counts[usable]).mean()
