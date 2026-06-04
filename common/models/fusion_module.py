import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_, orthogonal_
import numpy as np
from torch.distributions.utils import broadcast_all, probs_to_logits, logits_to_probs, lazy_property, clamp_probs
import torch.nn.functional as F
import models
from timm.models.layers import DropPath
from models.basic_modules import ConvNormAct1d, ConvNorm1d
from torch_geometric.nn import GATv2Conv

class EdgePredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x_i, x_j):
        edge_input = torch.cat([x_i, x_j], dim=-1)
        return self.mlp(edge_input)

class ModalityGraphFusion(nn.Module):
    def __init__(self, feat_dim=256, num_modalities=3):
        super().__init__()
        self.edge_predictor = EdgePredictor(feat_dim)
        self.gnn = GATv2Conv(feat_dim, feat_dim, heads=1, concat=False, edge_dim=1)
        self.num_modalities = num_modalities

    def forward(self, x, y):
        node_feats = torch.stack([x, y],dim=1)
        B, N, D = node_feats.shape
        x = node_feats.reshape(B * N, D)

        device = x.device
        row, col = torch.meshgrid(torch.arange(N), torch.arange(N), indexing="ij")
        mask = row != col
        row = row[mask].flatten()
        col = col[mask].flatten()

        edge_index_single = torch.stack([row, col], dim=0)
        E = edge_index_single.size(1)

        edge_indices = []
        for b in range(B):
            offset = b * N
            edge_indices.append(edge_index_single + offset)
        edge_index = torch.cat(edge_indices, dim=1)
        x_i = x[edge_index[0]]
        x_j = x[edge_index[1]]
        edge_attr = self.edge_predictor(x_i, x_j)
        
        x_out = self.gnn(x, edge_index.to(device), edge_attr.to(device))
        x_out = x_out.view(B, N, D)
        x_out = x_out.mean(1)
        x_out = x_out.view(B, D)
        return x_out
