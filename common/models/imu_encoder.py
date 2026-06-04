import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.init import kaiming_normal_, orthogonal_
import numpy as np
from torch.distributions.utils import broadcast_all, probs_to_logits, logits_to_probs, lazy_property, clamp_probs
import torch.nn.functional as F
import models.depth_encoder
from timm.models.layers import DropPath
import math
from models.basic_modules import ConvNormAct1d, ConvNorm1d
from models.pgconv import PGConv1d
import torch.nn as nn
from scipy.sparse import csgraph
from scipy.linalg import eigh


class GraphPositionalEncoding1D(nn.Module):
    def __init__(self, g_dim=1, dim=768):
        super().__init__()
        self.g_dim = g_dim
        self.dim = dim
        self.proj = nn.Conv1d(g_dim, dim, kernel_size=1)
        self.cached_L = None
        self.cached_eigvec = None

    def build_adjacency(self, L):
        adj = np.zeros((L, L), dtype=np.float32)
        for i in range(L):
            if i > 0:     adj[i, i - 1] = 1
            if i < L - 1: adj[i, i + 1] = 1
        adj = (adj + adj.T) / 2
        return adj

    def compute_laplacian_eigvec(self, adj, k):
        L = csgraph.laplacian(adj, normed=True)
        eigvals, eigvecs = eigh(L)
        eigvecs_k = eigvecs[:, 1:k+1]
        return eigvecs_k

    def forward(self, B, L):
        device = self.proj.weight.device
        dtype = self.proj.weight.dtype

        if self.cached_L != L:
            adj = self.build_adjacency(L)
            eigvecs = self.compute_laplacian_eigvec(adj, self.g_dim)
            eigvecs = torch.tensor(eigvecs.T, device=device).to(dtype)
            self.cached_eigvec = eigvecs
            self.cached_L = L

        pos_enc = self.cached_eigvec.unsqueeze(0).repeat(B, 1, 1).contiguous()
        pos_enc = self.proj(pos_enc)
        return pos_enc


class Former1d(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=6,
                 use_pos_emb=True, num_heads=6, qkv_bias=True, attn_drop=0., drop=0.):
        super().__init__()

        self.dim = dim
        self.pos_embd = None
        self.expan_ration = 1
        if use_pos_emb:
            self.pos_embd = GraphPositionalEncoding1D(dim=self.dim)

        self.norm_xca = models.depth_encoder.LayerNorm(self.dim, eps=1e-6)

        self.gamma_xca = nn.Parameter(layer_scale_init_value * torch.ones(self.dim),
                                      requires_grad=True) if layer_scale_init_value > 0 else None
        self.xca = models.depth_encoder.XCA(self.dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.pwconv1 = nn.Linear(self.dim, expan_ratio * self.dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expan_ratio * self.dim, self.dim)

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((self.dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def forward(self, x):
        input_ = x
        B, C, L = x.shape

        if self.pos_embd:
            pos_encoding = self.pos_embd(B, L)
            x = x + pos_encoding

        x = x.permute(0, 2, 1)

        x = x + self.gamma_xca * self.xca(self.norm_xca(x))
        
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 2, 1).contiguous()

        x = input_ + self.drop_path(x)

        return x



class InertialEncoder(nn.Module):
    def __init__(self, in_chans=6, height=256, width=320,
                 global_block=[1, 1, 1], global_block_type=['Former1d', 'Former1d', 'Former1d'],
                 drop_path_rate=0.2, layer_scale_init_value=1e-6, expan_ratio=6,
                 heads=[8, 8, 8], use_pos_embd_xca=[True, False, False], **kwargs):

        super().__init__()

        self.num_ch_enc = np.array([48, 80, 128])
        self.imu = [1, 1, 2]
        self.dims = [48, 80, 128]
        self.dilation = [[1], [1], [1]]

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, 2)]

        self.stages = nn.ModuleList()
        cur = 0
        for i in range(3):
            stage_blocks = []
            for j in range(self.imu[i]):
                if j > self.imu[i] - global_block[i] - 1:
                    if global_block_type[i] == 'Former1d':
                        stage_blocks.append(Former1d(dim=self.dims[i], drop_path=dp_rates[0],
                                                 expan_ratio=expan_ratio,
                                                 use_pos_emb=use_pos_embd_xca[i], num_heads=heads[i],
                                                 layer_scale_init_value=layer_scale_init_value,
                                                 ))

                    else:
                        raise NotImplementedError
                else:
                    stage_blocks.append(MonaGatedBlock1d(dim=self.dims[i], dilation=self.dilation[i][j], kernel_size=3, drop_path=dp_rates[0]))

            self.stages.append(nn.Sequential(*stage_blocks))
            cur += self.imu[i]

        self.conv1 = nn.Conv1d(self.dims[0], self.dims[1], kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv1d(self.dims[1], self.dims[2], kernel_size=3, padding=1, bias=False)

        self.encoder_conv = nn.Sequential(
            nn.Conv1d(6, 48, kernel_size=3, padding=1),
            nn.BatchNorm1d(48),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0),
            )
        self.proj = nn.Sequential(
            nn.Linear(128 * 1 * 2, 256),
        )
        self.drop_out = nn.Dropout(0.1)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, 2, -1)

        x = self.encoder_conv(x.permute(0, 2, 1).float())

        for i in range(0, 3):
            for s in range(len(self.stages[i]) - 1):
                x = self.stages[i][s](x)

            x = self.stages[i][-1](x)
                    
            if i == 0:
                x = self.conv1(x)
            elif i == 1:
                x = self.conv2(x)

        x = x.permute(0, 2, 1)

        out = self.proj(x.view(x.shape[0], -1))
        return out.view(batch_size, 256)
