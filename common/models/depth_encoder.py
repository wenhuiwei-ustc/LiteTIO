import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
import math
import torch.cuda
from models.basic_modules import get_norm, get_act, ConvNormAct, ConvNorm2d, MSPatchEmb
from models.pgconv import PGConv2d
from functools import partial

import torch.nn as nn
from scipy.sparse import csgraph
from scipy.linalg import eigh


class GraphPositionalEncoding(nn.Module):

    def __init__(self, g_dim=16, dim=768):
        super().__init__()
        self.g_dim = g_dim
        self.dim = dim
        self.proj = nn.Conv2d(g_dim, dim, kernel_size=1)

        self.cached_adj = None
        self.cached_eigvec = None

    def build_adjacency(self, H, W):
        N = H * W
        adj = np.zeros((N, N), dtype=np.float32)
        
        for i in range(H):
            for j in range(W):
                idx = i * W + j
                if i > 0:    adj[idx, idx - W] = 1
                if i < H-1:  adj[idx, idx + W] = 1
                if j > 0:    adj[idx, idx - 1] = 1
                if j < W-1:  adj[idx, idx + 1] = 1
        adj = (adj + adj.T) / 2
        return adj

    def compute_laplacian_eigvec(self, adj, k):
        L = csgraph.laplacian(adj, normed=True)
        eigvals, eigvecs = eigh(L)
        eigvecs_k = eigvecs[:, 1:k+1]
        return eigvecs_k

    def forward(self, B, H, W):
        device = next(self.parameters()).device

        if (self.cached_adj is None) or (self.cached_adj.shape[0] != H*W):
            adj = self.build_adjacency(H, W)
            eigvecs = self.compute_laplacian_eigvec(adj, self.g_dim)
            self.cached_adj = torch.tensor(adj, device=device)
            self.cached_eigvec = torch.tensor(eigvecs, device=device)

        pos_enc = self.cached_eigvec.T.reshape(self.g_dim, H, W)

        pos_enc = pos_enc.unsqueeze(0).repeat(B, 1, 1, 1)

        pos_enc = self.proj(pos_enc)
        return pos_enc



class XCA(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]


        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class BNGELU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-5)
        self.act = nn.GELU()

    def forward(self, x):
        output = self.bn(x)
        output = self.act(output)

        return output


class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding=0, dilation=(1, 1), groups=1, bn_act=False, bias=False):
        super().__init__()

        self.bn_act = bn_act

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_act:
            self.bn_gelu = BNGELU(nOut)

    def forward(self, x):
        output = self.conv(x)

        if self.bn_act:
            output = self.bn_gelu(output)

        return output

class FormerP(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=6,
                 use_pos_emb=True, num_heads=6, qkv_bias=True, attn_drop=0., drop=0.):
        super().__init__()

        self.dim = dim
        self.pos_embd = None

        if use_pos_emb:
            self.pos_embd = GraphPositionalEncoding(dim=self.dim)

        self.norm_xca = LayerNorm(self.dim, eps=1e-6)

        self.gamma_xca = nn.Parameter(layer_scale_init_value * torch.ones(self.dim),
                                      requires_grad=True) if layer_scale_init_value > 0 else None
        self.xca = XCA(self.dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.pwconv1 = nn.Linear(self.dim, self.dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(self.dim, self.dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((self.dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input_ = x

        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)

        if self.pos_embd:
            pos_encoding = self.pos_embd(B, H, W).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
            x = x + pos_encoding

        x = x + self.gamma_xca * self.xca(self.norm_xca(x))

        x = x.reshape(B, H, W, C)

        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)

        x = input_ + self.drop_path(x)

        return x


class Former(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, expan_ratio=1,
                 use_pos_emb=True, num_heads=6, qkv_bias=True, attn_drop=0., drop=0.):
        super().__init__()

        self.dim = dim
        self.pos_embd = None

        if use_pos_emb:
            self.pos_embd = GraphPositionalEncoding(dim=self.dim)

        self.norm_xca = LayerNorm(self.dim, eps=1e-6)

        self.gamma_xca = nn.Parameter(layer_scale_init_value * torch.ones(self.dim),
                                      requires_grad=True) if layer_scale_init_value > 0 else None
        self.xca = XCA(self.dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.pwconv1 = nn.Linear(self.dim, expan_ratio * self.dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expan_ratio * self.dim, self.dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((self.dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input_ = x

        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)

        if self.pos_embd:
            pos_encoding = self.pos_embd(B, H, W).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
            x = x + pos_encoding

        x = x + self.gamma_xca * self.xca(self.norm_xca(x))

        x = x.reshape(B, H, W, C)

        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)

        x = input_ + self.drop_path(x)

        return x

class AvgPool(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, ratio):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        for pool in self.pool:
            x = pool(x)

        return x

class en(nn.Module):
    def __init__(self, in_chans, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, 8, 1)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(8, in_chans, 1)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        feat = self.act2(self.conv2(self.act1(self.conv1(x))))
        return x * feat + x


class DepthEncoder(nn.Module):
    def __init__(self, in_chans=1, height=192, width=640,
                 global_block=[1, 1, 1, 1], global_block_type=['Former', 'Former', 'Former', 'Former'],
                 drop_path_rate=0.2, layer_scale_init_value=1e-6, expan_ratio=6,
                 heads=[8, 8, 8, 8], use_pos_embd_xca=[True, False, False, False], **kwargs):

        super().__init__()


        self.num_ch_enc = np.array([32, 32, 64, 128])
        self.depth = [2, 3, 3, 5]
        self.dims = [32, 32, 64, 128]
        self.dilation = [[1, 2, 3], [1, 2, 3], [1,2,3,2,4], [1,2,3,2,4]]


        for g in global_block_type:
            assert g in ['None', 'Former']

        self.downsample_layers = nn.ModuleList()
        stem1 = nn.Sequential(
            Conv(in_chans, self.dims[0], kSize=3, stride=2, padding=1, bn_act=True),
            Conv(self.dims[0], self.dims[0], kSize=3, stride=1, padding=1, bn_act=True),
            Conv(self.dims[0], self.dims[0], kSize=3, stride=1, padding=1, bn_act=True),
        )

        self.stem2 = nn.Sequential(
            Conv(self.dims[0] + 1, self.dims[0], kSize=3, stride=2, padding=1, bn_act=False),
        )
        self.en = en(in_chans)

        self.downsample_layers.append(stem1)

        self.input_downsample = nn.ModuleList()
        for i in range(1, 5):
            self.input_downsample.append(AvgPool(i))

        for i in range(3):
            downsample_layer = nn.Sequential(
            Conv(self.dims[i] * 2 + 1, self.dims[i + 1], kSize=3, stride=2, padding=1, bn_act=False),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depth))]
        cur = 0
        self.mlp_ratios = [2,2,2]
        self.sr_ratios = [4,2,1]
        for i in range(4):
            stage_blocks = []
            for j in range(self.depth[i]):
                if j > self.depth[i] - global_block[i]-1:
                    if global_block_type[i] == 'Former':
                        stage_blocks.append(Former(dim=self.dims[i], drop_path=dp_rates[cur + j],
                                                 expan_ratio=expan_ratio,
                                                 use_pos_emb=use_pos_embd_xca[i], num_heads=heads[i],
                                                 layer_scale_init_value=layer_scale_init_value,
                                                 ))
                    else:
                        raise NotImplementedError
                else:
                    stage_blocks.append(PGConv2d(dim=self.dims[i],
                                                 dilation=self.dilation[i][j], kernel_size=3, drop_path=dp_rates[cur + j],
                                                 ))

            self.stages.append(nn.Sequential(*stage_blocks))
            cur += self.depth[i]

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        x = self.en(x)
        feature_t = x
        features = []

        x_down = []
        for i in range(4):
            x_down.append(self.input_downsample[i](x))

        tmp_x = []
        x = self.downsample_layers[0](x)
        x = self.stem2(torch.cat((x, x_down[0]), dim=1))

        tmp_x.append(x)

        input_ = x
        for s in range(len(self.stages[0]) - 1):
           x = self.stages[0][s](x)
        x = self.stages[0][-1](x)
        x = x.reshape(input_.shape)
        tmp_x.append(x)
        features.append(x)

        for i in range(1, 4):
            tmp_x.append(x_down[i])
            x = torch.cat(tmp_x, dim=1)
            x = self.downsample_layers[i](x)

            tmp_x = [x]
            input_ = x

            for s in range(len(self.stages[i]) - 1):
                x = self.stages[i][s](x)
            x = self.stages[i][-1](x)
            x = x.reshape(input_.shape)
            tmp_x.append(x)

            features.append(x)

        return features

    def forward(self, x):
        x = self.forward_features(x)
        return x
