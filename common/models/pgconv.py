from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class PseudoGraphConv2d(nn.Module):
    def __init__(self, in_channels, dil):
        super().__init__()
        self.theta = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.phi = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False, dilation=dil)
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.GELU()

    def forward(self, x):
        theta_x = self.theta(x)
        phi_x = self.phi(x)
        edge_feat = phi_x - theta_x
        out = self.bn(edge_feat + theta_x)
        return self.act(out)


class PseudoGraphConv1d(nn.Module):
    def __init__(self, in_channels, dil):
        super().__init__()
        self.theta = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False)
        self.phi = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False, dilation=dil)
        self.bn = nn.BatchNorm1d(in_channels)
        self.act = nn.GELU()

    def forward(self, x):
        theta_x = self.theta(x)
        phi_x = self.phi(x)
        edge_feat = phi_x - theta_x
        out = self.bn(edge_feat + theta_x)
        return self.act(out)


class PGConv1d(nn.Module):
    def __init__(self, dim, dilation, expansion_ratio=4/3, kernel_size=7, conv_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 act_layer=nn.GELU,
                 drop_path=0.0):
        super().__init__()

        self.norm = norm_layer(dim)
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.act = act_layer()

        conv_channels = int(conv_ratio * dim)
        padding = int((kernel_size - 1) / 2) * dilation
        self.split_indices = (hidden, hidden - conv_channels, conv_channels)
        self.enhanced_conv = PseudoGraphConv1d(conv_channels, dilation)
        self.fc2 = nn.Linear(hidden, dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.res_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        shortcut = x
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=-1)
        c = c.permute(0, 2, 1)
        c = self.enhanced_conv(c)
        c = c.permute(0, 2, 1)
        x = self.fc2(self.act(g) * torch.cat((i, c), dim=-1))
        x = self.drop_path(x)
        x = x + shortcut
        x = x.permute(0, 2, 1)
        return x


class PGConv2d(nn.Module):
    def __init__(self, dim, dilation, expansion_ratio=4/3, kernel_size=7, conv_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 act_layer=nn.GELU,
                 drop_path=0.0):
        super().__init__()

        self.norm = norm_layer(dim)
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.act = act_layer()

        conv_channels = int(conv_ratio * dim)
        padding = int((kernel_size - 1) / 2) * dilation
        self.split_indices = (hidden, hidden - conv_channels, conv_channels)
        self.enhanced_conv = PseudoGraphConv2d(conv_channels, dilation)
        self.fc2 = nn.Linear(hidden, dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.res_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        shortcut = x
        x = self.norm(x)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=-1)
        c = c.permute(0, 3, 1, 2)
        c = self.enhanced_conv(c)
        c = c.permute(0, 2, 3, 1)
        x = self.fc2(self.act(g) * torch.cat((i, c), dim=-1))
        x = self.drop_path(x)
        x = x + shortcut
        x = x.permute(0, 3, 2, 1)

        return x
