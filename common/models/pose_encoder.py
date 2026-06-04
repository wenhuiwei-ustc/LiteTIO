import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath
import math
import torch.cuda
from models.depth_encoder import *

from torch.autograd import Variable

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

class PoseEncoder(nn.Module):

    def __init__(self, in_chans=2, height=256, width=320,
                 global_block=[1, 1, 1], global_block_type=['Former', 'Former', 'Former'],
                 drop_path_rate=0.2, layer_scale_init_value=1e-6, expan_ratio=3,
                 heads=[8, 8, 8], use_pos_embd_xca=[True, False, False], **kwargs):

        super().__init__()
        self.num_ch_enc = np.array([32, 64, 128])
        self.pose = [2, 2, 4]
        self.dims = [32, 64, 128]
        self.dilation = [[1,2], [1,2], [1, 2, 3, 4]]


        for g in global_block_type:
            assert g in ['None', 'Former']

        self.stem = nn.Sequential(
            Conv(in_chans, self.dims[0], kSize=3, stride=2, padding=1, bn_act=True),
            Conv(self.dims[0], self.dims[0], kSize=3, stride=2, padding=1, bn_act=True),
            Conv(self.dims[0], self.dims[0], kSize=3, stride=2, padding=1, bn_act=True),
        )

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.pose))]
        cur = 0
        for i in range(3):
            stage_blocks = []
            for j in range(self.pose[i]):
                if j > self.pose[i] - global_block[i]-1:
                    if global_block_type[i] == 'Former':
                        stage_blocks.append(FormerP(dim=self.dims[i], drop_path=dp_rates[cur + j],
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
            cur += self.pose[i]

        self.conv1 = Conv(self.dims[0], self.dims[1], kSize=3, stride=2, padding=1, bn_act=False)
        self.conv2 = Conv(self.dims[1], self.dims[2], kSize=3, stride=2, padding=1, bn_act=False)
        self.conv3 = Conv(self.dims[2], self.dims[2], kSize=3, stride=2, padding=1, bn_act=False)

        self.visual_headk = nn.Sequential(
            nn.Linear(2560, 256),
        )
        self.en = en(in_chans)

        self.apply(self._init_weights)

    def forward_features(self, x):
        x = self.en(x)
        x = self.stem(x)

        for i in range(0, 3):
            input_ = x
            for s in range(len(self.stages[i]) - 1):
                x = self.stages[i][s](x)
            x = self.stages[i][-1](x)
            x = x.reshape(input_.shape)

            if i == 0:
                x = self.conv1(x)
            elif i == 1:
                x = self.conv2(x)
            elif i == 2:
                x = self.conv3(x)

        features = x
        B, C, H, W = features.shape

        features = features.view(B, -1)

        features = self.visual_headk(features)
        return features

    def forward(self, x):
        x = self.forward_features(x)
        return x

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
