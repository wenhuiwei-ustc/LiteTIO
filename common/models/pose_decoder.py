import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath
import math
import torch.cuda
import models
from collections import OrderedDict

import torch.nn.init as init


class PoseDecoder(nn.Module):
    def __init__(self):
        super(PoseDecoder, self).__init__()
        f_len = 256
        
        stride = 1
        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv1d(f_len, 256, 1)
        self.convs[("pose", 0)] = nn.Conv1d(256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv1d(256, 256, 1, stride, 1)
        self.convs[("pose", 2)] = nn.Conv1d(256, 6*1, 1)

        self.relu = nn.ReLU()
        self.net = nn.ModuleList(list(self.convs.values()))


    def forward(self, fused):
        
        B, L = fused.shape
        fused = fused.view(B, L, 1)

        cat_features = self.relu(self.convs["squeeze"](fused))

        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)


        out = out.mean(2)
        out = 0.01 * out.view(-1, 6)

        return out
