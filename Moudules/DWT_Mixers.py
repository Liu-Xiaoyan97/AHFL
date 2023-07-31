#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/7/21 11:01
# @Author  : lxy15058247683@aliyun.com
# @FileName: DWT_Mixers.py
# @Copyright: MIT
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
# from DWT import DWT1DForward,DWT1DInverse
from pytorch_wavelets import DWT1DForward, DWT1DInverse

class DWT_MIXER(nn.Module):
    """vocab_size, n_heads, max_seq_len, hidden_dim, index, kernel_size, dilation, padding, num_mixers,
                 num_classes"""
    def __init__(self, vocab_size: int, num_mixers: int, max_seq_len: int, hidden_dim: int,
                 mlp_hidden_dim: int, num_classes, **kwargs):
        super(DWT_MIXER, self).__init__(**kwargs)
        self.hidden_dim=hidden_dim
        self.max_len=max_seq_len
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_dim)
        self.mixers = nn.Sequential(*[
            MixerLayer(max_seq_len, hidden_dim, mlp_hidden_dim, mlp_hidden_dim) for _ in range(num_mixers)
        ])
        self.classification = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.embedding(inputs)
        outputs = self.mixers(inputs)
        means = outputs.mean(dim=-1, keepdim=True).mean(1)
        output = self.classification(outputs.mean(1) - means)
        return output


class MixerLayer(nn.Module):

    def __init__(self, max_seq_len: int, hidden_dim: int, channel_hidden_dim: int, seq_hidden_dim: int, **kwargs):
        super(MixerLayer, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.FC1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.FC3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.xfm = DWT1DForward(J=1, mode='zero', wave='haar')
        self.ifm = DWT1DInverse(mode='zero', wave='haar')
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.mlp_2 = MlpLayer(hidden_dim, channel_hidden_dim)

        self.scale = hidden_dim ** -0.5

    def forward(self, x):
        input =x
        B,N,D=input.shape
        x1 = self.FC1(input)
        x_dwt0,x_dwt1 = self.xfm(x1.permute(0,2,1))
        x_shift1 = torch.roll(x1, (0, 0, -1), (1, 1, 1))
        # x_shift2 = torch.roll(x1, (0, 0, 1), (1, 1, 1))
        x_shift2 = torch.roll(x1, (0, 0, 1), (1, 1, 1))
        # x_resu = x1+x_shift1
        x_dwt3, x_dwt4 = self.xfm(x_shift1.permute(0, 2, 1))
        x_dwt5, x_dwt6 = self.xfm(x_shift2.permute(0, 2, 1))
        # x_dwt5, x_dwt6 = self.xfm(x_shift1.permute(0, 2, 1))
        x_resu1=torch.stack([x_dwt0.permute(0,2,1), x_dwt3.permute(0,2,1)], 2).reshape(-1, N, D)
        x_resu2=torch.stack([x_dwt5.permute(0,2,1), x_dwt0.permute(0,2,1)], 2).reshape(-1, N, D)
        resu =self.FC3(x_resu1+x_resu2)
        redus = resu + x1
        residual = redus
        outputs = self.layer_norm(residual)
        outputs = self.mlp_2(outputs) + residual
        return outputs


class MlpLayer(nn.Module):

    def __init__(self, hidden_dim: int, intermediate_dim: int, **kwargs):
        super(MlpLayer, self).__init__(**kwargs)
        self.layers = nn.Sequential(*[
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, hidden_dim)
        ])
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)
