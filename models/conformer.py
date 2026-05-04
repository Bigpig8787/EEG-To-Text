"""
Conformer Encoder: temporal conv → spatial conv → pooling → transformer.
Shared by pre-training and multi-view models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ConformerEncoder(nn.Module):
    """
    Input:  (batch, C, T)
    Output: (batch, T//pool_stride, d_model)
    """
    def __init__(self, n_channels=105, d_model=512, n_filters=40,
                 temporal_kernel=200, pool_stride=100,
                 n_heads=8, n_transformer_layers=4, dropout=0.1):
        super().__init__()
        self.n_channels = n_channels
        self.d_model = d_model
        self.pool_stride = pool_stride

        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, n_filters, kernel_size=(1, temporal_kernel),
                      padding=(0, temporal_kernel // 2), bias=False),
            nn.BatchNorm2d(n_filters),
        )
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(n_filters, n_filters, kernel_size=(n_channels, 1), bias=False),
            nn.BatchNorm2d(n_filters),
            nn.ELU(),
        )
        self.pool = nn.AvgPool2d(kernel_size=(1, pool_stride))
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(n_filters, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout, max_len=1000)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)

    def forward(self, x, mask=None):
        x = x.unsqueeze(1)
        x = self.temporal_conv(x)
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(1), 0.0)
        x = self.spatial_conv(x)
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(1), 0.0)
        x = self.pool(x)
        if mask is not None:
            pooled_mask = F.max_pool1d(
                mask.float().squeeze(1),
                kernel_size=self.pool_stride, stride=self.pool_stride,
            ).bool().unsqueeze(1).unsqueeze(1)
            x = x.masked_fill(pooled_mask, 0.0)
        x = self.dropout(x)
        x = x.squeeze(2).permute(0, 2, 1)
        x = self.projection(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        return x
