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
                      padding='same', bias=False),
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
        """Encode raw EEG, optionally under an MAE mask.

        Follows the same convention as the SNN `LocalTransformerV2`: positional
        encoding is added *before* masking, and masked tokens are then **dropped
        from the sequence** rather than zeroed. Every surviving token therefore
        still carries the absolute position it held in the full sequence, and the
        transformer only ever attends over real signal.

        Args:
            x: (B, C, T). Masked time points are already zeroed by `create_remask`.
            mask: (B, 1, T) bool, True = masked. `None` for fine-tuning / inference.

        Returns:
            mask is None: (B, L, d_model) with L = T // pool_stride.
            mask given:   ((B, L_visible, d_model), keep), where `keep` is a
                (B, L) bool marking which latent slots survived. The decoder needs
                it to put the visible tokens back in place.
        """
        x = x.unsqueeze(1)
        x = self.temporal_conv(x)
        if mask is not None:
            # The convolutions are 200 samples wide, so visible signal bleeds into
            # masked positions. Re-zero after each one to keep the masked windows
            # information-free.
            x = x.masked_fill(mask.unsqueeze(1), 0.0)
        x = self.spatial_conv(x)
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(1), 0.0)
        x = self.pool(x)
        x = self.dropout(x)
        x = x.squeeze(2).permute(0, 2, 1)
        x = self.projection(x)

        x = self.pos_encoding(x)

        if mask is None:
            return self.transformer(x)

        # A pool window survives only if *none* of its `pool_stride` time points
        # was masked. `create_remask` masks whole pool-aligned blocks, and the same
        # number of them for every sample in the batch, so `keep.sum(1)` is uniform
        # and the reshape below is well defined.
        window_masked = F.max_pool1d(
            mask.float().squeeze(1),
            kernel_size=self.pool_stride, stride=self.pool_stride,
        ).bool()                                   # (B, L) True = window touched by mask
        keep = ~window_masked                      # (B, L) True = fully visible
        batch, _, d_model = x.shape
        x = x[keep].reshape(batch, -1, d_model)    # (B, L_visible, d_model)
        x = self.transformer(x)
        return x, keep
