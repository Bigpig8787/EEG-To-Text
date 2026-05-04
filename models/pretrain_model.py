"""Pre-training model: Conformer Encoder + CNN Decoder + Re-Masked Token Prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.conformer import ConformerEncoder


class ConformerDecoder(nn.Module):
    def __init__(self, n_channels=105, d_model=512, n_filters=40,
                 pool_stride=100, target_T=5000):
        super().__init__()
        self.target_T = target_T
        self.projection = nn.Linear(d_model, n_filters)
        self.temporal_deconv = nn.Sequential(
            nn.ConvTranspose1d(n_filters, n_filters, kernel_size=pool_stride, stride=pool_stride),
            nn.ELU(),
        )
        self.channel_reconstruct = nn.Conv1d(n_filters, n_channels, kernel_size=1)

    def forward(self, x):
        x = self.projection(x)
        x = x.permute(0, 2, 1)
        x = self.temporal_deconv(x)
        if x.shape[2] > self.target_T:
            x = x[:, :, :self.target_T]
        elif x.shape[2] < self.target_T:
            x = F.pad(x, (0, self.target_T - x.shape[2]))
        x = self.channel_reconstruct(x)
        return x


class ConformerPreTrainModel(nn.Module):
    def __init__(self, n_channels=105, d_model=512, n_filters=40,
                 temporal_kernel=200, pool_stride=100,
                 n_heads=8, n_transformer_layers=4,
                 dropout=0.1, target_T=5000):
        super().__init__()
        self.encoder = ConformerEncoder(
            n_channels=n_channels, d_model=d_model, n_filters=n_filters,
            temporal_kernel=temporal_kernel, pool_stride=pool_stride,
            n_heads=n_heads, n_transformer_layers=n_transformer_layers,
            dropout=dropout
        )
        self.decoder = ConformerDecoder(
            n_channels=n_channels, d_model=d_model, n_filters=n_filters,
            pool_stride=pool_stride, target_T=target_T
        )

    def forward(self, x_masked, mask=None):
        return self.decoder(self.encoder(x_masked, mask=mask))


def create_remask(x, mask_ratio=0.15):
    """Re-Masked Token Prediction: randomly mask time points each call."""
    batch, C, T = x.shape
    num_mask = int(T * mask_ratio)
    mask = torch.zeros(batch, 1, T, dtype=torch.bool, device=x.device)
    for i in range(batch):
        indices = torch.randperm(T, device=x.device)[:num_mask]
        mask[i, 0, indices] = True
    x_masked = x.clone()
    x_masked[mask.expand_as(x)] = 0.0
    return x_masked, mask
