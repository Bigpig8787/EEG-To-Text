"""CLIP-style EEG encoder for contrastive pre-training.

Wraps the existing ConformerEncoder with a projection head into the BART
text-encoder embedding space. Encoder state dict is compatible with
`models.multiview` `load_pretrained_encoder`, so saved weights can be reused.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.conformer import ConformerEncoder


class ConformerCLIPModel(nn.Module):
    def __init__(self, n_channels=105, d_model=512, n_filters=40,
                 temporal_kernel=200, pool_stride=100,
                 n_heads=8, n_transformer_layers=4,
                 dropout=0.1, proj_dim=1024):
        super().__init__()
        self.encoder = ConformerEncoder(
            n_channels=n_channels, d_model=d_model, n_filters=n_filters,
            temporal_kernel=temporal_kernel, pool_stride=pool_stride,
            n_heads=n_heads, n_transformer_layers=n_transformer_layers,
            dropout=dropout,
        )
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, proj_dim),
        )

    def encode(self, x_masked, mask=None):
        h = self.encoder(x_masked, mask=mask)  # (B, T', d_model)
        # Mean-pool over time. If mask provided over original T, we ignore for
        # simplicity — encoder's downsampling collapses time, so a uniform mean
        # is a reasonable approximation.
        pooled = h.mean(dim=1)
        z = self.proj(pooled)
        return F.normalize(z, dim=-1)

    def forward(self, x_masked, mask=None):
        return self.encode(x_masked, mask=mask)

    def load_encoder_weights(self, encoder_path, strict=True):
        sd = torch.load(encoder_path, map_location='cpu')
        if isinstance(sd, dict) and 'model_state_dict' in sd:
            sd = sd['model_state_dict']
            sd = {k.replace('encoder.', '', 1): v for k, v in sd.items() if k.startswith('encoder.')}
        missing, unexpected = self.encoder.load_state_dict(sd, strict=strict)
        return missing, unexpected
