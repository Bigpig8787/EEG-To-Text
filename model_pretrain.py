"""
EEG Pre-Training with Conformer Encoder and Re-Masked Token Prediction.

Based on EEG2TEXT (Liu et al., 2024) Section 3.3 and EEG Conformer (Song et al., 2022).

Architecture:
    Conformer Encoder (temporal conv → spatial conv → pool → transformer)
        ↓
    CNN Decoder (deconv to reconstruct original EEG)

Pre-training objective:
    Re-Masked Token Prediction: randomly mask 15% of time points,
    re-randomize mask each epoch. Reconstruct the original unmasked EEG signal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ── Conformer Encoder ────────────────────────────────────────────
class ConformerEncoder(nn.Module):
    """
    EEG Conformer-style encoder: temporal conv → spatial conv → pooling → transformer.
    
    Input:  (batch, C, T)     e.g. (batch, 105, 5000)
    Output: (batch, S, d_model)  e.g. (batch, 100, 256)
    
    where S = T // pool_stride
    """
    def __init__(self, n_channels=105, d_model=256, n_filters=40,
                 temporal_kernel=25, pool_stride=50,
                 n_heads=8, n_transformer_layers=4, dropout=0.1):
        super().__init__()
        
        self.n_channels = n_channels
        self.d_model = d_model
        self.pool_stride = pool_stride
        
        # 1) Temporal convolution: learn frequency filters
        #    (batch, 1, C, T) → (batch, n_filters, C, T)
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, n_filters, kernel_size=(1, temporal_kernel),
                      padding=(0, temporal_kernel // 2), bias=False),
            nn.BatchNorm2d(n_filters),
        )
        
        # 2) Spatial convolution: project across channels
        #    (batch, n_filters, C, T) → (batch, n_filters, 1, T)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(n_filters, n_filters, kernel_size=(n_channels, 1),
                      groups=1, bias=False),
            nn.BatchNorm2d(n_filters),
            nn.ELU(),
        )
        
        # 3) Average pooling: compress time dimension
        #    (batch, n_filters, 1, T) → (batch, n_filters, 1, T//pool_stride)
        self.pool = nn.AvgPool2d(kernel_size=(1, pool_stride))
        self.dropout = nn.Dropout(dropout)
        
        # 4) Project to d_model
        self.projection = nn.Linear(n_filters, d_model)
        
        # 5) Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout, max_len=1000)
        
        # 6) Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)
    
    def forward(self, x):
        """
        Args:
            x: (batch, C, T) raw EEG signal
        Returns:
            (batch, S, d_model) where S = T // pool_stride
        """
        # (batch, C, T) → (batch, 1, C, T)
        x = x.unsqueeze(1)
        
        # temporal conv: (batch, 1, C, T) → (batch, F, C, T)
        x = self.temporal_conv(x)
        
        # spatial conv: (batch, F, C, T) → (batch, F, 1, T)
        x = self.spatial_conv(x)
        
        # pool: (batch, F, 1, T) → (batch, F, 1, S) where S = T // pool_stride
        x = self.pool(x)
        x = self.dropout(x)
        
        # reshape: (batch, F, 1, S) → (batch, S, F)
        x = x.squeeze(2)          # (batch, F, S)
        x = x.permute(0, 2, 1)    # (batch, S, F)
        
        # project: (batch, S, F) → (batch, S, d_model)
        x = self.projection(x)
        
        # positional encoding + transformer
        x = self.pos_encoding(x)
        x = self.transformer(x)   # (batch, S, d_model)
        
        return x


# ── CNN Decoder (for pre-training reconstruction) ────────────────
class ConformerDecoder(nn.Module):
    """
    CNN Decoder to reconstruct original EEG from Conformer encoder output.
    
    Input:  (batch, S, d_model)  e.g. (batch, 100, 256)
    Output: (batch, C, T)        e.g. (batch, 105, 5000)
    """
    def __init__(self, n_channels=105, d_model=256, n_filters=40,
                 pool_stride=50, target_T=5000):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_filters = n_filters
        self.pool_stride = pool_stride
        self.target_T = target_T
        
        # 1) Project back from d_model to n_filters
        self.projection = nn.Linear(d_model, n_filters)
        
        # 2) Upsample time dimension using ConvTranspose1d
        #    (batch, n_filters, S) → (batch, n_filters, S * pool_stride)
        self.temporal_deconv = nn.Sequential(
            nn.ConvTranspose1d(n_filters, n_filters,
                               kernel_size=pool_stride, stride=pool_stride),
            nn.ELU(),
        )
        
        # 3) Reconstruct channels
        #    (batch, n_filters, T) → (batch, n_channels, T)
        self.channel_reconstruct = nn.Sequential(
            nn.Conv1d(n_filters, n_channels, kernel_size=1),
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, S, d_model)
        Returns:
            (batch, C, T) reconstructed EEG
        """
        # (batch, S, d_model) → (batch, S, n_filters)
        x = self.projection(x)
        
        # (batch, S, n_filters) → (batch, n_filters, S)
        x = x.permute(0, 2, 1)
        
        # upsample: (batch, n_filters, S) → (batch, n_filters, S*pool_stride)
        x = self.temporal_deconv(x)
        
        # truncate or pad to exact target_T
        if x.shape[2] > self.target_T:
            x = x[:, :, :self.target_T]
        elif x.shape[2] < self.target_T:
            pad = self.target_T - x.shape[2]
            x = F.pad(x, (0, pad))
        
        # reconstruct channels: (batch, n_filters, T) → (batch, C, T)
        x = self.channel_reconstruct(x)
        
        return x


# ── Pre-Training Model ───────────────────────────────────────────
class ConformerPreTrainModel(nn.Module):
    """
    Full pre-training model: Conformer Encoder + CNN Decoder.
    
    Pre-training objective: reconstruct masked EEG signals.
    """
    def __init__(self, n_channels=105, d_model=256, n_filters=40,
                 temporal_kernel=25, pool_stride=50,
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
    
    def forward(self, x_masked):
        """
        Args:
            x_masked: (batch, C, T) masked EEG signal
        Returns:
            x_recon: (batch, C, T) reconstructed EEG signal
        """
        encoded = self.encoder(x_masked)   # (batch, S, d_model)
        decoded = self.decoder(encoded)    # (batch, C, T)
        return decoded


# ── Masking Utilities ─────────────────────────────────────────────
def create_remask(x, mask_ratio=0.15):
    """
    Re-Masked Token Prediction: randomly mask time points.
    Called every epoch to create different masks.
    
    Args:
        x: (batch, C, T) original EEG signal
        mask_ratio: fraction of time points to mask
    
    Returns:
        x_masked: (batch, C, T) masked EEG (masked positions set to 0)
        mask: (batch, 1, T) boolean mask, True = masked
    """
    batch, C, T = x.shape
    
    # create mask: True = masked
    num_mask = int(T * mask_ratio)
    mask = torch.zeros(batch, 1, T, dtype=torch.bool, device=x.device)
    
    for i in range(batch):
        # randomly select time points to mask
        indices = torch.randperm(T, device=x.device)[:num_mask]
        mask[i, 0, indices] = True
    
    # apply mask: set masked time points to 0
    x_masked = x.clone()
    x_masked[mask.expand_as(x)] = 0.0
    
    return x_masked, mask


# ── Positional Encoding ──────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
