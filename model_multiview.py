"""
Multi-View Conformer Translator for EEG-to-Text decoding.

Uses pre-trained Conformer encoder weights to initialize 10 brain-region encoders.
Each region encoder uses AdaptiveAvgPool1d to output fixed-length sequences.
A global transformer fuses all regions before feeding into BART for text generation.

Architecture:
    10 × Regional Conformer Encoder (pre-trained temporal conv + random spatial conv)
        ↓ AdaptiveAvgPool1d(tokens_per_view)
    concat along sequence dim → (batch, 10*tokens_per_view, d_model)
        ↓ Global Transformer
        ↓ FC → (batch, 10*tokens_per_view, 1024)
        ↓ BART decoder
        → text output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from channel_mapping import BRAIN_REGION_CHANNELS, BRAIN_REGION_CHANNEL_COUNT


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
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ── Regional Conformer Encoder (one brain region) ────────────────
class RegionalConformerEncoder(nn.Module):
    """
    Conformer encoder for a single brain region.
    
    Input:  (batch, C_region, T)
    Output: (batch, tokens_per_view, d_model)
    """
    def __init__(self, n_channels, d_model=512, n_filters=40,
                 temporal_kernel=25, pool_stride=10,
                 tokens_per_view=100,
                 n_heads=8, n_transformer_layers=4, dropout=0.1):
        super().__init__()
        
        self.n_channels = n_channels
        self.tokens_per_view = tokens_per_view
        
        # 1) Temporal conv (channel-independent, can load pre-trained weights)
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, n_filters, kernel_size=(1, temporal_kernel),
                      padding=(0, temporal_kernel // 2), bias=False),
            nn.BatchNorm2d(n_filters),
        )
        
        # 2) Spatial conv (channel-dependent, random init for each region)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(n_filters, n_filters, kernel_size=(n_channels, 1),
                      groups=1, bias=False),
            nn.BatchNorm2d(n_filters),
            nn.ELU(),
        )
        
        # 3) Pooling
        self.pool = nn.AvgPool2d(kernel_size=(1, pool_stride))
        self.dropout1 = nn.Dropout(dropout)
        
        # 4) Adaptive pool to fixed length
        self.adaptive_pool = nn.AdaptiveAvgPool1d(tokens_per_view)
        
        # 5) Project to d_model
        self.projection = nn.Linear(n_filters, d_model)
        
        # 6) Positional encoding + Transformer
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout, max_len=tokens_per_view + 100)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)
    
    def forward(self, x):
        """
        Args:
            x: (batch, C_region, T)
        Returns:
            (batch, tokens_per_view, d_model)
        """
        # (batch, C, T) → (batch, 1, C, T)
        x = x.unsqueeze(1)
        
        # temporal conv: (batch, 1, C, T) → (batch, F, C, T)
        x = self.temporal_conv(x)
        
        # spatial conv: (batch, F, C, T) → (batch, F, 1, T)
        x = self.spatial_conv(x)
        
        # pool: (batch, F, 1, T) → (batch, F, 1, T//stride)
        x = self.pool(x)
        x = self.dropout1(x)
        
        # reshape: (batch, F, 1, T') → (batch, F, T')
        x = x.squeeze(2)
        
        # adaptive pool: (batch, F, T') → (batch, F, tokens_per_view)
        x = self.adaptive_pool(x)
        
        # (batch, F, tokens_per_view) → (batch, tokens_per_view, F)
        x = x.permute(0, 2, 1)
        
        # project: (batch, tokens_per_view, F) → (batch, tokens_per_view, d_model)
        x = self.projection(x)
        
        # transformer
        x = self.pos_encoding(x)
        x = self.transformer(x)
        
        return x


# ── Multi-View Conformer Translator ──────────────────────────────
class MultiViewConformerTranslator(nn.Module):
    """
    Multi-View EEG-to-Text model.
    
    10 regional Conformer encoders → concat → global transformer → BART
    """
    def __init__(self, pretrained_bart, d_model=512, n_filters=40,
                 temporal_kernel=25, pool_stride=10,
                 tokens_per_view=100,
                 n_heads=8, n_encoder_layers=4,
                 n_global_layers=3, dropout=0.1,
                 decoder_embedding_size=1024):
        super().__init__()
        
        self.tokens_per_view = tokens_per_view
        
        # 10 regional encoders
        self.view_encoders = nn.ModuleDict({
            region: RegionalConformerEncoder(
                n_channels=ch_count,
                d_model=d_model,
                n_filters=n_filters,
                temporal_kernel=temporal_kernel,
                pool_stride=pool_stride,
                tokens_per_view=tokens_per_view,
                n_heads=n_heads,
                n_transformer_layers=n_encoder_layers,
                dropout=dropout,
            )
            for region, ch_count in BRAIN_REGION_CHANNEL_COUNT.items()
        })
        
        # Global transformer
        global_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.global_transformer = nn.TransformerEncoder(global_layer, num_layers=n_global_layers)
        
        # Project to BART embedding size
        self.fc1 = nn.Linear(d_model, decoder_embedding_size)
        
        # BART
        self.pretrained = pretrained_bart
    
    def load_pretrained_encoder(self, pretrained_encoder_path):
        """
        Load pre-trained Conformer encoder weights.
        Only copies temporal_conv weights (channel-independent).
        Spatial conv stays randomly initialized (different channel counts per region).
        """
        print(f'[INFO] Loading pre-trained encoder from: {pretrained_encoder_path}')
        pretrained_state = torch.load(pretrained_encoder_path, map_location='cpu')
        
        # extract temporal conv weights from pre-trained encoder
        temporal_conv_weights = {}
        projection_weights = {}
        transformer_weights = {}
        
        for key, value in pretrained_state.items():
            if key.startswith('temporal_conv.'):
                temporal_conv_weights[key] = value
            elif key.startswith('projection.'):
                projection_weights[key] = value
            elif key.startswith('transformer.'):
                transformer_weights[key] = value
        
        print(f'  Found {len(temporal_conv_weights)} temporal_conv parameters')
        print(f'  Found {len(projection_weights)} projection parameters')
        print(f'  Found {len(transformer_weights)} transformer parameters')
        
        # copy to each regional encoder
        loaded_count = 0
        for region, encoder in self.view_encoders.items():
            encoder_state = encoder.state_dict()
            
            # copy temporal conv (same for all regions)
            for key, value in temporal_conv_weights.items():
                if key in encoder_state:
                    encoder_state[key] = value
                    loaded_count += 1
            
            # copy projection (same dimensions)
            for key, value in projection_weights.items():
                if key in encoder_state:
                    if encoder_state[key].shape == value.shape:
                        encoder_state[key] = value
                        loaded_count += 1
            
            # copy transformer (same dimensions)
            for key, value in transformer_weights.items():
                if key in encoder_state:
                    if encoder_state[key].shape == value.shape:
                        encoder_state[key] = value
                        loaded_count += 1
            
            encoder.load_state_dict(encoder_state)
        
        print(f'  Loaded {loaded_count} parameters across {len(self.view_encoders)} regional encoders')
        print(f'  Spatial conv: randomly initialized (different channel counts per region)')
    
    def encode(self, view_inputs):
        """
        Encode raw EEG views into a fused representation.
        
        Args:
            view_inputs: dict {region: (batch, C_region, T)}
        Returns:
            (batch, 10*tokens_per_view, decoder_embedding_size)
        """
        view_outputs = []
        for region, encoder in self.view_encoders.items():
            out = encoder(view_inputs[region])  # (batch, tokens_per_view, d_model)
            view_outputs.append(out)
        
        # concat: (batch, 10*tokens_per_view, d_model)
        combined = torch.cat(view_outputs, dim=1)
        
        # global transformer
        global_out = self.global_transformer(combined)
        
        # project to BART size
        projected = F.relu(self.fc1(global_out))  # (batch, 10*tokens_per_view, 1024)
        
        return projected
    
    def forward(self, view_inputs, input_masks_batch, input_masks_invert, target_ids_batch_converted):
        encoded = self.encode(view_inputs)
        
        # attention mask for BART
        attn_mask = torch.ones(encoded.shape[0], encoded.shape[1]).to(encoded.device)
        
        out = self.pretrained(
            inputs_embeds=encoded,
            attention_mask=attn_mask,
            return_dict=True,
            labels=target_ids_batch_converted
        )
        return out
    
    @torch.no_grad()
    def generate(self, view_inputs, input_masks_batch, input_masks_invert,
                 target_ids_batch_converted, **kwargs):
        encoded = self.encode(view_inputs)
        attn_mask = torch.ones(encoded.shape[0], encoded.shape[1]).to(encoded.device)
        return self.pretrained.generate(
            inputs_embeds=encoded,
            attention_mask=attn_mask,
            **kwargs
        )
