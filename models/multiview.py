"""
Multi-View Conformer Translator for EEG-to-Text decoding.

Key fix from previous version: 
- freeze/unfreeze only affects gradient computation, does NOT rebuild optimizer
- AdaptiveAvgPool1d ensures fixed output length per view
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from models.conformer import ConformerEncoder, PositionalEncoding
from models.base import BaseEEGModel
from data.channel_mapping import BRAIN_REGION_CHANNEL_COUNT


class RegionalConformerEncoder(nn.Module):
    """Conformer encoder for a single brain region with a learnable [CLS] summary token.

    Returns (cls, local):
      cls   - (B, 1, d_model)   per-view summary for the global transformer
      local - (B, T, d_model)   per-view local tokens passed to the BART decoder
    """
    def __init__(self, n_channels, d_model=512, n_filters=40,
                 temporal_kernel=200, pool_stride=100,
                 tokens_per_view=32,
                 n_heads=8, n_transformer_layers=4, dropout=0.1):
        super().__init__()
        self.tokens_per_view = tokens_per_view

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
        self.dropout1 = nn.Dropout(dropout)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(tokens_per_view)
        self.projection = nn.Linear(n_filters, d_model)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout, max_len=tokens_per_view + 100)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = x.squeeze(2)
        x = self.adaptive_pool(x)
        x = x.permute(0, 2, 1)
        x = self.projection(x)
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        return x[:, :1, :], x[:, 1:, :]


class MultiViewConformerTranslator(BaseEEGModel):
    def __init__(self, pretrained_bart, d_model=512, n_filters=40,
                 temporal_kernel=200, pool_stride=100,
                 tokens_per_view=32,
                 n_heads=8, n_encoder_layers=4,
                 n_global_layers=3, dropout=0.1,
                 decoder_embedding_size=1024):
        super().__init__()
        self.tokens_per_view = tokens_per_view
        self.region_names = list(BRAIN_REGION_CHANNEL_COUNT.keys())

        self.view_encoders = nn.ModuleDict({
            region: RegionalConformerEncoder(
                n_channels=ch_count, d_model=d_model, n_filters=n_filters,
                temporal_kernel=temporal_kernel, pool_stride=pool_stride,
                tokens_per_view=tokens_per_view, n_heads=n_heads,
                n_transformer_layers=n_encoder_layers, dropout=dropout,
            )
            for region, ch_count in BRAIN_REGION_CHANNEL_COUNT.items()
        })

        global_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=dropout, batch_first=True
        )
        self._global_transformer = nn.TransformerEncoder(global_layer, num_layers=n_global_layers)

        # Learnable view-position embedding: gives each of the 10 summary tokens
        # an identity so the global transformer can distinguish regions.
        n_views = len(self.region_names)
        self.view_pos_embed = nn.Parameter(torch.zeros(1, n_views, d_model))
        nn.init.trunc_normal_(self.view_pos_embed, std=0.02)

        self.fc1 = nn.Linear(d_model, decoder_embedding_size)
        self.pretrained = pretrained_bart

    # ── BaseEEGModel interface ──────────────────────────────────────

    @property
    def eeg_encoder(self):
        return self.view_encoders

    @property
    def global_transformer(self):
        return self._global_transformer

    @property
    def language_model(self):
        return self.pretrained

    def get_tuning_targets(self):
        enc = (list(self.view_encoders.parameters()) +
               list(self._global_transformer.parameters()) +
               [self.view_pos_embed] +
               list(self.fc1.parameters()))
        return {'encoder': enc, 'lm': list(self.pretrained.parameters())}

    # ── End BaseEEGModel interface ──────────────────────────────────

    def load_pretrained_encoder(self, pretrained_encoder_path):
        """Load pre-trained weights: temporal conv + projection + transformer."""
        print(f'[INFO] Loading pre-trained encoder: {pretrained_encoder_path}')
        state = torch.load(pretrained_encoder_path, map_location='cpu')

        temporal_w = {k: v for k, v in state.items() if k.startswith('temporal_conv.')}
        proj_w = {k: v for k, v in state.items() if k.startswith('projection.')}
        trans_w = {k: v for k, v in state.items() if k.startswith('transformer.')}

        count = 0
        for region, encoder in self.view_encoders.items():
            es = encoder.state_dict()
            for k, v in temporal_w.items():
                if k in es:
                    es[k] = v; count += 1
            for k, v in proj_w.items():
                if k in es and es[k].shape == v.shape:
                    es[k] = v; count += 1
            for k, v in trans_w.items():
                if k in es and es[k].shape == v.shape:
                    es[k] = v; count += 1
            encoder.load_state_dict(es)

        print(f'  Loaded {count} params across {len(self.view_encoders)} encoders')

    def set_active_views(self, n_active=3):
        """Randomly freeze 7, unfreeze 3 view encoders. Returns active names."""
        active = random.sample(self.region_names, n_active)
        for region in self.region_names:
            requires_grad = (region in active)
            for p in self.view_encoders[region].parameters():
                p.requires_grad = requires_grad
        # global transformer, fc1 always trainable
        for p in self._global_transformer.parameters():
            p.requires_grad = True
        for p in self.fc1.parameters():
            p.requires_grad = True
        return active

    def encode(self, view_inputs):
        # Hierarchical aggregation via per-view [CLS] summary tokens.
        cls_list, local_list = [], []
        for region, encoder in self.view_encoders.items():
            cls, local = encoder(view_inputs[region])
            cls_list.append(cls)
            local_list.append(local)

        # Cross-view integration on the compact 10-token summary sequence.
        cls_seq = torch.cat(cls_list, dim=1)              # (B, V, d_model), V=10
        cls_seq = cls_seq + self.view_pos_embed           # view identity
        global_cls = self._global_transformer(cls_seq)    # (B, V, d_model)

        # Reinject globally-integrated summaries back into each view's local
        # tokens (residual broadcast) so the decoder sees cross-view context
        # without losing per-view temporal detail.
        enriched_locals = []
        for i, local in enumerate(local_list):
            enriched_locals.append(local + global_cls[:, i:i+1, :])
        enriched_locals = torch.cat(enriched_locals, dim=1)   # (B, V*T, d_model)

        # Prepend enriched summaries so BART gets both global + local view.
        combined = torch.cat([global_cls, enriched_locals], dim=1)
        projected = F.relu(self.fc1(combined))
        return projected

    def forward(self, view_inputs, input_masks_batch, input_masks_invert, target_ids_batch):
        encoded = self.encode(view_inputs)
        attn_mask = torch.ones(encoded.shape[0], encoded.shape[1]).to(encoded.device)
        return self.pretrained(inputs_embeds=encoded, attention_mask=attn_mask,
                               return_dict=True, labels=target_ids_batch)

    @torch.no_grad()
    def generate(self, view_inputs, input_masks_batch, input_masks_invert,
                 target_ids_batch, **kwargs):
        encoded = self.encode(view_inputs)
        attn_mask = torch.ones(encoded.shape[0], encoded.shape[1]).to(encoded.device)
        return self.pretrained.generate(inputs_embeds=encoded, attention_mask=attn_mask, **kwargs)
