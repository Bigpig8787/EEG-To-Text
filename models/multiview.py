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
                 temporal_kernel=200, pool_stride=25,
                 tokens_per_view=128, n_cls_tokens=4,
                 n_heads=8, n_transformer_layers=4, dropout=0.1):
        super().__init__()
        self.tokens_per_view = tokens_per_view
        self.n_cls_tokens = n_cls_tokens

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
        self.dropout1 = nn.Dropout(dropout)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(tokens_per_view)
        self.projection = nn.Linear(n_filters, d_model)

        self.cls_token = nn.Parameter(torch.zeros(1, n_cls_tokens, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout, max_len=tokens_per_view + n_cls_tokens + 100)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_transformer_layers)

    def forward(self, x, eeg_len=None, raw_max_len=5000):
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
        k = self.n_cls_tokens
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # Build padding mask based on actual EEG length. adaptive_pool maps the
        # full raw_max_len window to tokens_per_view bins, so bin i is valid
        # iff its left edge i * raw_max_len / T < eeg_len.
        if eeg_len is not None:
            T = self.tokens_per_view
            valid_n = torch.ceil(eeg_len.float() * T / raw_max_len).long()
            valid_n = valid_n.clamp(min=1, max=T)
            ar = torch.arange(T, device=x.device).unsqueeze(0)
            local_valid = ar < valid_n.unsqueeze(1)                  # (B, T)
            cls_valid = torch.ones(B, k, dtype=torch.bool, device=x.device)
            full_valid = torch.cat([cls_valid, local_valid], dim=1)  # (B, k+T)
            key_padding_mask = ~full_valid                           # True = ignore
        else:
            local_valid = torch.ones(B, self.tokens_per_view,
                                     dtype=torch.bool, device=x.device)
            key_padding_mask = None
        x = self.pos_encoding(x)
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        return x[:, :k, :], x[:, k:, :], local_valid


class MultiViewConformerTranslator(BaseEEGModel):
    def __init__(self, pretrained_bart, d_model=512, n_filters=40,
                 temporal_kernel=200, pool_stride=25,
                 tokens_per_view=128, n_cls_per_view=4,
                 n_heads=8, n_encoder_layers=4,
                 n_global_layers=3, dropout=0.1,
                 decoder_embedding_size=1024):
        super().__init__()
        self.tokens_per_view = tokens_per_view
        self.n_cls_per_view = n_cls_per_view
        self.region_names = list(BRAIN_REGION_CHANNEL_COUNT.keys())

        self.view_encoders = nn.ModuleDict({
            region: RegionalConformerEncoder(
                n_channels=ch_count, d_model=d_model, n_filters=n_filters,
                temporal_kernel=temporal_kernel, pool_stride=pool_stride,
                tokens_per_view=tokens_per_view, n_cls_tokens=n_cls_per_view,
                n_heads=n_heads,
                n_transformer_layers=n_encoder_layers, dropout=dropout,
            )
            for region, ch_count in BRAIN_REGION_CHANNEL_COUNT.items()
        })

        global_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=dropout, batch_first=True
        )
        self._global_transformer = nn.TransformerEncoder(global_layer, num_layers=n_global_layers)

        # Learnable view-position embedding: gives each of the V*k summary tokens
        # an identity so the global transformer can distinguish regions and slots.
        n_views = len(self.region_names)
        self.view_pos_embed = nn.Parameter(torch.zeros(1, n_views * n_cls_per_view, d_model))
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

    def encode(self, view_inputs, raw_eeg_lens=None):
        # Hierarchical aggregation via per-view [CLS] summary tokens.
        # Each region emits k CLS tokens for richer cross-view bandwidth.
        cls_list, local_list, valid_list = [], [], []
        for region, encoder in self.view_encoders.items():
            cls, local, local_valid = encoder(view_inputs[region], eeg_len=raw_eeg_lens)
            cls_list.append(cls)         # each (B, k, d_model)
            local_list.append(local)
            valid_list.append(local_valid)

        # Cross-view integration on V*k summary tokens.
        k = self.n_cls_per_view
        cls_seq = torch.cat(cls_list, dim=1)              # (B, V*k, d_model)
        cls_seq = cls_seq + self.view_pos_embed           # region+slot identity
        global_cls = self._global_transformer(cls_seq)    # (B, V*k, d_model)

        # Reinject globally-integrated summaries back into each view's local
        # tokens (residual broadcast of region-mean over k CLSs) so the decoder
        # sees cross-view context without losing per-view temporal detail.
        enriched_locals = []
        for i, local in enumerate(local_list):
            region_cls = global_cls[:, i*k:(i+1)*k, :].mean(dim=1, keepdim=True)
            enriched_locals.append(local + region_cls)
        enriched_locals = torch.cat(enriched_locals, dim=1)   # (B, V*T, d_model)

        # Prepend enriched summaries so BART gets both global + local view.
        combined = torch.cat([global_cls, enriched_locals], dim=1)
        projected = F.relu(self.fc1(combined))

        # Build BART attention mask: 1 = attend, 0 = ignore.
        # V*k global CLS tokens are always valid; V*tokens_per_view local
        # tokens follow per-view validity from each view encoder.
        B = projected.size(0)
        V = len(self.region_names)
        cls_mask = torch.ones(B, V * k, dtype=torch.long, device=projected.device)
        if raw_eeg_lens is not None:
            local_mask = torch.cat(valid_list, dim=1).long()       # (B, V*T)
        else:
            local_mask = torch.ones(B, V * self.tokens_per_view,
                                    dtype=torch.long, device=projected.device)
        attn_mask = torch.cat([cls_mask, local_mask], dim=1)       # (B, V*k + V*T)
        return projected, attn_mask

    def forward(self, view_inputs, input_masks_batch, input_masks_invert,
                target_ids_batch, raw_eeg_lens=None):
        encoded, attn_mask = self.encode(view_inputs, raw_eeg_lens=raw_eeg_lens)
        return self.pretrained(inputs_embeds=encoded, attention_mask=attn_mask,
                               return_dict=True, labels=target_ids_batch)

    @torch.no_grad()
    def generate(self, view_inputs, input_masks_batch, input_masks_invert,
                 target_ids_batch, raw_eeg_lens=None, **kwargs):
        encoded, attn_mask = self.encode(view_inputs, raw_eeg_lens=raw_eeg_lens)
        return self.pretrained.generate(inputs_embeds=encoded, attention_mask=attn_mask, **kwargs)
