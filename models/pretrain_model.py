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
        self.n_latent = target_T // pool_stride

        # MAE decoder inputs. The encoder drops masked windows entirely, so the
        # decoder has to rebuild the full latent sequence: one learnable token
        # standing in for every dropped window, plus a learnable positional
        # embedding at latent resolution so it knows *where* each slot sits.
        # Mirrors `components/ann_decoder/transformer_decoder.py` on the SNN side.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.latent_pos = nn.Parameter(torch.zeros(1, self.n_latent, d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.latent_pos, std=0.02)

        self.projection = nn.Linear(d_model, n_filters)
        self.temporal_deconv = nn.Sequential(
            nn.ConvTranspose1d(n_filters, n_filters, kernel_size=pool_stride, stride=pool_stride),
            nn.ELU(),
        )
        self.channel_reconstruct = nn.Conv1d(n_filters, n_channels, kernel_size=1)

    def forward(self, x, keep=None):
        """
        Args:
            x: (B, L_visible, d_model) from the encoder when `keep` is given,
                otherwise the full (B, L, d_model) latent sequence.
            keep: (B, L) bool from `ConformerEncoder`, True = slot survived.
        """
        if keep is not None:
            batch, n_latent = keep.shape
            # `expand` is a view; `clone` makes it writable while keeping the
            # gradient path back to `mask_token`.
            full = self.mask_token.expand(batch, n_latent, -1).clone()
            # Row-major order of `full[keep]` matches the order the encoder used
            # for `x[keep]`, so visible tokens land back in their own slots.
            full[keep] = x.reshape(-1, x.size(-1))
            x = full

        if x.size(1) > self.latent_pos.size(1):
            raise ValueError(
                f'latent length {x.size(1)} exceeds decoder positional embedding '
                f'length {self.latent_pos.size(1)}; target_T // pool_stride must '
                f'match the encoder.'
            )
        x = x + self.latent_pos[:, :x.size(1), :]

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
        encoded = self.encoder(x_masked, mask=mask)
        if mask is None:
            return self.decoder(encoded)
        latent, keep = encoded
        return self.decoder(latent, keep=keep)


def create_remask(x, mask_ratio=0.15, block_size=100):
    """Re-Masked Token Prediction with pool-aligned block masking.

    `ConformerEncoder` keeps a pool window only when none of its `pool_stride`
    time points is masked. Under per-time-point random masking at ratio 0.15 and
    stride 100 a window survives with probability 0.85 ** 100 ~= 9e-8, so in
    practice the encoder is handed an empty sequence and pre-training learns
    nothing. Masking whole pool-aligned blocks makes each window either fully
    masked or fully visible.

    Every sample in the batch drops the same number of blocks, which keeps the
    visible token count uniform — the encoder's reshape into (B, L_visible, d)
    depends on that.

    Args:
        x: (B, C, T) raw EEG.
        mask_ratio: fraction of pool windows to mask.
        block_size: must equal the encoder's `pool_stride`.

    Returns:
        `(x_masked, mask)`, mask is (B, 1, T) bool with True = masked.
    """
    batch, C, T = x.shape
    mask = torch.zeros(batch, 1, T, dtype=torch.bool, device=x.device)
    n_blocks = T // block_size

    if n_blocks == 0:
        # Sequence shorter than one pool window. No window can survive anyway, so
        # there is nothing to align to — fall back to per-time-point masking.
        num_mask = max(1, int(T * mask_ratio))
        for i in range(batch):
            indices = torch.randperm(T, device=x.device)[:num_mask]
            mask[i, 0, indices] = True
    else:
        # Cap at n_blocks - 1 so at least one window always reaches the encoder.
        n_mask_blocks = min(max(1, int(round(n_blocks * mask_ratio))), n_blocks - 1)
        for i in range(batch):
            for b in torch.randperm(n_blocks, device=x.device)[:n_mask_blocks].tolist():
                mask[i, 0, b * block_size:(b + 1) * block_size] = True
        # The trailing T % block_size samples stay visible; they fall outside every
        # aligned pool window the encoder produces.

    x_masked = x.clone()
    x_masked[mask.expand_as(x)] = 0.0
    return x_masked, mask
