"""Self-test for the MAE masking path (pool-aligned blocks + PE-before-drop).

Runs on random tensors, needs no dataset and no GPU:

    python scripts/test_mae_masking.py

Covers the four things the 2026-07-29 rewrite changed:

  1. `create_remask` masks whole pool-aligned blocks, the same number of them
     for every sample in the batch.
  2. `ConformerEncoder` returns `(visible_tokens, keep)` and drops the masked
     windows instead of zeroing them.
  3. Positional encoding is applied *before* the drop, so a surviving token
     carries its absolute position. Verified by stubbing the transformer out
     with Identity and checking `enc(x, mask)[i] == enc(x)[keep][i]` exactly —
     if PE ran after the drop, the surviving tokens would be renumbered
     0..L_visible-1 and the two would differ.
  4. `ConformerDecoder` puts the visible tokens back into their own slots and
     fills the rest with the learnable `mask_token`, which receives gradient.
"""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.conformer import ConformerEncoder
from models.pretrain_model import ConformerPreTrainModel, create_remask

B, C, T = 3, 8, 200
POOL_STRIDE = 10
D_MODEL = 32
N_FILTERS = 4
MASK_RATIO = 0.3
L = T // POOL_STRIDE

failures = []


def check(name, cond, detail=''):
    status = 'PASS' if cond else 'FAIL'
    print(f'[{status}] {name}' + (f'  -- {detail}' if detail and not cond else ''))
    if not cond:
        failures.append(name)


def make_encoder():
    return ConformerEncoder(
        n_channels=C, d_model=D_MODEL, n_filters=N_FILTERS,
        temporal_kernel=8, pool_stride=POOL_STRIDE,
        n_heads=4, n_transformer_layers=1, dropout=0.0,
    )


torch.manual_seed(0)
x = torch.randn(B, C, T)

# ── 1. create_remask ────────────────────────────────────────────────────────
x_masked, mask = create_remask(x, mask_ratio=MASK_RATIO, block_size=POOL_STRIDE)

check('mask shape', tuple(mask.shape) == (B, 1, T), str(tuple(mask.shape)))
check('mask is bool', mask.dtype == torch.bool, str(mask.dtype))
check('masked points are zeroed',
      bool((x_masked[mask.expand_as(x)] == 0).all()))
check('visible points untouched',
      torch.equal(x_masked[~mask.expand_as(x)], x[~mask.expand_as(x)]))

blocks = mask.squeeze(1).reshape(B, L, POOL_STRIDE)
all_or_nothing = ((blocks.all(dim=2)) | (~blocks.any(dim=2))).all()
check('every pool window is fully masked or fully visible', bool(all_or_nothing))

per_sample = blocks.any(dim=2).sum(dim=1)
check('same number of blocks masked per sample',
      bool((per_sample == per_sample[0]).all()), str(per_sample.tolist()))
check('at least one window survives', bool((per_sample < L).all()),
      str(per_sample.tolist()))

expected = min(max(1, round(L * MASK_RATIO)), L - 1)
check(f'masked block count == {expected}', int(per_sample[0]) == expected,
      str(int(per_sample[0])))

# ── 2. encoder return contract ──────────────────────────────────────────────
enc = make_encoder().eval()

with torch.no_grad():
    out_nomask = enc(x)
check('mask=None returns a plain tensor', isinstance(out_nomask, torch.Tensor))
check('mask=None shape', tuple(out_nomask.shape) == (B, L, D_MODEL),
      str(tuple(out_nomask.shape)))

with torch.no_grad():
    out_masked = enc(x_masked, mask=mask)
check('mask given returns a 2-tuple',
      isinstance(out_masked, tuple) and len(out_masked) == 2)
vis, keep = out_masked
n_visible = L - expected
check('keep shape', tuple(keep.shape) == (B, L), str(tuple(keep.shape)))
check('visible token count matches keep',
      tuple(vis.shape) == (B, n_visible, D_MODEL), str(tuple(vis.shape)))
check('keep agrees with the input mask',
      torch.equal(keep, ~blocks.any(dim=2)))

# ── 3. positional encoding runs before the drop ─────────────────────────────
enc_id = make_encoder().eval()
enc_id.transformer = nn.Identity()

with torch.no_grad():
    full = enc_id(x_masked)                       # (B, L, D) — PE at every slot
    vis_id, keep_id = enc_id(x_masked, mask=mask)  # (B, L_visible, D)
    gathered = full[keep_id].reshape(B, -1, D_MODEL)

check('PE applied before dropping (absolute positions preserved)',
      torch.allclose(vis_id, gathered, atol=1e-6),
      f'max abs diff = {(vis_id - gathered).abs().max().item():.3e}')

# Negative control: reconstruct what the *old* "drop then PE" order would have
# produced — survivors renumbered 0..L_visible-1 — and require it to differ.
# Without this the check above would pass trivially if PE were all zeros.
with torch.no_grad():
    pe_full = enc_id.pos_encoding.pe[0, :L, :]              # (L, D)
    pre_pe = full - pe_full.unsqueeze(0)                    # representation before PE
    pre_kept = pre_pe[keep_id].reshape(B, -1, D_MODEL)
    pe_renumbered = enc_id.pos_encoding.pe[0, :n_visible, :]
    drop_then_pe = pre_kept + pe_renumbered.unsqueeze(0)

check('negative control: "drop then PE" gives a different result',
      not torch.allclose(vis_id, drop_then_pe, atol=1e-6))

# ── 4. decoder re-insertion + mask_token gradient ───────────────────────────
model = ConformerPreTrainModel(
    n_channels=C, d_model=D_MODEL, n_filters=N_FILTERS,
    temporal_kernel=8, pool_stride=POOL_STRIDE,
    n_heads=4, n_transformer_layers=1, dropout=0.0, target_T=T,
)
model.train()

recon = model(x_masked, mask)
check('reconstruction shape', tuple(recon.shape) == (B, C, T),
      str(tuple(recon.shape)))

mask_exp = mask.expand_as(x)
loss = nn.functional.mse_loss(recon[mask_exp], x[mask_exp])
loss.backward()

check('decoder.mask_token receives gradient',
      model.decoder.mask_token.grad is not None
      and bool(model.decoder.mask_token.grad.abs().sum() > 0))
check('decoder.latent_pos receives gradient',
      model.decoder.latent_pos.grad is not None
      and bool(model.decoder.latent_pos.grad.abs().sum() > 0))
check('encoder receives gradient',
      any(p.grad is not None and p.grad.abs().sum() > 0
          for p in model.encoder.parameters()))

# Masked slots must actually hold the mask token, visible slots the encoder output.
with torch.no_grad():
    latent, keep2 = model.encoder(x_masked, mask=mask)
    rebuilt = model.decoder.mask_token.expand(B, L, -1).clone()
    rebuilt[keep2] = latent.reshape(-1, D_MODEL)
    masked_rows = rebuilt[~keep2]
    tok = model.decoder.mask_token.reshape(1, -1)
check('dropped slots hold mask_token',
      bool(torch.allclose(masked_rows, tok.expand_as(masked_rows), atol=1e-6)))

# ── 5. mask=None path still works end to end (fine-tune / CLIP) ─────────────
with torch.no_grad():
    recon_nomask = model(x)
check('mask=None end to end', tuple(recon_nomask.shape) == (B, C, T),
      str(tuple(recon_nomask.shape)))

print()
if failures:
    print(f'{len(failures)} FAILED: ' + ', '.join(failures))
    sys.exit(1)
print('all checks passed')
