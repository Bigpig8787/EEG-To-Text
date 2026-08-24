# Version Log — EEG-To-Text Multi-View Conformer

## 2026-08-25 — architecture made configurable; ANN run matched to the SNN t=4 baseline

**Context.** An earlier attempt (commit `1661006`, reverted in `837d92a`) scaled
the *SNN* up to the ANN's parameter count and deliberately left sequence length
alone. The direction is now the other way round: the **SNN t=4 baseline is the
anchor and is not touched at all**, and the ANN is scaled *down* to it, with
every hyperparameter aligned — architecture and sequence length together.

**The architecture is no longer hard-coded.** `train_multiview.py` and
`eval_multiview.py` used to construct `MultiViewConformerTranslator` with literal
numbers in two places that had to be kept in sync by hand. All eleven knobs are
now CLI arguments, and **every default reproduces the old literals**, so existing
scripts build the identical model and write the identical `save_name`
(verified: legacy `RegionalConformerEncoder` is 12,682,336 params either way).
`eval_multiview.py` reads the geometry back out of the training config JSON, so
eval can no longer drift from training.

**Alignment to `Spiking-EEG2TEXT/configs/multiview_snn_v2.json`:**

| SNN t=4 knob | value | ANN flag | was |
|---|---|---|---|
| `local_dim[0]` t_conv out | 64 | `--n_filters 64` | 40 |
| `local_dim[1]` s_conv out | 32 | `--n_spatial_filters 32` | tied to n_filters |
| `local_dim[2]` attn dim | 256 | `--d_model 256` | 512 |
| `local_kernel_size` (default) | 250 | `--temporal_kernel 250` | 200 |
| `local_pool_kernel_size` | 100 | `--pool_stride 100` | 50 |
| `local_attn_length` | 32 | `--tokens_per_view 32` | 64 |
| `n_class_per_view` | 4 | `--n_cls_per_view 4` | 8 |
| `local_num_attn_layers` | 2 | `--n_encoder_layers 2` | 4 |
| `global_num_attn_layers` | 2 | `--n_global_layers 2` | 3 |
| `learning_rate_step2` | 5e-6 | `-lr2 0.000005` | 5e-7 |
| `lora.lr_lora` | 1e-4 | `--lr_lora 0.0001` | coupled to LR2×2.0 |
| `lora.lora_r` | 8 | `--lora_r 8` | 16 |

Already identical, no flag needed: batch_size 4, grad_accum_steps 2, 50+70
epochs, lr1 5e-5, warmup_ratio 0.2, weight_decay 0.05, clip_norm 1.0,
label_smooth 0.1, patience 10 / no_early_stop, seed 312, heads 8,
`lora_alpha = lora_r`, lora_dropout 0.15, targets q/k/v/out_proj, FFN 4×,
`global_out_dim` 1024.

**Measured result** (encoder side, BART excluded — it is byte-identical on both
sides, and LoRA r=8 contributes the same 2,359,296 to each):

| | encoder side | BART input sequence |
|---|---|---|
| SNN t=4 (anchor, unchanged) | 19,158,656 | 10×(4+32) = 360 |
| ANN matched | 18,119,808 | 10×(4+32) = 360 |

ANN is 5.4% smaller. The conv stem now matches layer for layer
(temporal 16,128 and spatial 53,312 on both sides). The residual gap is
**structural, not a hyperparameter gap**: per region the SNN carries a learnable
`positional_embedding` (34,816) and an extra `before_transformer_linear`
(66,048) that the ANN has no counterpart for — its `PositionalEncoding` is
sinusoidal and parameter-free, and its projection goes straight to `d_model`.
10 × ~103,680 accounts for essentially the whole difference. No hyperparameter
can close it.

**What changed.**

- `config.py` — eleven architecture arguments (`--d_model`, `--n_filters`,
  `--n_spatial_filters`, `--temporal_kernel`, `--pool_stride`,
  `--tokens_per_view`, `--n_cls_per_view`, `--n_heads`, `--n_encoder_layers`,
  `--n_global_layers`, `--dropout`), plus `--lr_lora` and `--save_suffix`.
- `train_multiview.py` — reads them, passes them to the model, logs the geometry
  and the encoder-side parameter count. `--lr_lora` sets the adapter LR outright
  instead of `LR2 × 2.0`; the SNN specifies it that way, so matching it needs an
  absolute value. `--save_suffix` keeps runs apart because neither `lora_r` nor
  the architecture appears in `save_name`.
- `models/multiview.py` — new `n_spatial_filters` (default `None` = old
  behaviour) so the stem can widen then narrow, 1→64→32, the way the SNN's
  `SpikeEncoder1T1S` does with its 3-tuple `dim`. `load_pretrained_encoder` now
  shape-checks the `temporal_conv` tensors too — the proj/transformer loops
  already did, and without it a geometry change turns a warm start into a
  `load_state_dict` size-mismatch crash. Skipped tensors are now reported.
- `eval_multiview.py` — geometry read from the config JSON, with the old
  literals as fallbacks so pre-existing configs still evaluate correctly.
- `scripts/train_multiview_snn_matched.bat`, `scripts/eval_multiview_snn_matched.bat` — new.

**Verified:** all four touched files compile; both bats' flags parse; the matched
`RegionalConformerEncoder` forwards to `cls (B,4,256)` / `local (B,32,256)` with
and without `eeg_len`; legacy defaults still produce the byte-identical model and
the byte-identical `save_name`.

```cmd
REM the matched run
scripts\train_multiview_snn_matched.bat
scripts\eval_multiview_snn_matched.bat
```

**Warning:** `checkpoints/pretrain/encoder_best.pt` was trained at the old
geometry and will not load into the matched model. Every tensor is skipped with a
warning and STEP 1 starts from random init. Re-run ANN MAE pre-training at this
geometry first if a warm start is wanted — and note the SNN baseline it is being
compared against *does* start from a pre-trained encoder.

**One knob cannot be aligned.** The SNN's local transformer has no dropout at all
(spiking blocks do not use it); its global transformer defaults to 0.1, which is
what the ANN uses everywhere. The ANN is left at `--dropout 0.1` rather than 0 —
removing regularisation from an ANN is a larger behavioural change than the
alignment would buy.

## 2026-07-29 — MAE pre-training rewritten to match the SNN encoder

**Bug: pre-training was learning from an empty sequence.**

`ConformerEncoder` keeps a pooling window only if none of its `pool_stride`
time points is masked (`F.max_pool1d` over the mask). `create_remask` masked
*individual time points* at ratio 0.15, so a window survived with probability
`0.85 ** 100 ≈ 8.7e-8`. In practice every window was zeroed and the encoder
output carried no EEG signal — the transformer was reconstructing from
positional encoding alone.

The SNN side fixed the same defect in `EEGSNN` v1.1.2; the ANN side never got
the port. This is that port.

**What changed — encoder now follows the MAE convention used by
`spiking_models/local_transformer/local_transformer_v2.py`:**

- `models/pretrain_model.py` `create_remask(x, mask_ratio, block_size=100)` —
  masks whole **pool-aligned blocks**, the same number for every sample in the
  batch (uniform visible-token count is required by the encoder's reshape).
  `block_size` must equal `pool_stride`; every caller now passes it.
  At least one window is always left visible.
- `models/conformer.py` `ConformerEncoder.forward` — positional encoding is
  applied **before** masking, then masked tokens are **dropped from the
  sequence** instead of zeroed. Surviving tokens keep their absolute position;
  attention never sees a masked slot. Returns `(visible_tokens, keep)` when a
  mask is given, a plain tensor when it is not (fine-tune path unchanged).
- `models/pretrain_model.py` `ConformerDecoder` — gains a learnable
  `mask_token` and a learnable `latent_pos` embedding at latent resolution.
  Visible tokens are scattered back into their own slots, dropped slots get
  `mask_token`, then the existing deconv reconstructs the full length. Mirrors
  `components/ann_decoder/transformer_decoder.py` on the SNN side.
- `training/pretrain_pipeline.py` — was calling `model(masked)` **without** the
  mask, so the encoder never masked at all. Now passes it.
- `models/pretrain_clip_model.py` — unpacks the encoder's new tuple return.

**Callers updated:** `train_pretrain.py`, `train_pretrain_io_log.py`
(incl. `save_io_sample`), `train_pretrain_clip.py`, `training/pretrain_pipeline.py`.

**New:** `scripts/test_mae_masking.py` — dataset-free self-test. Checks block
alignment, the encoder return contract, PE-before-drop (with a negative control
reproducing the old "drop then PE" order), and `mask_token` gradient flow.

**Breaking:** `ConformerDecoder` gained two parameters, so old *full-model*
pre-train checkpoints no longer load with `strict=True`.
`checkpoints/pretrain/encoder_best.pt` is encoder-only and still loads — but it
was trained under the broken mask, so it should be regarded as untrained and
pre-training re-run.

```cmd
REM re-run pre-training after this change
python train_pretrain.py --pool_stride 100 --mask_ratio 0.15

REM verify the masking path first (no dataset needed)
python scripts/test_mae_masking.py
```

**Still open:** `ConformerEncoder` calls `self.transformer(x)` with no
`src_key_padding_mask`, so raw EEG zero-padded up to 5000 is attended over.
`RegionalConformerEncoder` (the fine-tune path) already handles this correctly.

## 2026-05-22 — tpv64 full BART fine-tune (no LoRA), resume from step-1

**Wired the dead `--no_lora` flag.** Was previously parsed by config.py but never read; `NO_LORA` was hardcoded `False`.

- `train_multiview.py:234` — `NO_LORA = False` → `NO_LORA = args.get('no_lora', False)`
  - `--no_lora` now actually enables STEP 2 full BART fine-tune (unfreeze all BART params, no LoRA adapters). Other bat scripts unaffected (default still `False`).

**New run:** tpv64 geometry, LoRA removed, STEP 2 resumed from the tpv64 LoRA run's step-1 (encoder warm-up) checkpoint.
- `--resume .../multiview_cls8_lora_tpv64/best/..._unique_sent_s1.pt` skips STEP 1 + skips `encoder_best.pt` load.
- save_name suffix `_cont` (resume) + `_full` (no-lora) → `..._unique_sent_cont_full`.
- Output: `..._unique_sent_cont_full_merged.pt` (plain state_dict — full-FT path does no LoRA merge).
- step2 LRs: enc = LR2×0.2 = 1e-7, BART = LR2×0.1 = 5e-8.

**New scripts:**
- `scripts/train_multiview_cls8_full_tpv64_resume.bat`
- `scripts/eval_multiview_cls8_full_tpv64_resume.bat`

**Note:** `--resume` ckpt must be tpv64 geometry. The `_s1.pt` must exist on the lab PC (`.pt` excluded from git) — produced by `train_multiview_cls8_lora_tpv64.bat`.

## 2026-05-19 — tpv90 → tpv64, pool_stride 25 → 50

**Geometry change.** Multiview token geometry:

- `pool_stride`: 25 → **50** (AvgPool 5000 → 100)
- `tokens_per_view`: 90 → **64** (AdaptiveAvgPool 100 → 64)
- BART seq: V*(k+T) = 10*(8+64) = **720 ≤ 1024** (safe)

**Files changed:**
- `train_multiview.py:289` — MultiViewConformerTranslator call args
- `eval_multiview.py:106` — must match training geometry
- `models/multiview.py:27-28` — RegionalConformerEncoder defaults
- `models/multiview.py:99-100` — MultiViewConformerTranslator defaults
- `README.md` — Key Design Parameters table (pretrain geometry untouched)

**New scripts:**
- `scripts/train_multiview_cls8_lora_tpv64.bat`
- `scripts/eval_multiview_cls8_lora_tpv64.bat`

**Breaking:** Cannot resume from tpv90/tpv32 checkpoints — token geometry differs. Train from scratch. New save dir `./checkpoints/multiview_cls8_lora_tpv64` (no collision with tpv90).

Old tpv90 scripts/checkpoints retained for comparison.
