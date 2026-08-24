# Version Log — EEG-To-Text Multi-View Conformer

## 2026-08-23 — LoRA r=8 run for the parameter-matched ANN-vs-SNN comparison

**Why.** The existing ANN-vs-SNN comparison confounds two variables: the SNN's
encoder is ~7× smaller *and* it feeds BART half as many tokens. The new
experiment holds sequence length fixed and matches parameter count only. The
ANN is the anchor — its architecture does not change — and the SNN was scaled up
to meet it (`Spiking-EEG2TEXT/configs/multiview_snn_match_ann.json`).

**Measured budgets** (BART-large = 406,291,456, identical on both sides; LoRA
r=8 on q/k/v/out_proj covers 144 Linear layers = 2,359,296, identical on both
sides):

| | encoder side | trainable | TOTAL |
|---|---|---|---|
| ANN d=512, 4 enc / 3 global | 136,619,264 | 138,978,560 | 542,910,720 |
| SNN matched d=512, 4/3 | 140,107,904 | 142,467,200 | 546,399,360 |

+2.6% on the SNN side, from its extra `before_transformer_linear`, the U-readout
projection, and the per-LIF LayerNorms.

Within the ANN, `view_encoders` is 23.32% of all parameters and BART 74.84%;
inside one regional encoder the 4-layer transformer is 99.4% and the whole conv
stem under 0.6%. Changing `n_filters` or `temporal_kernel` is therefore useless
for parameter matching — only `d_model` and layer count move the number.

**What changed.**

- `config.py` — new `-suffix` / `--save_suffix` argument (default `''`).
- `train_multiview.py` — appends it to `save_suffix`. `save_name` does not
  include `lora_r`, so an r=8 run would otherwise overwrite the r=16 run's
  `config/decoding/<save_name>.json`. No existing run name changes.
- `scripts/train_multiview_lora_r8.bat` — new. Identical to
  `train_multiview.bat` except `--lora_r 8`, `--save_suffix _r8`, and
  `-s ./checkpoints/multiview_lora_r8`.

**Ordering.** One GPU, so this run must not overlap the SNN pipeline. The SNN
needs a fresh MAE pre-training first (its old encoder checkpoint is the wrong
shape), so the SNN side starts and this run is queued behind it.

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
