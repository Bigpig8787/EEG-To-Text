# Version Log — EEG-To-Text Multi-View Conformer

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
