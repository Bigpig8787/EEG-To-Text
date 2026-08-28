# 基本方針
- 中文回答
- 調查或除錯時請利用子代理以節省上下文
- 重要的決策定期記錄在Markdown文件中
- 記得如果架構還是使用方法有變都幫我記錄到readme.md裡
- 每一版重大改動都寫入 verion.md裡
- 有改動模型內容後輸出應該的.bat

# CLAUDE.md — EEG-To-Text Multi-View Conformer Project

## Project Overview

Open vocabulary EEG-to-Text decoding. Reads brain EEG signals recorded while subjects read sentences, and decodes the raw EEG into the original text using a Multi-View Conformer encoder + BART decoder.

Based on:
- **EEG2TEXT** (Liu et al., 2024): Multi-view transformer with EEG pre-training
- **EEG Conformer** (Song et al., 2022): CNN + Transformer for EEG
- **Baseline** (Wang & Ji, 2021): BrainTranslator, open vocabulary EEG-to-text

## Environment

- **OS**: Windows 10/11 (lab computer)
- **GPU**: NVIDIA RTX 5080 (16GB VRAM)
- **Conda env**: `EEGBCI.EEGToText` (Python 3.10)
- **Key packages**: `torch`, `transformers`, `peft==0.6.2`, `torchmetrics`, `nltk`, `scipy`, `h5py`
- **Working directory**: `D:\EEG-BCI\EEG-To-Text\`

## Project Structure

```
D:\EEG-BCI\EEG-To-Text\
├── models/
│   ├── __init__.py
│   ├── conformer.py              # Shared Conformer encoder (temporal+spatial conv → pool → transformer)
│   ├── pretrain_model.py          # Pre-training model (encoder + CNN decoder + re-mask)
│   ├── multiview.py               # Multi-View Conformer Translator (10 brain regions → BART)
│   └── brain_translator.py        # Baseline models (Wang & Ji, 2021)
├── data/
│   ├── __init__.py
│   ├── channel_mapping.py         # 105 EEG channels → 10 brain regions mapping
│   ├── dataset.py                 # ZuCo dataset (word-level + raw EEG, RAW_EEG_MAX_LEN=5000)
│   └── pretrain_dataset.py        # Pre-training dataset (raw EEG only)
├── util/
│   ├── construct_dataset_mat_to_pickle_v1.py   # ZuCo v1.0 .mat → .pickle
│   ├── construct_dataset_mat_to_pickle_v2.py   # ZuCo v2.0 .mat → .pickle
│   └── data_loading_helpers_modified.py        # Helper for v2.0 loading
├── scripts/
│   ├── prepare_dataset.bat
│   ├── train_pretrain.bat
│   ├── train_multiview.bat
│   └── eval_multiview.bat
├── dataset/ZuCo/                  # Dataset files (NOT in git)
│   ├── task1-SR/Matlab files/     # Note: folder name has a space
│   ├── task1-SR/pickle/task1-SR-dataset.pickle
│   ├── task2-NR/...
│   ├── task3-TSR/...
│   └── task2-NR-2.0/...
├── checkpoints/                   # Model weights (NOT in git)
│   ├── pretrain/encoder_best.pt   # Pre-trained Conformer encoder (stride=10, d_model=512, val_loss=0.290)
│   └── multiview_lora/best/*.pt   # Multi-view fine-tuned model
├── config.py                      # Argument parser
├── metrics.py                     # BLEU/ROUGE metrics (needs nltk punkt data)
├── train_pretrain.py              # Pre-training script
├── train_multiview.py             # Multi-view training with LoRA + AMP + warmup
├── eval_multiview.py              # Evaluation script
└── README.md
```

## Architecture

### Pipeline
```
1. Pre-training (completed):
   raw EEG (105, 5000) → Conformer Encoder → CNN Decoder → reconstruct masked EEG
   Purpose: learn EEG representations via Re-Masked Token Prediction

2. Multi-view fine-tuning (current focus):
   raw EEG (105, 5000) → split by 10 brain regions
   10 × RegionalConformerEncoder (pre-trained temporal conv + transformer)
     ├─ temporal conv (pre-trained ✓)
     ├─ spatial conv (random init, different channel counts per region)
     ├─ AvgPool(stride=100) → ~50 tokens
     ├─ AdaptiveAvgPool1d(32) → 32 tokens per view
     └─ transformer (pre-trained ✓)
   concat → (1000, 512)
   → Global Transformer (3 layers)
   → FC → (1000, 1024)
   → BART decoder (LoRA fine-tuned, r=16, alpha=32, on q_proj + v_proj)
   → text output
```

### 10 Brain Regions (channel counts)
prefrontal(26), premotor(16), brocas(4), auditory_assoc(9), primary_motor(9), primary_sensory(11), somatic_sensory(9), auditory(4), wernickes(6), visual(11) = 105 total

### Key Design Parameters
- `RAW_EEG_MAX_LEN = 5000` (covers 90th percentile of sentence lengths)
- `temporal_kernel = 200`, `pool_stride = 100` (5000 → 50 tokens, paper-spec)
- `tokens_per_view = 32` (50 → 32 via AdaptiveAvgPool1d)
- `mask_ratio = 0.15` (paper-spec)
- `d_model = 512`
- 10 views × 32 tokens = 320 tokens (within BART max 1024)

## Current Status

### Completed
- [x] ZuCo data preprocessing (.mat → .pickle with rawData extraction)
- [x] Channel-to-region mapping verified (105 channels, no overlap)
- [x] Pre-training with Conformer (stride=10, d_model=512, best val_loss=0.290)
- [x] Multi-view model with LoRA + AMP + warmup + gradient accumulation
- [x] Evaluation pipeline (BLEU/ROUGE)

### Current Problem: Overfitting
Every training run shows train loss dropping but dev loss exploding. Summary of attempts:

| Run | Config | Best dev | Issue |
|-----|--------|----------|-------|
| 1 | --one_step, SGD, lr=5e-7, freeze 3/7 | 5.31 | Skipped Step 1, lr too small |
| 2 | --two_step, AdamW, lr=5e-5, freeze 3/7 | 4.45 | dev exploded after epoch 0 |
| 3 | --two_step, AdamW, lr=5e-6, freeze 3/7 | 4.47 | dev exploded after epoch 1 |
| 4 | LoRA, AMP, warmup, lr=5e-5, all 10 views | 4.02 | dev only good during warmup (lr=1.67e-5), exploded when lr hit 5e-5 |
| 5 | Same as 4 but lr=5e-6 | pending | Not yet run |

**Key observation**: Best dev loss always occurs when lr is smallest (warmup phase). Once lr ramps up, dev explodes. This suggests the optimal lr is around 1e-5 to 5e-6.

### Likely Root Causes
1. **Data/param ratio**: 10,710 train samples vs 138M trainable params (encoder+LoRA+global)
2. **Encoder lr too high**: pre-trained encoders are being destroyed by aggressive updates
3. **AdaptiveAvgPool compression**: 500→100 tokens loses information from pre-training
4. **EEG signal noise**: no artifact rejection or quality filtering on raw EEG

### Next Steps to Try
1. Lower lr to 5e-6 with more epochs (50)
2. Freeze pre-trained encoder transformer layers, only train spatial conv + global transformer + LoRA
3. Reduce encoder trainable params (freeze temporal conv since it's pre-trained)
4. Add dropout/weight decay tuning
5. Try label smoothing on BART loss
6. Data augmentation on raw EEG (time shift, amplitude scaling)

## Training Commands

### Pre-training (**needs re-running** — see 2026-07-29 in version.md)
```cmd
REM verify the MAE masking path first (random tensors, no dataset, no GPU)
python scripts\test_mae_masking.py

scripts\train_pretrain.bat
```
Output: `checkpoints/pretrain/encoder_best.pt`

The existing `encoder_best.pt` (val_loss 0.290) was trained under a broken mask
— per-time-point masking collapsed every pooling window, so the encoder saw no
EEG. Treat it as untrained.

### Multi-view fine-tuning
```cmd
scripts\train_multiview.bat
```
Uses: LoRA (peft), AMP (torch.cuda.amp), warmup+cosine scheduler, gradient accumulation (2 steps), early stopping (patience=7), all 10 view encoders trained together.

### Evaluation
```cmd
scripts\eval_multiview.bat
```
Runs 4 combinations: teacher_forcing × noise. Main result = free generation + real EEG.
**Note**: Requires `nltk` punkt data (`python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"`)

## Important Files to Understand

### train_multiview.py
The main training script. Key features:
- LoRA applied to BART q_proj and v_proj (only 2.3M BART params trainable)
- 3 param groups with different lr: encoder_lr > other_lr (0.5x) > lora_lr (0.1x).
  `--lr_lora` overrides the LoRA group with an absolute LR instead of a multiple of LR2.
- Mixed precision (AMP) with GradScaler
- Warmup 10% + cosine decay scheduler
- Gradient accumulation (batch=2, accum=2, effective=4)
- Early stopping with patience=7
- **The model geometry is no longer hard-coded here.** `--d_model`, `--n_filters`,
  `--n_spatial_filters`, `--temporal_kernel`, `--pool_stride`, `--tokens_per_view`,
  `--n_cls_per_view`, `--n_heads`, `--n_encoder_layers`, `--n_global_layers` and
  `--dropout` all come from `config.py`, with defaults equal to the old literals.
  `eval_multiview.py` reads them back from `config/decoding/<save_name>.json`, so
  eval geometry always follows training. `--save_suffix` keeps runs apart when they
  differ only in `lora_r` or the architecture, neither of which is in `save_name`.

### models/multiview.py
The model. Key methods:
- `load_pretrained_encoder()`: loads temporal conv + projection + transformer from pre-trained weights. Spatial conv stays random (different channel counts per region). Every tensor is shape-checked; a checkpoint from a different geometry is skipped with a warning rather than crashing `load_state_dict`.
- `RegionalConformerEncoder(n_spatial_filters=...)`: lets the conv stem widen then narrow (1 → `n_filters` → `n_spatial_filters`). `None` keeps the old behaviour where both convs share `n_filters`.
- `set_active_views(n)`: freeze/unfreeze strategy (currently unused, all 10 train together)
- `encode()`: 10 regional encoders → concat → global transformer → FC
- `forward()`: encode → BART with cross-entropy loss
- `generate()`: encode → BART beam search

### data/dataset.py
ZuCo dataset loader. Returns both word-level features (for baseline) AND raw_eeg_views (for multi-view). The `get_input_sample()` function discards samples where any word has missing frequency band data, even if raw EEG is fine — this loses some data unnecessarily for the multi-view model.

### data/channel_mapping.py
Maps 105 EEG channels to 10 brain regions. Each channel index (0-104) maps to the original EGI electrode number minus the 23 removed outer-ring electrodes.

## MAE Pre-training (rewritten 2026-07-29)

Now matches the SNN `LocalTransformerV2` / `TransformerDecoder` convention:

1. `create_remask(x, mask_ratio, block_size=pool_stride)` masks whole
   **pool-aligned blocks**, the same count for every sample in the batch.
2. `ConformerEncoder` adds positional encoding **first**, then **drops** the
   masked windows. Survivors keep their absolute position; attention only sees
   real signal. Returns `(visible_tokens, keep)` when masked.
3. `ConformerDecoder` scatters the visible tokens back into their slots, fills
   the dropped ones with a learnable `mask_token`, adds a learnable latent
   positional embedding, then deconvolves back to full length.

The previous per-time-point masking left a pooling window with survival
probability `0.85 ** 100 ≈ 9e-8`, so pre-training reconstructed from positional
encoding alone.

## Pre-training Results
| Config | val_loss | Training time |
|--------|----------|---------------|
| stride=50, d_model=256 | 0.525 | 268 min |
| stride=10, d_model=512 | **0.290** | 278 min |

**Both numbers predate the 2026-07-29 masking fix and are not comparable to
anything trained after it.** Pre-trained encoder weights:
`checkpoints/pretrain/encoder_best.pt` (copied from
`pretrain_s10_d512/conformer_encoder_best.pt`).

## Known Issues
1. `peft` must be version 0.6.2 (newer versions incompatible with installed transformers)
2. `nltk` punkt data must be downloaded before eval
3. `data.py` discards samples based on word-level features even when only raw EEG is needed
4. `.gitignore` should exclude `*.pt`, `*.pickle`, `*.mat` (large binary files)
5. `util/construct_dataset_mat_to_pickle_v2.py` needs `data_loading_helpers_modified.py` in the same directory
6. `ConformerEncoder` calls `self.transformer(x)` with no `src_key_padding_mask`,
   so raw EEG zero-padded up to 5000 is still attended over. The fine-tune path
   (`RegionalConformerEncoder` in `models/multiview.py`) already builds and
   passes one; the pre-train encoder does not.
7. Old *full-model* pre-train checkpoints no longer load with `strict=True` —
   `ConformerDecoder` gained `mask_token` and `latent_pos` on 2026-07-29.
   Encoder-only checkpoints are unaffected.
8. The `-n True` noise control replaces every view with `torch.rand_like`, i.e.
   uniform [0,1). Real EEG is roughly zero-mean with both signs, so this is an
   out-of-distribution input rather than an information-free one — a collapse
   under it is weaker evidence that the model uses EEG than a distribution-
   preserving control (EEG shuffled across sentences) would be. The SNN side
   uses the same `rand_like`, so ANN-vs-SNN comparison stays fair;
   `EEGSNN/compare/compare_ann_snn_noise.py` has the stricter partial-replace
   control but it is not wired into the main eval.
9. Teacher-forcing metrics are computed from `argmax` over gold-conditioned
   logits, and pad positions are decoded into the prediction string as well.
   They are an upper bound on the LM, not evidence of EEG→text decoding. Report
   free generation + real EEG only.

## References
- Wang, Z. and Ji, H. (2021). Open vocabulary EEG-to-text decoding and zero-shot sentiment classification.
- Liu, H. et al. (2024). EEG2TEXT: Open vocabulary EEG-to-text decoding with EEG pre-training and multi-view transformer.
- Song, Y. et al. (2022). EEG Conformer: Convolutional transformer for EEG decoding and visualization.
- Hollenstein, N. et al. (2018). ZuCo, a simultaneous EEG and eye-tracking resource for natural sentence reading.