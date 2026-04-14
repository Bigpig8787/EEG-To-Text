# EEG-To-Text: Multi-View Conformer with EEG Pre-Training

Open vocabulary EEG-to-Text decoding using Conformer encoder, EEG pre-training, and multi-view transformer.

Based on [EEG2TEXT (Liu et al., 2024)](https://arxiv.org/abs/2405.02165) and [EEG Conformer (Song et al., 2022)](https://ieeexplore.ieee.org/document/9991178).

## Project Structure

```
├── models/
│   ├── conformer.py          # Shared Conformer encoder (temporal+spatial conv → pool → transformer)
│   ├── pretrain_model.py     # Pre-training model (encoder + CNN decoder + re-mask)
│   ├── multiview.py          # Multi-View Conformer Translator (10 brain regions → BART)
│   └── brain_translator.py   # Baseline models (Wang & Ji, 2021)
├── data/
│   ├── channel_mapping.py    # 105 channels → 10 brain regions mapping (EGI HydroCel 128)
│   ├── dataset.py            # ZuCo dataset (word-level + raw EEG)
│   └── pretrain_dataset.py   # Pre-training dataset (raw EEG only)
├── util/
│   ├── construct_dataset_mat_to_pickle_v1.py   # ZuCo v1.0 preprocessing
│   └── construct_dataset_mat_to_pickle_v2.py   # ZuCo v2.0 preprocessing
├── scripts/
│   ├── prepare_dataset.bat   # Run all preprocessing
│   ├── train_pretrain.bat    # Pre-training
│   ├── train_multiview.bat   # Multi-view fine-tuning
│   └── eval_multiview.bat    # Evaluation
├── train_pretrain.py         # Pre-training script
├── train_multiview.py        # Multi-view training script
├── eval_multiview.py         # Evaluation script
├── config.py                 # Argument parser
├── metrics.py                # BLEU/ROUGE metrics (from baseline)
└── README.md
```

## Setup

### 1. Download ZuCo Dataset

Download from [OSF](https://osf.io/q3zws/files/) and [OSF v2](https://osf.io/2urht/files/):

```
dataset/ZuCo/
├── task1-SR/Matlab files/     ← ZuCo v1.0 task1-SR .mat files
├── task2-NR/Matlab files/     ← ZuCo v1.0 task2-NR .mat files
├── task3-TSR/Matlab files/    ← ZuCo v1.0 task3-TSR .mat files
└── task2-NR-2.0/Matlab files/ ← ZuCo v2.0 task1-NR .mat files
```

### 2. Preprocess

```cmd
scripts\prepare_dataset.bat
```

This converts `.mat` → `.pickle` with `rawData` (sentence-level raw EEG) extraction.

Output: `dataset/ZuCo/<task>/pickle/<task>-dataset.pickle`

### 3. Verify

Each pickle should contain `rawData` with shape `(105, T)`:
```python
import pickle
with open('dataset/ZuCo/task1-SR/pickle/task1-SR-dataset.pickle', 'rb') as f:
    data = pickle.load(f)
subj = list(data.keys())[0]
print(data[subj][0]['rawData'].shape)  # (105, T)
```

## Training Pipeline

### Step 1: Pre-Training (Conformer + Re-Masked Token Prediction)

```cmd
scripts\train_pretrain.bat
```

**What it does:**
- Uses all ZuCo raw EEG data (single-view, 105 channels)
- Randomly masks 15% of time points, re-randomizes each epoch
- Conformer encoder learns to reconstruct the masked EEG
- Saves encoder weights for downstream multi-view use

**Hyperparameters:**
| Parameter | Value |
|---|---|
| batch_size | 4 |
| learning_rate | 5e-5 |
| epochs | 50 |
| d_model | 512 |
| pool_stride | 10 |
| mask_ratio | 0.15 |
| optimizer | AdamW |
| scheduler | CosineAnnealingLR |

**Output:** `checkpoints/pretrain/encoder_best.pt`

### Step 2: Multi-View Fine-Tuning

```cmd
scripts\train_multiview.bat
```

**What it does:**
1. Loads pre-trained encoder weights (temporal conv + transformer)
2. Initializes 10 regional encoders (spatial conv randomly initialized per region)
3. **Step 1 (20 epochs):** Freeze most BART params, train encoders with lr=5e-5
4. **Step 2 (30 epochs):** Unfreeze all BART, fine-tune with lr=5e-7
5. Each epoch: randomly unfreeze 3 view encoders, freeze 7 (per EEG2TEXT paper Table 7)

**Architecture:**
```
raw EEG (105, 5000)
    ↓ split by 10 brain regions
10 × RegionalConformerEncoder:
    ├─ temporal conv (pre-trained ✓)
    ├─ spatial conv (random init)
    ├─ AvgPool(stride=10) → ~500 tokens
    ├─ AdaptiveAvgPool1d(100) → 100 tokens
    ├─ transformer (pre-trained ✓)
    ↓
concat → (1000, 512)
    ↓ Global Transformer (3 layers)
    ↓ FC → (1000, 1024)
    ↓ BART decoder (pre-trained, fine-tuned)
    → text
```

**Output:** `checkpoints/multiview/best/<name>.pt`

### Step 3: Evaluation

```cmd
scripts\eval_multiview.bat
```

Runs 4 combinations (teacher_forcing × noise) and reports BLEU-1~4, ROUGE-1.

**Output:** `results/<name>_metrics.json`

## Key Design Decisions

### Channel-to-Region Mapping
Based on EEG2TEXT paper Table 1, using EGI HydroCel 128 system (105 channels after removing 23 outer-ring electrodes). See `data/channel_mapping.py`.

### RAW_EEG_MAX_LEN = 5000
Covers 90th percentile of sentence lengths (T distribution: median=2354, 90th=5183). Shorter sentences are zero-padded, longer ones are truncated.

### pool_stride = 10
Compresses 5000 → 500 tokens for the Conformer transformer. AdaptiveAvgPool1d(100) further compresses to 100 tokens per view for multi-view (10 × 100 = 1000 ≤ BART max 1024).

### Freeze 3/7 Strategy
Each epoch, 3 random view encoders are trainable while 7 are frozen. This saves VRAM and acts as regularization. Over 50 epochs, each encoder is trained ~15 times.

## Bug Fixes from Previous Version

1. **Optimizer rebuild** — Previously rebuilt optimizer each epoch (reset momentum → training instability). Fixed: only toggle `requires_grad`, optimizer stays the same.
2. **`import F`** — Was inside `main()`, not accessible in `train_one_epoch()`. Fixed: moved to top of file.
3. **`--one_step` flag** — Skipped Step 1, used only lr=5e-7 (too small). Fixed: use `--two_step` for proper two-stage training.
4. **SGD → AdamW** — More stable for this task.
5. **StepLR → CosineAnnealingLR** — Smoother learning rate decay.

## Results

### Pre-Training
| Configuration | Best val_loss |
|---|---|
| stride=50, d_model=256 | 0.525 |
| stride=10, d_model=512 | 0.290 |

### Multi-View Fine-Tuning
(To be filled after successful training run)

## References

- Wang, Z. and Ji, H. (2021). Open vocabulary EEG-to-text decoding and zero-shot sentiment classification.
- Liu, H. et al. (2024). EEG2TEXT: Open vocabulary EEG-to-text decoding with EEG pre-training and multi-view transformer.
- Song, Y. et al. (2022). EEG Conformer: Convolutional transformer for EEG decoding and visualization.
- Hollenstein, N. et al. (2018). ZuCo, a simultaneous EEG and eye-tracking resource for natural sentence reading.
