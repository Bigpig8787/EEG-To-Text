# TODO / 改動紀錄

## 2026-05-10 — Step2 全 BART Fine-tune 模式 (no LoRA)

### 目的
從 step1 ckpt (`_s1.pt`) 接續訓練，step2 改為全 BART unfrozen fine-tune（取代 LoRA），跑 15 epochs。

### 改動檔案

#### 1. `config.py`
- 新增 `--no_lora` 旗標 (`action='store_true'`, default False)
- 用於 step2 停用 LoRA、改全 BART fine-tune

#### 2. `train_multiview.py`
- 讀取 `NO_LORA = args.get('no_lora', False)`
- `save_suffix` 加 `_full` 區分輸出檔
- Step 2 區塊新增 NO_LORA 分支：
  - 跳過 `get_peft_model(...)` 與 LoRA 套用
  - `for p in model.pretrained.parameters(): p.requires_grad = True` — 全 BART unfrozen
  - opt2 改為兩組：`enc_p2: LR2*0.2`、`other_p2 (全 BART): LR2*0.1`
  - 跳過 `merge_and_unload()`，直接 `torch.save` plain state_dict 為 `_merged.pt`
- 原 LoRA 路徑保留為 `else` 分支，無 `--no_lora` 時行為不變

#### 3. `scripts/train_multiview_full.bat` (新檔)
- `--resume` 指向 5-task step1 ckpt：
  `./checkpoints/multiview/best/task1_task2_task3_taskNRv2_taskTSRv2_multiview_2step_b4_50_70_5e-05_5e-07_unique_sent_s1.pt`
- `--no_lora` + `--num_epoch_step1 0` + `--num_epoch_step2 15`
- `-lr2 0.000001` (LR2=1e-6 → BART=1e-7, encoder=2e-7)
- `--no_early_stop`

### 輸出檔名
- best/last: `task1_task2_task3_taskNRv2_taskTSRv2_multiview_2step_b4_0_15_5e-05_1e-06_unique_sent_cont_full.pt`
- merged (eval-compatible plain state_dict): `..._cont_full_merged.pt`
- 不覆蓋既有 `_s1.pt` / `_merged.pt`

### 驗證
- `python -c "import ast; ast.parse(...)"` 兩檔皆 OK
- 仍待實機跑訓練驗證

### 待辦 / 風險
- [ ] 確認 `_s1.pt` 實存於 `./checkpoints/multiview/best/`，若僅 `last/` 需改 `--resume` 路徑
- [ ] LR2=1e-6 過激進可降回 5e-7（BART=5e-8 偏保守但更穩）
- [ ] 跑完後用 `eval_multiview.py` 對 `_cont_full_merged.pt` 評分，比較 BLEU/ROUGE vs 原 `_merged.pt`

## 2026-05-11 — CLIP-style EEG↔Text Contrastive Pre-training

### 目的
在現有 masked-token reconstruction pretrain 之上，再加一階段對比式 pretrain，把 EEG encoder 輸出對齊到 BART 文字 encoder 的 sentence embedding 空間。預期讓下游 multiview 訓練 BLEU/ROUGE 提升。

### 設計選擇
- **兩階段**：保留 `train_pretrain.py` (MSE recon)，**接續**訓練 contrastive，不取代
- **起點**：load 既有 `./checkpoints/pretrain/encoder_best.pt`
- **Loss**：純 InfoNCE (symmetric cross-entropy on EEG↔Text logits, τ=0.07)
- **文字端**：`facebook/bart-large` encoder 完全凍結 (1024-dim sentence embedding via mean-pool over non-pad tokens)
- **EEG 端**：`ConformerEncoder` + 兩層 projection head (Linear→GELU→Linear) → L2 norm
- **batch=32** (in-batch negatives 對 contrastive 至關重要)
- **30 epochs**, lr=5e-5, AMP + grad clip 1.0

### 新增檔案

#### 1. `data/pretrain_clip_dataset.py`
- `EEGPretrainCLIPDataset`：仿 `EEGPretrainDataset`，但同時保存 `sent['content']` 文字
- `__getitem__` 回 `(raw_eeg, actual_len, text_str)`
- 8:2 train/dev split 與原 dataset 一致

#### 2. `models/pretrain_clip_model.py`
- `ConformerCLIPModel`：包 `ConformerEncoder` + `proj` (Linear 512→512→1024)
- `encode()` 做 mean-pool over time → projection → L2 normalize
- `load_encoder_weights(path)` 接受兩種格式：
  - 純 encoder state_dict (`encoder_best.pt`)
  - 完整 model dict (含 `model_state_dict` + `encoder.` 前綴的 `pretrain_best.pt`)
- 訓練後存的 `encoder_clip_best.pt` 是純 encoder state_dict，**完全相容** `train_multiview.py` 的 `model.load_pretrained_encoder()`

#### 3. `train_pretrain_clip.py`
- 完整 train/val 迴圈、AMP、tqdm、custom collate (BART tokenizer 批次處理)
- InfoNCE：`logits = z_eeg @ z_txt.T / τ`，cross_entropy 兩方向取平均
- 監控指標：loss + retrieval **acc@1** (對角線命中率)
- 存兩個 ckpt：
  - `pretrain_clip_best.pt`：含 model+args+epoch (full snapshot)
  - `encoder_clip_best.pt`：encoder only (給下游用)

#### 4. `scripts/pretrain_clip.bat`
- batch=32, 30 ep, lr=5e-5, τ=0.07, mask_ratio=0.15 (EEG augmentation 仍開)
- `--encoder_init_path ./checkpoints/pretrain/encoder_best.pt`

### 用法
```bat
REM 1. 先跑既有 pretrain (MSE) 取得 encoder_best.pt
scripts\train_pretrain.bat

REM 2. 再跑 contrastive 微調 → 產生 encoder_clip_best.pt
scripts\pretrain_clip.bat

REM 3. 下游訓練改用 contrastive encoder：
REM    把 encoder_clip_best.pt 重命名 / 複製為 encoder_best.pt
REM    或在 train_multiview.py 的 encoder_path 換指向
```

### 驗證
- 三檔 AST parse 全 OK
- 仍待實機跑訓練驗證 InfoNCE loss 下降 + acc@1 上升

### 待辦 / 風險
- [ ] batch=32 可能 OOM (BART-large encoder 凍結但仍要 forward)；不夠就降到 16 並可考慮多 GPU
- [ ] τ=0.07 是 CLIP 預設；ZuCo 樣本數小可能要調 0.1-0.2 比較穩
- [ ] mean-pool over BART encoder hidden 比 CLS-pool 簡單，若效果差可改 `<s>` token embedding
- [ ] EEG 端 mean-pool 沒做 mask-aware (encoder 下採樣後時間軸已模糊)，可能要改 attention pool
- [ ] 若 contrastive pretrain 後下游 BLEU 反而降 → 對齊空間與 BART decoder 期望輸入不匹配，需試 freeze projection 或調整目標空間 (例如改對齊到 BART decoder 的 cross-attn input)
- [ ] 需要驗證 ZuCo pickle 確實有 `'content'` 欄位 (來自 dataset.py 的使用得知有，但仍應實機測)
