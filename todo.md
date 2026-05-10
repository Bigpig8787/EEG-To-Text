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
