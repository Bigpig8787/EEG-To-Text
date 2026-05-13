ㄋ# TODO / 改動紀錄

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

## 2026-05-11 — Attention Padding Mask（length-aware）

### 動機
原架構 attention 完全沒 padding mask：
- `RegionalConformerEncoder.transformer` 直接吃所有 32 tokens（不論實際 EEG 長度）
- `encode()` 給 BART 的 `attention_mask = torch.ones(B, 330)` — 全部 attend
→ encoder/decoder 把 padding zeros 當訊號學，吃 capacity。先補 mask 再談動 attention 結構。

### 改動檔案

#### 1. `data/dataset.py`
- `get_input_sample`：`actual_T = min(T, RAW_EEG_MAX_LEN)`；`raw` 存在時 `raw_eeg_len = actual_T`，缺 raw 時 `= 0`。
- `__getitem__` 回傳 tuple 加第 10 元素 `s.get('raw_eeg_len', 0)`。

#### 2. `models/multiview.py`
- `RegionalConformerEncoder.forward(x, eeg_len=None, raw_max_len=5000)`：
  - `valid_n = ceil(eeg_len * T / raw_max_len).clamp(1, T)`（adaptive_pool 把 raw_max_len 映射到 T bins，bin i 有效 iff `i*raw_max_len/T < eeg_len`）
  - 建 `src_key_padding_mask`（CLS 永遠 valid，locals 由 `arange<valid_n` 決定）
  - `transformer(..., src_key_padding_mask=...)`
  - 回傳 3-tuple `(cls, local, local_valid)`
- `MultiViewConformerTranslator.encode(view_inputs, raw_eeg_lens=None)`：
  - 收 `valid_list`，建 BART `attn_mask = [cls_mask(B,V) | cat(valid_list)(B,V*T)]`，shape `(B, V+V*T)=(B, 330)`
  - 回傳 `(projected, attn_mask)`
- `forward` / `generate`：加 `raw_eeg_lens=None` kwarg，傳給 `encode`，用回傳的 `attn_mask` 取代 `torch.ones`。

#### 3. `train_multiview.py`
- DataLoader unpack 改 10-tuple，加 `raw_eeg_len`。
- `raw_eeg_lens = raw_eeg_len.to(device)`，傳 `model(..., raw_eeg_lens=raw_eeg_lens)`。

#### 4. `eval_multiview.py`
- 同上：unpack + 傳 `raw_eeg_lens` 給 `model()` 與 `model.generate()`。

### 向後相容
- `raw_eeg_lens=None` 為 default → `training/trainer.py`、`eval/evaluator.py`、`train_multiview_overfit.py` 沒改也不爆（fallback 走 `torch.ones`，等於舊行為）。
- 無新 `nn.Module` 參數 → 既有 `_s1.pt` / `_merged.pt` ckpt 可直接 load。

### 驗證
- `python -c "import ast; ast.parse(...)"` 4 檔皆 OK。
- 仍待實機跑訓練驗證。

### 預期影響
- Encoder 不再把 padding zeros 當訊號吃 → 短句樣本（< 5000 samples）特別受惠。
- BART cross-attention 只看 valid view tokens → 減少噪聲、應提升 BLEU/ROUGE。

### 待辦 / 風險
- [ ] 用最新 `_merged.pt` 當 `--resume` 重跑 step2，比較 mask 前後分數。
- [ ] 若 `raw_eeg_len=0`（缺 rawData）樣本仍流入 batch，`valid_n.clamp(min=1)` 會強制保 1 token，會引入小量噪聲；若多可考慮 batch-level skip。
- [ ] 其他 caller（`training/trainer.py`、`eval/evaluator.py`、`train_multiview_overfit.py`）尚未受惠 padding mask，需要時同步改。

## 2026-05-11 — k-CLS per region（cross-view bandwidth）

### 動機
原架構每 region 只 1 個 CLS 進 global transformer → 把 32 tokens 壓 1 token 才能跟其他 region 互動，bottleneck 嚴重。改成每 region 4 個 CLS（k=4），global transformer 看 V*k=40 tokens，bandwidth 4×，cost 仍便宜（40²=1600 vs 10²=100；遠小於 full token V*T=320, 320²=102400）。

### 改動檔案

#### `models/multiview.py`
- `RegionalConformerEncoder.__init__`：加 `n_cls_tokens=4`；`cls_token` shape `(1,1,d)`→`(1,k,d)`；`pos_encoding.max_len` 含 k。
- `RegionalConformerEncoder.forward`：cls expand 成 (B,k,d)；padding mask 的 `cls_valid` 改 `ones(B,k)`；slice 回傳 `x[:,:k,:]` / `x[:,k:,:]`。
- `MultiViewConformerTranslator.__init__`：加 `n_cls_per_view=4`；傳給每個 region encoder；`view_pos_embed` shape `(1,V,d)`→`(1,V*k,d)`（每個 slot 獨立身份）。
- `MultiViewConformerTranslator.encode`：
  - `cls_seq = cat(cls_list, dim=1)` → `(B, V*k, d)`
  - `global_cls` 同樣 `(B, V*k, d)`
  - residual broadcast：每 region 取其 k 個 enriched CLS 的 mean，加回該 region locals
  - `cls_mask = ones(B, V*k)`；BART 看 `V*k + V*T = 40+320 = 360` tokens

### 向後相容（破）
- `cls_token`、`view_pos_embed` shape 改 → 舊 `_s1.pt` / `_merged.pt` (k=1) 用預設 k=4 load 會炸。
- 解法：
  - (a) **fresh restart** step1（推；reset CLS 重學）
  - (b) eval/train 顯式傳 `n_cls_per_view=1` 復原舊行為

### 驗證
- AST OK。
- 仍待實機跑 fresh step1 驗 BLEU/ROUGE。

### 待辦 / 風險
- [ ] fresh step1 (k=4) 跑完比較 k=1 baseline。
- [ ] k=4 是猜測；可 sweep k∈{2,4,8}。
- [ ] residual broadcast 用 mean(k CLSs) 可改 sum 或 attention pool，差異待測。

---

### 未採用方案（備案）
之前討論過的 cross-view bandwidth 方案，目前只實作 option 1，其餘留待 option 1 結果不佳再嘗試：

- **Option 2 — skip global transformer**：直接把 V*T=320 tokens 給 BART，讓 BART 自己學 cross-view（少一層先驗，看 BART 容量是否吃得下）。改動最小：刪 global_transformer + view_pos_embed + cls。
- **Option 3 — Perceiver latents**：N 個 learnable query 對 V*T cross-attend → 固定 N 個 latent 進 BART。bandwidth 可調、cost 跟 N 線性。改動最大：新 module。
- **Option 4 — full token global transformer**：所有 V*T=320 tokens 進 global，最強表達能力，但 attn cost 102400（vs k=4 的 1600）。GPU 吃得下才考慮。
