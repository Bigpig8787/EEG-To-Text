# 下次訓練前必看 NOTES

最後更新：2026-04-22

## 當前超參數（paper spec）

| 參數 | 值 |
|---|---|
| d_model | 512 |
| n_views | 10 |
| n_filters | 40 |
| temporal_kernel | 25 |
| n_encoder_layers (local) | 4 |
| n_global_layers | 3 |
| dropout | 0.1 |
| LR1 (Step-1 encoder warm-up) | 5e-5 |
| LR2 (Step-2 LoRA fine-tune) | 5e-7 |
| Batch size | 4 |
| Epochs | 25 (Step-1) + 35 (Step-2) |
| LoRA r | 16 (alpha = r) |
| label_smooth | 0.1 |
| weight_decay | 0.05 |

對應檔案：
- `train_multiview.py:248` — 模型架構
- `scripts/train_multiview.bat` — CLI 參數
- `eval_multiview.py:102` — 評分時的模型重建（必須和 train 一致）

## 資料切分

- `data/dataset.py:144` — 目前 **train 80% / dev 20%**（test 與 dev 共用 20%）
- `unique_sent` setting
- 改回 8:1:1 需還原 `dev_div = train_div + int(0.1 * total)`

## ⚠️ 開始訓練前必確認

### 1. Checkpoint 檔名會變
`save_name` 格式：`{TASK}_multiview_2step_b{BS}_{E1}_{E2}_{LR1}_{LR2}_unique_sent`

新的檔名會是：
```
task1_task2_taskNRv2_multiview_2step_b4_25_35_5e-05_5e-07_unique_sent_merged.pt
```

舊的 `..._5e-06_..._merged.pt`（架構 2 layers + dropout 0.3）**不能**用新的 eval 腳本載入，會 shape mismatch。

### 2. 訓練完要同步更新 eval 腳本
`scripts/eval_multiview.bat` 的 `CKPT` / `CONF` 路徑含 LR2，每次換參數都要改：
```bat
set CKPT=./checkpoints/multiview/best/task1_task2_taskNRv2_multiview_2step_b4_25_35_5e-05_5e-07_unique_sent_merged.pt
set CONF=./config/decoding/task1_task2_taskNRv2_multiview_2step_b4_25_35_5e-05_5e-07_unique_sent.json
```

### 3. 模型規模與資源
- 層數從 2/2 → 4/3，參數量約 **70M → 120-140M**（Step-1 可訓練）
- 加上 BART-large 全模型 ≈ **~500M total**
- 磁碟：FP32 checkpoint 約 **2 GB**
- VRAM：訓練時（AdamW + AMP）粗估 **8-12 GB**，如爆記憶體先降 batch size

### 4. 過擬合風險
- dropout 從 0.3 降回 0.1 + 層數變深 → 更容易過擬合 ZuCo 小資料集
- 依賴 `label_smooth=0.1` + `weight_decay=0.05` + EEG augmentation 撐住
- 如果 val loss 在 epoch 1 就是最低 → 回頭調高 dropout（例如 0.2）或降 layers

### 5. 訓練指令
```bat
scripts\train_multiview.bat
```
會跑 Step-1（25 epochs）+ Step-2（35 epochs）+ LoRA merge，最後才產出 `_merged.pt`。

### 6. 評分指令（訓練完後）
```bat
scripts\eval_multiview.bat
```
跑 4 種組合；**[3/4] Free generation + real EEG** 是主要結果。
