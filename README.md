# EEG-To-Text：多視角 Conformer + LoRA 微調

本專案實作論文 [arxiv 2405.02165](https://arxiv.org/abs/2405.02165) 的 Multi-View EEG-to-Text 解碼系統。
以 10 個腦區的原始 EEG 訊號為輸入，透過 Conformer 編碼器 + BART 解碼器生成對應文字。

---

## 目錄結構

```
EEG-To-Text/
├── data/
│   ├── dataset.py            # ZuCo_dataset：EEG+文字對齊資料集
│   ├── pretrain_dataset.py   # EEGPretrainDataset：僅原始 EEG，供預訓練用
│   ├── channel_mapping.py    # 105 通道 → 10 腦區映射
│   └── pipeline.py           # EEGDataPipeline：DataLoader 封裝
├── models/
│   ├── base.py               # BaseEEGModel 抽象介面
│   ├── conformer.py          # ConformerEncoder（共用骨幹）
│   ├── multiview.py          # MultiViewConformerTranslator（主模型）
│   ├── brain_translator.py   # BrainTranslator / BrainTranslatorNaive（基準模型）
│   └── pretrain_model.py     # ConformerPreTrainModel（預訓練用）
├── training/
│   ├── trainer.py            # Trainer + TrainerConfig（通用訓練迴圈）
│   ├── pretrain_pipeline.py  # PretrainPipeline
│   └── finetune_pipeline.py  # FinetunePipeline（兩步驟微調）
├── eval/
│   └── evaluator.py          # Evaluator（評分模組）
├── scripts/
│   ├── train_pretrain.bat    # 一鍵預訓練
│   ├── train_multiview.bat   # 一鍵兩步驟微調
│   └── eval_multiview.bat    # 一鍵評分
├── train_pretrain.py         # 預訓練入口（standalone）
├── train_multiview.py        # 微調入口（standalone）
├── eval_multiview.py         # 評分入口（standalone）
├── run_pipeline.py           # 完整 pipeline 一鍵入口
├── config.py                 # argparse 設定
└── metrics.py                # WER / BLEU / ROUGE 計算
```

---

## 模型架構

### MultiViewConformerTranslator

```
原始 EEG (105 ch × 5000 tp)
        │
        ▼ split_raw_eeg_by_region()
┌──────────────────────────────┐
│  10 個腦區 RegionalEncoder   │  各自獨立處理
│  prefrontal (26ch)           │
│  premotor   (16ch)           │
│  brocas     ( 4ch)           │
│  ...                         │
└──────────────────────────────┘
        │  每區輸出 (B, 100, 512)
        ▼ concat
  (B, 1000, 512)  — 10 區 × 100 tokens
        │
        ▼ GlobalTransformer (2 層)
  (B, 1000, 512)
        │
        ▼ fc1 + ReLU
  (B, 1000, 1024)   ← BART embedding size
        │
        ▼ BART decoder (inputs_embeds)
  生成文字
```

### RegionalConformerEncoder（每個腦區）

```
Input: (B, C_region, 5000)
  → temporal_conv  Conv2d(1→40, k=(1,200))  + BN
  → spatial_conv   Conv2d(40→40, k=(C,1))  + BN + ELU
  → AvgPool2d(stride=100)  →  (B, 40, 50)
  → Dropout(0.3)
  → AdaptiveAvgPool1d(32)  →  (B, 40, 32)
  → projection Linear(40→512)
  → PositionalEncoding
  → TransformerEncoder(2 層, d_model=512, nhead=8)
Output: (B, 100, 512)
```

### 腦區對應（BRAIN_REGION_CHANNEL_COUNT）

| 腦區 | 通道數 |
|------|--------|
| prefrontal | 26 |
| premotor | 16 |
| primary_sensory | 11 |
| visual | 11 |
| somatic_sensory | 9 |
| auditory_assoc | 9 |
| primary_motor | 9 |
| wernickes | 6 |
| brocas | 4 |
| auditory | 4 |

---

## 訓練策略

### Step 1：Encoder Warm-up（BART 凍結）

- 凍結所有 BART 參數
- 只訓練：`view_encoders` + `global_transformer` + `fc1`
- LR = 5e-5，AdamW weight_decay=0.05
- Cosine schedule + 10% warmup
- 25 epochs（預設）

### Step 2：LoRA Fine-tune

- 對 BART 套用 LoRA（r=16, alpha=32, dropout=0.15）
- Target modules：`q_proj`, `k_proj`, `v_proj`, `out_proj`
- 3 個 param group：

| Group | LR | 說明 |
|-------|----|------|
| enc_p2（EEG 側） | LR2 = 5e-6 | encoder 穩定微調 |
| lora_p2（LoRA adapter） | LR2 × 2.0 = 1e-5 | **必須高於 encoder**，讓 BART 快速適配 EEG 輸入 |
| other_p2 | LR2 × 0.5 = 2.5e-6 | 其他參數 |

- 35 epochs（預設）
- 結束後 `merge_and_unload()` 合併 LoRA 權重，存為 `_merged.pt`

### EEG Data Augmentation（訓練時）

每個 batch 在 GPU 上即時套用，驗證集不套用：

| 增強方式 | 機率 | 參數 |
|----------|------|------|
| 振幅縮放 | 70% | uniform(0.80, 1.20) |
| 高斯噪聲 | 50% | σ = 0.03 |
| 時間位移 | 50% | ±5% of T（circular roll） |

---

## 超參數一覽（目前最佳設定）

| 參數 | 值 | 說明 |
|------|-----|------|
| `n_encoder_layers` | 2 | 每腦區 Transformer 層數（原 4，過擬合） |
| `n_global_layers` | 2 | Global Transformer 層數（原 3） |
| `dropout` | 0.3 | 全局 dropout（原 0.1，過低） |
| `lora_r` | 16 | LoRA rank（r=32 參數過多） |
| `lora_alpha` | 32 | = r × 2 |
| `lora_dropout` | 0.15 | LoRA 內部 dropout |
| `lora_targets` | q/k/v/out_proj | 覆蓋所有 attention projection |
| `weight_decay` | 0.05 | L2 正則（原 0.01） |
| `label_smooth` | 0.1 | 交叉熵 label smoothing |
| `batch_size` | 4 | GPU 記憶體限制 |
| `grad_accum_steps` | 2 | 等效 batch = 8 |
| `mask_ratio`（pretrain） | 0.15 | 遮蔽比（paper-spec） |
| `temporal_kernel` | 200 | CNN kernel（paper-spec，原 25） |
| `pool_stride` | 100 (pretrain) / 50 (multiview) | pretrain CNN stride=100（5000→50）；multiview 預設 stride=50（5000→100） |
| `tokens_per_view` | 64 (multiview) | AdaptiveAvgPool 後每 view token 數。multiview=64（V*(k+T)=10*(8+64)=720≤1024）；pretrain 不適用 |

### 架構已改為 CLI 可設定（2026-08-25）

`MultiViewConformerTranslator` 的幾何參數原本寫死在 `train_multiview.py` 與
`eval_multiview.py` 兩處，必須手動同步。現在全部是 `config.py` 的參數，
**預設值等於原本寫死的數字**，既有腳本產生的模型與 `save_name` 完全不變。

| 參數 | 預設 | 說明 |
|------|------|------|
| `--d_model` | 512 | 腦區與 global transformer 寬度 |
| `--n_filters` | 40 | temporal conv 輸出通道 |
| `--n_spatial_filters` | None | spatial conv 輸出通道；None = 同 `n_filters`（原行為） |
| `--temporal_kernel` | 200 | temporal conv kernel 長度 |
| `--pool_stride` | 50 | 原始時間軸 AvgPool kernel |
| `--tokens_per_view` | 64 | adaptive pooling 後每腦區 local token 數 |
| `--n_cls_per_view` | 8 | 每腦區 CLS token 數 |
| `--n_heads` | 8 | attention heads（local 與 global 共用） |
| `--n_encoder_layers` | 4 | 每腦區 transformer 層數 |
| `--n_global_layers` | 3 | global transformer 層數 |
| `--dropout` | 0.1 | encoder 與 transformer 內部 dropout |
| `--lr_lora` | None | STEP 2 LoRA 絕對學習率；None = 沿用 `LR2 × 2.0` |
| `--save_suffix` | `''` | 附加到 `save_name`，避免只差 `lora_r` 或架構的兩次 run 互相覆蓋 |

`eval_multiview.py` 改為從訓練寫出的 `config/decoding/<save_name>.json` 讀回幾何
參數，evaluation 不可能再與訓練不一致。舊的 config 沒有這些欄位時會退回上表預設值。

### 與 SNN t=4 對齊的 ANN run

`scripts/train_multiview_snn_matched.bat` 把 ANN 的每一個超參數對齊
`Spiking-EEG2TEXT/configs/multiview_snn_v2.json`（SNN 是錨點，完全不動；ANN 縮小
去對齊），架構與序列長度同時受控：

```
--d_model 256 --n_filters 64 --n_spatial_filters 32 --temporal_kernel 250
--pool_stride 100 --tokens_per_view 32 --n_cls_per_view 4
--n_encoder_layers 2 --n_global_layers 2 -lr2 0.000005 --lr_lora 0.0001 --lora_r 8
```

實測 encoder 側參數量：ANN 18,119,808 vs SNN 19,158,656（ANN 少 5.4%），
進 BART 的序列兩邊都是 10×(4+32)=360 token。殘餘 5.4% 是結構差異而非超參數差異：
SNN 每個腦區多了可學的 `positional_embedding` 與 `before_transformer_linear`，
ANN 的 `PositionalEncoding` 是無參數的 sinusoidal，projection 也直接接到 `d_model`。
細節見 `version.md`。

> **注意**：`checkpoints/pretrain/encoder_best.pt` 是舊幾何訓練出來的，載不進這個
> 模型。`load_pretrained_encoder` 會逐一跳過形狀不符的張量並印出警告，STEP 1 等同
> 從隨機初始化開始。要暖啟動就得先用同一組幾何重跑 ANN 的 MAE pre-training。

---

## 快速開始

### 環境需求

```bash
pip install torch transformers peft tqdm nltk torchmetrics
python -c "import nltk; nltk.download('punkt_tab')"
```

### 1. 資料預處理

ZuCo 資料集需先從 `.mat` 轉 `.pickle`：

```bash
python util/construct_dataset_mat_to_pickle_v1.py
```

預期目錄結構：
```
dataset/ZuCo/
├── task1-SR/pickle/task1-SR-dataset.pickle
├── task2-NR/pickle/task2-NR-dataset.pickle
├── task3-TSR/pickle/task3-TSR-dataset.pickle
└── task2-NR-2.0/pickle/task2-NR-2.0-dataset.pickle
```

### 2. 預訓練（Conformer Encoder）

```bash
scripts\train_pretrain.bat
# 或直接執行：
python train_pretrain.py -b 4 -ne 50 -lr 5e-5 --temporal_kernel 200 --pool_stride 100 --mask_ratio 0.15 --dropout 0.2 --n_transformer_layers 2 -s ./checkpoints/pretrain -cuda cuda:0
```

輸出：
- `checkpoints/pretrain/pretrain_best.pt`（完整模型）
- `checkpoints/pretrain/encoder_best.pt`（只有 encoder，供 fine-tune 載入）

### 3. 兩步驟微調

```bash
scripts\train_multiview.bat
# 或直接執行：
python train_multiview.py \
    -m MultiViewConformerTranslator \
    -t task1_task2_taskNRv2 \
    -2step -pre \
    -ne1 25 -ne2 35 \
    -lr1 5e-5 -lr2 5e-6 \
    -b 4 \
    --no_early_stop \
    --lora_r 16 \
    --label_smooth 0.1 \
    -s ./checkpoints/multiview \
    -cuda cuda:0
```

輸出（在 `checkpoints/multiview/`）：
- `best/{name}_s1.pt` — Step 1 最佳權重
- `best/{name}.pt`    — Step 2 最佳權重
- `best/{name}_merged.pt` — LoRA 合併後的最終推論用權重

### 4. 評分

```bash
scripts\eval_multiview.bat
```

評分 4 個條件（teacher forcing / free generation × real EEG / noise）：

```bash
python eval_multiview.py \
    -checkpoint ./checkpoints/multiview/best/{name}_merged.pt \
    -conf ./config/decoding/{name}.json \
    -tf True -n False -cuda cuda:0
```

輸出：
- `results/{task}_results.txt` — 預測 vs. 答案
- `results/{task}_metrics.json` — WER / BLEU-1~4 / ROUGE

### 5. 完整 Pipeline（一鍵）

```bash
python run_pipeline.py \
    --task task1_task2_taskNRv2 \
    --cuda cuda:0 \
    --no_early_stop \
    --checkpoint_dir ./checkpoints/pipeline
```

跳過 pretrain：
```bash
python run_pipeline.py --skip_pretrain --pretrained_encoder ./checkpoints/pretrain/encoder_best.pt
```

從 Step 2 斷點繼續：
```bash
python run_pipeline.py --skip_pretrain --resume_step2 ./checkpoints/multiview/last/model_s2.pt
```

只評分：
```bash
python run_pipeline.py --eval_only --eval_ckpt ./checkpoints/multiview/best/model_merged.pt
```

---

## 模組化 API

### DataPipeline

```python
from data.pipeline import EEGDataPipeline

pipeline = EEGDataPipeline(
    task_name='task1_task2_taskNRv2',
    subject='ALL',
    eeg_type='GD',
    batch_size=4,
)
loaders = pipeline.build()          # {'train': ..., 'dev': ..., 'test': ...}

# 快取到磁碟（加速下次載入）
pipeline.save_cache('./cache/zuco.pkl')
pipeline2 = EEGDataPipeline.load_cache('./cache/zuco.pkl', batch_size=4)
```

### 模型介面（BaseEEGModel）

所有模型均實作以下介面，可直接替換：

```python
model.eeg_encoder          # view_encoders ModuleDict（各腦區編碼器）
model.global_transformer   # 全局 Transformer
model.language_model       # BART (BartForConditionalGeneration)

model.freeze_language_model()    # 凍結 BART（Step 1 用）
model.unfreeze_language_model()  # 解凍 BART
model.get_encoder_params()       # EEG 側全部參數
model.get_lm_params()            # LM 全部參數
model.get_tuning_targets()       # → {'encoder': [...], 'lm': [...]}

model.encode(view_inputs)        # EEG → (B, 1000, 1024) encoder hidden states
model.forward(view_inputs, masks, masks_inv, target_ids)   # 訓練用
model.generate(view_inputs, masks, masks_inv, **gen_kwargs) # 推論用
```

### 替換模型

只需實作 `BaseEEGModel` 的抽象方法，即可直接接入所有訓練/評分流程：

```python
from models.base import BaseEEGModel

class MyNewModel(BaseEEGModel):
    @property
    def eeg_encoder(self): return self.my_encoder
    @property
    def global_transformer(self): return self.my_global_tf
    @property
    def language_model(self): return self.pretrained
    def get_tuning_targets(self): ...
    def encode(self, inputs, ...): ...
    def forward(self, inputs, ...): ...
    def generate(self, inputs, ...): ...

# 接入 FinetunePipeline
from training.finetune_pipeline import FinetunePipeline, FinetuneConfig
pipeline = FinetunePipeline(MyNewModel(...), loaders, tokenizer, FinetuneConfig(), device)
pipeline.run()
```

### Trainer（單獨使用）

```python
from training.trainer import Trainer, TrainerConfig

cfg = TrainerConfig(
    num_epochs=30,
    patience=10,
    no_early_stop=False,
    label_smooth=0.1,
    checkpoint_dir='./checkpoints/custom',
    checkpoint_name='my_model',
    resume_from='./checkpoints/custom/last/my_model.pt',  # 從斷點繼續
    save_every=5,  # 每 5 epoch 額外存一份
)
trainer = Trainer(model, dataloaders, optimizer, scheduler, tokenizer, cfg, device)
best_model = trainer.train()
```

### Evaluator（獨立評分）

```python
from eval.evaluator import Evaluator

# 從 checkpoint 載入
ev = Evaluator.from_checkpoint(
    './checkpoints/multiview/best/model_merged.pt',
    model_class=MultiViewConformerTranslator,
    model_kwargs={'pretrained_bart': bart, ...},
    tokenizer=tokenizer,
    device=device,
)

# 評分單一條件
metrics = ev.evaluate(test_loader, teacher_forcing=False, use_noise=False)
print(metrics)  # {'bleu-4': ..., 'rouge1_fmeasure': ..., 'wer': ..., ...}

# 評分全部 4 條件
all_results = ev.evaluate_all(test_loader)
ev.save_results(all_results, './results/eval.json')
```

---

## 已知問題與修復記錄

### [2026-04-19] 過擬合問題系統性修正

**問題**：Free generation BLEU-4 = 0.0000，Teacher forcing BLEU-4 = 0.0287（論文目標 0.121）

**根本原因**（按嚴重程度）：

1. **LoRA LR 設定錯誤（致命）**
   - `lora_p2` LR = `LR2 × 0.1 = 5e-7`，LoRA adapter 幾乎無法更新
   - **修正**：改為 `LR2 × 2.0 = 1e-5`

2. **模型容量過大**
   - `n_encoder_layers=4, n_global_layers=3` → 43 Transformer 層，11K 樣本養不起
   - **修正**：`n_encoder_layers=2, n_global_layers=2`

3. **Dropout 不足**
   - 全局 `dropout=0.1` 對 416M 參數模型嚴重不足
   - **修正**：`dropout=0.3`，LoRA dropout: 0.1 → 0.15

4. **Fine-tune 無 EEG 資料增強**
   - 模型直接記憶訓練集 EEG 波形
   - **修正**：加入振幅縮放 + 高斯噪聲 + 時間位移（`augment_eeg_views()`）

5. **Weight decay 過低**
   - `weight_decay=0.01` → **修正**：`0.05`

6. **LoRA rank 過高**
   - `r=32` 對 11K 樣本而言參數過多 → **修正**：`r=16`

7. **Pretrain mask ratio 過低**
   - `mask_ratio`：曾調至 `0.30` 試圖增強 robustness，現回到 paper-spec `0.15`

---

### [2026-04-18] Free generation 完全失效

**問題**：生成參數 `repetition_penalty=5.0` 導致任何 4-gram 都無法形成，BLEU-4 = 0.0000

**修正**（`eval_multiview.py`）：
- `repetition_penalty`: 5.0 → 1.5
- `max_length` → `max_new_tokens=50`
- 加入 `no_repeat_ngram_size=3`
- 加入 `forced_bos_token_id`

---

### [2026-04-18] 兩步驟訓練架構

**問題**：原始訓練為單步驟，LoRA 從一開始就套在 BART 上，EEG encoder 還沒學會任何表示就開始與 BART 聯合訓練。

**修正**：重構為兩步驟訓練（符合論文 2405.02165）：
- Step 1：凍結 BART，專門訓練 EEG encoder（warm-up）
- Step 2：對 BART 套用 LoRA，全部聯合微調

---

## 評分指標說明

| 指標 | 說明 | 論文目標（free gen） |
|------|------|---------------------|
| BLEU-4 | 4-gram 精確匹配 | **0.121** |
| BLEU-1 | 1-gram 精確匹配 | ~0.40 |
| ROUGE-1 F1 | Unigram 重疊 | ~0.35 |
| WER | 詞錯誤率（越低越好） | ~0.70 |

評分以 **free generation + real EEG**（非 teacher forcing）為主要指標。
Teacher forcing 分數永遠高於 free generation，不代表模型真正學到 EEG→text 映射。
Noise EEG 與 Real EEG 分數若相近，代表模型**沒有利用 EEG 訊號**。

---

## 參考論文

- 主要論文：[Multi-View EEG-to-Text, arxiv 2405.02165](https://arxiv.org/abs/2405.02165)
- ZuCo 資料集：Hollenstein et al., 2018/2020
- BART：[Lewis et al., 2019](https://arxiv.org/abs/1910.13461)
- LoRA：[Hu et al., 2021](https://arxiv.org/abs/2106.09685)
- Conformer（EEG）：EEGNet-based spatial-temporal filtering
