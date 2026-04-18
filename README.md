# EEG-To-Text: 多視角 Conformer 與 EEG 預訓練

本專案實作了開放詞彙的 EEG 到文本（EEG-to-Text）解碼，結合了 Conformer 編碼器、EEG 預訓練技術以及多視角 Transformer 架構。透過 ZuCo 資料集，本專案提供了一個強大的 Pipeline，能將大腦信號解碼為自然語言。

本專案基於以下研究：
- [EEG2TEXT (Liu et al., 2024)](https://arxiv.org/abs/2405.02165)
- [EEG Conformer (Song et al., 2022)](https://ieeexplore.ieee.org/document/9991178)

---

## 🏗️ 新架構總覽

專案已重構為模組化、面向 Pipeline 的架構，以提升可維護性與實驗靈活性。

```text
EEG-To-Text/
├── models/
│   ├── base.py              # BaseEEGModel 抽象介面
│   ├── multiview.py         # 多視角 Conformer 翻譯器（實作 BaseEEGModel）
│   ├── conformer.py         # 共享的 Conformer 編碼器區塊
│   └── brain_translator.py  # 基準模型 (Wang & Ji, 2021)
├── data/
│   ├── pipeline.py          # EEGDataPipeline（載入、快取、生成 DataLoader）
│   └── channel_mapping.py   # 105 通道 → 10 個腦區的映射
├── training/
│   ├── trainer.py           # Trainer + TrainerConfig（訓練迴圈核心）
│   ├── pretrain_pipeline.py # PretrainPipeline（遮蔽重建預訓練）
│   └── finetune_pipeline.py # FinetunePipeline（兩步驟 LoRA 微調）
├── eval/
│   └── evaluator.py         # 評估模組（計算 BLEU, ROUGE 分數）
└── run_pipeline.py          # 統一的 Pipeline 執行入口腳本
```

### 核心組件說明

- **`EEGDataPipeline`**: 負責資料集載入、受試者篩選與 BART 分詞（Tokenization）。內建快取機制，能顯著提升重複實驗的效率。
- **`BaseEEGModel`**: 抽象介面，確保任何新模型都提供必要的屬性（`eeg_encoder`, `global_transformer`, `language_model`）以及用於 LoRA/微調設定的 `get_tuning_targets()`。
- **`Trainer`**: 統一的訓練引擎，支援早停（Early Stopping）、權重存檔（Checkpointing）與梯度累積。
- **`Evaluator`**: 解耦的評估邏輯，可獨立於訓練流程之外執行。

---

## 🚀 快速開始

`run_pipeline.py` 是執行訓練與評估的唯一入口。

### 1. 執行完整 Pipeline
依序執行預訓練、兩階段微調以及最終評估。
```bash
python run_pipeline.py --task task1_task2_taskNRv2 --cuda cuda:0 --no_early_stop
```

### 2. 恢復微調（從 Step 2 開始）
跳過預訓練，直接從特定的權重存檔繼續 Step 2 訓練。
```bash
python run_pipeline.py --skip_pretrain --resume_step2 ./checkpoints/pipeline/finetune/last/multiview_s2.pt
```

### 3. 僅執行評估
對已存檔的模型進行評分。
```bash
python run_pipeline.py --eval_only --eval_ckpt ./checkpoints/multiview/best/model_merged.pt
```

---

## 🛠️ 模型介面說明

若要更換模型，只需實作 `BaseEEGModel` 並更新 `run_pipeline.py` 中的 `build_model()` 函數即可。

**`BaseEEGModel` 必備屬性：**
- `model.eeg_encoder` → 腦區編碼器的 `ModuleDict`。
- `model.global_transformer` → 全局注意力層。
- `model.language_model` → 語言模型主體（如 BART）。
- `model.get_tuning_targets()` → 回傳參數組字典（如 'encoder', 'lm'），方便設定 LoRA。

---

## 📊 技術細節

### 腦區映射（Channel-to-Region Mapping）
基於 EGI HydroCel 128 系統，我們在移除 23 個外圈電極後，將剩下的 **105 個通道** 映射至 **10 個腦區**（左右大腦的額葉、顳葉、頂葉、枕葉）。詳見 `data/channel_mapping.py`。

### 多視角策略（Multi-View Strategy）
- **時空壓縮**：每個腦區編碼器使用 Conformer 區塊，透過 `pool_stride=10` 與 `AdaptiveAvgPool1d(100)` 為每個區域生成 100 個 Token。
- **拼接與全局注意力**：10 個視角 × 100 Token = 1000 Token，經由全局 Transformer 處理後輸入至 BART。
- **3/7 凍結策略**：微調期間，每輪隨機僅解凍 3 個腦區編碼器，其餘 7 個保持凍結，這是一種針對「視角」的正規化（Regularization）手段。

### 超參數設定
| 參數 | 預訓練 (Pre-Training) | 微調 (Fine-Tuning) |
|---|---|---|
| Batch Size | 4 | 4 |
| 學習率 (LR) | 5e-5 | 5e-5 (S1) / 5e-6 (S2) |
| 訓練輪數 (Epochs) | 50 | 20 (S1) + 30 (S2) |
| 優化器 | AdamW | AdamW |
| 排程器 | CosineAnnealing | CosineAnnealing |

---

## 📥 設定與預處理

### 1. 下載資料
下載 ZuCo 資料集並放置於 `dataset/ZuCo/` 目錄下：
- [ZuCo v1.0](https://osf.io/q3zws/files/)
- [ZuCo v2.0](https://osf.io/2urht/files/)

### 2. 環境設定
```bash
conda env create -f environment.yml
conda activate eeg2text
pip install -r requirements.txt
```

### 3. 資料預處理
將 `.mat` 檔案轉換為包含原始 EEG 數據的 `.pickle` 格式：
```bash
scripts/prepare_dataset.sh  # Windows 請執行 .bat
```

---

## 📝 舊版腳本支持
原有的腳本（如 `train_multiview.py`, `eval_multiview.py` 等）仍保留在根目錄下且功能完整，可用於不適用統一 Pipeline 的特定實驗。

## 🔗 參考文獻
- Wang, Z. and Ji, H. (2021). *Open vocabulary EEG-to-text decoding and zero-shot sentiment classification.*
- Liu, H. et al. (2024). *EEG2TEXT: Open vocabulary EEG-to-text decoding with EEG pre-training and multi-view transformer.*
- Song, Y. et al. (2022). *EEG Conformer: Convolutional transformer for EEG decoding and visualization.*
