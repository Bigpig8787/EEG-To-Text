#!/usr/bin/env bash
# ANN 對照組：SNN 1T1S grid 最佳格（d512 / kernel_count 256）的幾何，local transformer 4 層。
#
#   SNN 最佳格                         ANN 這裡
#   local_dim [512, 256, 512]          --n_filters 512 --n_spatial_filters 256 --d_model 512
#   local_kernel_size 100              --temporal_kernel 100
#   local_pool_kernel_size 40          --pool_stride 40
#   local_attn_length 32               --tokens_per_view 32
#   n_class_per_view 4                 --n_cls_per_view 4
#   local_num_heads 8                  --n_heads 8
#   local_num_attn_layers 2            --n_encoder_layers 4   ← 使用者指定 4 層（SNN 是 2）
#   global_num_attn_layers 2           --n_global_layers 2
#   200Hz, raw_eeg_max_len 2000        --pickle_suffix -dataset-200hz.pickle --raw_eeg_max_len 2000
#   pretrain 50 epoch, no early stop   -ne 50（ANN pretrain 本來就沒有 early stop）
#   STEP 1 40 epoch lr 5e-5 pat 10     -ne1 40 -lr1 5e-5 -patience 10
#   STEP 2 LoRA r8 60 epoch            -ne2 60 --lora_r 8 --lr_lora 1e-4
#     encoder lr = 1e-4 * 0.01 = 1e-6  -lr2 5e-6（ANN encoder lr = lr2 * 0.2 = 1e-6）
#   STEP 3 擴 T                        ANN 沒有時間步，不適用
#
# 用法（在 EEG-To-Text 根目錄）：
#   nohup bash scripts/run_paper_best_200hz.sh cuda:1 > logs/ann_paper_best_200hz.out 2>&1 &
# local transformer 層數（pretrain 與 finetune 一起改）：參數 2，預設 4。不是 4 時
# TAG / checkpoint 目錄自動加 _local<N>，不會蓋掉 4 層那次：
#   nohup bash scripts/run_paper_best_200hz.sh cuda:1 2 > logs/ann_paper_best_200hz_local2.out 2>&1 &
# 已完成的階段會跳過（看 encoder_best.pt / *_merged.pt / *_metrics.json 是否存在）。

set -euo pipefail

CUDA=${1:-cuda:1}
LOCAL_LAYERS=${2:-4}
DATA_ROOT=${DATA_ROOT:-/home/jovyan/datasets/soc-20260831131547}
SUFFIX=-dataset-200hz.pickle
MAX_LEN=2000
TASK=task1_task2_task3_taskNRv2_taskTSRv2
TAG=_paperbest200hz
PRE_DIR=./checkpoints/pretrain_paper_best_200hz
FT_DIR=./checkpoints/paper_best_200hz
if [[ $LOCAL_LAYERS != 4 ]]; then
    TAG+=_local$LOCAL_LAYERS
    PRE_DIR+=_local$LOCAL_LAYERS
    FT_DIR+=_local$LOCAL_LAYERS
fi

GEOM=(--d_model 512 --n_filters 512 --n_spatial_filters 256 --temporal_kernel 100
      --pool_stride 40 --n_heads 8)

mkdir -p logs
echo "[$(date '+%F %T')] ANN paper-best 200Hz on $CUDA, local layers $LOCAL_LAYERS, data $DATA_ROOT"

# ── 1. MAE pretrain ──────────────────────────────────────────────
if [[ -f $PRE_DIR/encoder_best.pt ]]; then
    echo "[skip] pretrain: $PRE_DIR/encoder_best.pt exists"
else
    python -u train_pretrain.py -cuda "$CUDA" -ne 50 -lr 5e-5 -b 4 \
        "${GEOM[@]}" --n_transformer_layers $LOCAL_LAYERS --mask_ratio 0.15 \
        --dataset_root "$DATA_ROOT" --pickle_suffix="$SUFFIX" --raw_eeg_max_len $MAX_LEN \
        -s "$PRE_DIR"
fi

# ── 2. STEP 1（凍 BART）+ STEP 2（LoRA）────────────────────────────
merged=$(ls $FT_DIR/best/*${TAG}_merged.pt 2>/dev/null | head -1 || true)
if [[ -n $merged ]]; then
    echo "[skip] finetune: $merged exists"
else
    python -u train_multiview.py -m MultiViewConformerTranslator -t $TASK \
        -2step -pre -ne1 40 -ne2 60 -lr1 0.00005 -lr2 0.000005 -b 4 \
        -patience 10 --lora_r 8 --lr_lora 0.0001 \
        "${GEOM[@]}" --tokens_per_view 32 --n_cls_per_view 4 \
        --n_encoder_layers $LOCAL_LAYERS --n_global_layers 2 \
        --dataset_root "$DATA_ROOT" --pickle_suffix="$SUFFIX" --raw_eeg_max_len $MAX_LEN \
        --pretrained_encoder "$PRE_DIR/encoder_best.pt" \
        --save_suffix $TAG -s "$FT_DIR" -cuda "$CUDA"
    merged=$(ls $FT_DIR/best/*${TAG}_merged.pt | head -1)
fi
conf=config/decoding/$(basename "${merged%_merged.pt}").json
[[ -f $conf ]] || { echo "missing $conf"; exit 1; }

# ── 3. eval：tf × noise 四組 ─────────────────────────────────────
for tf in False True; do
    for noise in False True; do
        name="${TASK}-multiview${TAG}"
        [[ $tf == True ]] && name+=_tf
        [[ $noise == True ]] && name+=_noise
        if [[ -f results/${name}_metrics.json ]]; then
            echo "[skip] eval $name"; continue
        fi
        python -u eval_multiview.py -checkpoint "$merged" -conf "$conf" \
            -cuda "$CUDA" -tf $tf -n $noise
    done
done
echo "[$(date '+%F %T')] done. results/${TASK}-multiview${TAG}*_metrics.json"
