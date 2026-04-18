@echo off
REM Pre-training: Conformer + Re-Masked Token Prediction
REM   mask_ratio=0.30 (was 0.15 -- higher mask = more robust representations)
REM   dropout=0.2, n_transformer_layers=2 (match fine-tune architecture)
python train_pretrain.py ^
    -b 4 ^
    -ne 50 ^
    -lr 0.00005 ^
    --pool_stride 10 ^
    --d_model 512 ^
    --n_transformer_layers 2 ^
    --dropout 0.2 ^
    --mask_ratio 0.30 ^
    -s ./checkpoints/pretrain ^
    -cuda cuda:0
