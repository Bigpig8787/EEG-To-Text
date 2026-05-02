@echo off
REM Pre-training: Conformer + Re-Masked Token Prediction
REM   Paper-spec: pool_stride=100, temporal_kernel=200, mask_ratio=0.15
REM   dropout=0.2, n_transformer_layers=2 (match fine-tune architecture)
python train_pretrain.py ^
    -b 4 ^
    -ne 50 ^
    -lr 0.00005 ^
    --temporal_kernel 200 ^
    --pool_stride 100 ^
    --d_model 512 ^
    --n_transformer_layers 2 ^
    --dropout 0.2 ^
    --mask_ratio 0.15 ^
    -s ./checkpoints/pretrain ^
    -cuda cuda:0
