@echo off
REM CLIP-style EEG↔Text contrastive pre-training
REM   - loads existing encoder_best.pt and continues with InfoNCE
REM   - frozen BART-large text encoder produces sentence targets
REM   - large batch (32) is critical for in-batch negatives
REM   - output: encoder_clip_best.pt (drop-in replacement for encoder_best.pt)
python train_pretrain_clip.py ^
    -b 32 ^
    -ne 30 ^
    -lr 0.00005 ^
    --temporal_kernel 200 ^
    --pool_stride 100 ^
    --d_model 512 ^
    --n_transformer_layers 2 ^
    --dropout 0.2 ^
    --mask_ratio 0.15 ^
    --temperature 0.07 ^
    --proj_dim 1024 ^
    --max_text_len 64 ^
    --text_model facebook/bart-large ^
    --encoder_init_path ./checkpoints/pretrain/encoder_best.pt ^
    -s ./checkpoints/pretrain ^
    -cuda cuda:0
