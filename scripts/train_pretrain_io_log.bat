@echo off
REM Pre-training with I/O logging.
REM Each epoch saves one (input, recon, mask) tensor to pretrain_io_samples/
REM and writes summary stats to pretrain_io.log.
python train_pretrain_io_log.py ^
    -b 4 ^
    -ne 50 ^
    -lr 0.00005 ^
    --pool_stride 10 ^
    --d_model 512 ^
    --n_transformer_layers 2 ^
    --dropout 0.2 ^
    --mask_ratio 0.30 ^
    -s ./checkpoints/pretrain_io ^
    -cuda cuda:0
