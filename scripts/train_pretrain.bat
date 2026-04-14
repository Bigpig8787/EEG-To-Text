@echo off
REM Pre-training: Conformer + Re-Masked Token Prediction
python train_pretrain.py -b 4 -ne 50 -lr 0.00005 --pool_stride 10 --d_model 512 -s ./checkpoints/pretrain -cuda cuda:0
