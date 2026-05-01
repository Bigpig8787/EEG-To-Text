@echo off
REM Two-step training (paper-spec hyperparameters):
REM   - encoder: 4 local / 3 global layers, dropout 0.1, weight_decay 0.05
REM   - LoRA r=16, alpha=r, LR2=5e-7 (paper spec)
REM   - EEG augmentation (amplitude scale + noise + time shift) during training
REM   - label_smooth=0.1, early stopping patience=10, 50+70 epochs (doubled)
python train_multiview.py ^
    --model_name MultiViewConformerTranslator ^
    --task_name task1_task2_taskNRv2 ^
    --two_step ^
    --pretrained ^
    --not_load_step1_checkpoint ^
    --num_epoch_step1 50 ^
    --num_epoch_step2 70 ^
    -lr1 0.00005 ^
    -lr2 0.0000005 ^
    -b 4 ^
    --lora_r 16 ^
    --label_smooth 0.1 ^
    -s ./checkpoints/multiview ^
    -cuda cuda:0
