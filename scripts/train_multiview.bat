@echo off
REM Two-step training (overfitting fixes applied):
REM   - encoder: 2 layers, dropout 0.3, weight_decay 0.05
REM   - LoRA r=16, alpha=32, LR = LR2*2.0 (was 0.1 — that was the main bug)
REM   - EEG augmentation (amplitude scale + noise + time shift) during training
REM   - label_smooth=0.1, no early stopping, 25+35 epochs
python train_multiview.py ^
    --model_name MultiViewConformerTranslator ^
    --task_name task1_task2_taskNRv2 ^
    --two_step ^
    --pretrained ^
    --not_load_step1_checkpoint ^
    --num_epoch_step1 25 ^
    --num_epoch_step2 35 ^
    -lr1 0.00005 ^
    -lr2 0.000005 ^
    -b 4 ^
    --no_early_stop ^
    --lora_r 16 ^
    --label_smooth 0.1 ^
    -s ./checkpoints/multiview ^
    -cuda cuda:0
