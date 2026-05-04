@echo off
REM Resume training from merged checkpoint:
REM   - skip encoder_best.pt load (merged ckpt already has encoder weights)
REM   - skip STEP 1 (encoder warm-up done in resumed ckpt)
REM   - STEP 2 only: attach FRESH LoRA, fine-tune at lower LR2 (1e-7)
REM   - output: ..._unique_sent_cont.pt and ..._unique_sent_cont_merged.pt
REM   - NO early stopping
python train_multiview.py ^
    --model_name MultiViewConformerTranslator ^
    --task_name task1_task2_taskNRv2 ^
    --two_step ^
    --pretrained ^
    --not_load_step1_checkpoint ^
    --num_epoch_step1 50 ^
    --num_epoch_step2 70 ^
    -lr1 0.00005 ^
    -lr2 0.0000001 ^
    -b 4 ^
    --lora_r 16 ^
    --label_smooth 0.1 ^
    --no_early_stop ^
    -s ./checkpoints/multiview ^
    -cuda cuda:0 ^
    --resume ./checkpoints/multiview/best/task1_task2_taskNRv2_multiview_2step_b4_50_70_5e-05_5e-07_unique_sent_merged.pt
