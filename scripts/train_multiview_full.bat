@echo off
REM Continue training from a STEP-1 checkpoint with FULL BART fine-tune (no LoRA).
REM   - resume from {save_name}_s1.pt (encoder warmed-up, BART pretrained-untouched)
REM   - skip encoder_best.pt load (resume ckpt already contains encoder weights)
REM   - skip STEP 1 (encoder warm-up done in resumed ckpt)
REM   - STEP 2: --no_lora unfreezes all BART params; encoder lr = LR2*0.2, BART lr = LR2*0.1
REM   - 15 epochs, NO early stopping
REM   - output: ..._unique_sent_cont_full.pt and ..._unique_sent_cont_full_merged.pt
python train_multiview.py ^
    --model_name MultiViewConformerTranslator ^
    --task_name task1_task2_task3_taskNRv2_taskTSRv2 ^
    --two_step ^
    --pretrained ^
    --not_load_step1_checkpoint ^
    --num_epoch_step1 0 ^
    --num_epoch_step2 15 ^
    -lr1 0.00005 ^
    -lr2 0.000001 ^
    -b 4 ^
    --label_smooth 0.1 ^
    --no_early_stop ^
    --no_lora ^
    -s ./checkpoints/multiview ^
    -cuda cuda:0 ^
    --resume ./checkpoints/multiview/best/task1_task2_task3_taskNRv2_taskTSRv2_multiview_2step_b4_50_70_5e-05_5e-07_unique_sent_s1.pt
