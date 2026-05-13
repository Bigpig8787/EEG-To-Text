@echo off
REM Step2 FULL BART fine-tune, resume from previous merged checkpoint.
REM   - 4 CLS tokens per view (default in MultiViewConformerEncoder)
REM   - NO_LORA forced True in train_multiview.py -> full BART fine-tune
REM   - Resume skips STEP 1 (encoder + BART weights loaded from merged ckpt)
REM   - Early stopping enabled: patience=10 (stop if val no improve 10 epoch)
REM   - LR2 = 5e-7 (paper spec)
REM   - Output: *_cont_full.pt and *_cont_full_merged.pt
python train_multiview.py ^
    --model_name MultiViewConformerTranslator ^
    --task_name task1_task2_task3_taskNRv2_taskTSRv2 ^
    --two_step ^
    --pretrained ^
    --not_load_step1_checkpoint ^
    --num_epoch_step1 0 ^
    --num_epoch_step2 70 ^
    -lr1 0.00005 ^
    -lr2 0.0000005 ^
    -b 4 ^
    --label_smooth 0.1 ^
    -patience 10 ^
    --no_lora ^
    -s ./checkpoints/multiview ^
    -cuda cuda:0 ^
    --resume ./checkpoints/multiview/best/task1_task2_task3_taskNRv2_taskTSRv2_multiview_2step_b4_50_70_5e-05_5e-07_unique_sent_merged.pt
