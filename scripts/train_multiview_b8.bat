@echo off
REM Two-step training — same as train_multiview.bat but batch size 4 -> 8.
REM Watch GPU memory; if OOM, lower grad_accum_steps in train_multiview.py
REM or reduce -b back to 4.
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
    -b 8 ^
    --lora_r 16 ^
    --label_smooth 0.1 ^
    -s ./checkpoints/multiview_b8 ^
    -cuda cuda:0
