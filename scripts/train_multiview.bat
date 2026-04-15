@echo off
REM Multi-View Conformer + LoRA: single-stage training
REM LoRA freezes BART body, so no need for Step 1/2
REM --one_step skips Step 1, goes directly to main training
python train_multiview.py ^
    --model_name MultiViewConformerTranslator ^
    --task_name task1_task2_taskNRv2 ^
    --one_step ^
    --pretrained ^
    --not_load_step1_checkpoint ^
    --num_epoch_step1 1 ^
    --num_epoch_step2 30 ^
    -lr1 0.00005 ^
    -lr2 0.000005 ^
    -b 2 ^
    -s ./checkpoints/multiview_lora ^
    -cuda cuda:0