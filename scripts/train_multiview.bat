@echo off
REM Multi-View Conformer + LoRA: single-stage training
python train_multiview.py ^
    --model_name MultiViewConformerTranslator ^
    --task_name task1_task2_taskNRv2 ^
    --one_step ^
    --pretrained ^
    --not_load_step1_checkpoint ^
    --num_epoch_step1 1 ^
    --num_epoch_step2 50 ^
    -lr1 0.000005 ^
    -lr2 0.000005 ^
    -b 2 ^
    -s ./checkpoints/multiview_lora ^
    -cuda cuda:0