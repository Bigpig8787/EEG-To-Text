@echo off
REM Two-step training: encoder warm-up then LoRA fine-tune (no early stopping, expanded LoRA, label smoothing)
python train_multiview.py ^
    --model_name MultiViewConformerTranslator ^
    --task_name task1_task2_taskNRv2 ^
    --two_step ^
    --pretrained ^
    --not_load_step1_checkpoint ^
    --num_epoch_step1 20 ^
    --num_epoch_step2 30 ^
    -lr1 0.00005 ^
    -lr2 0.000005 ^
    -b 4 ^
    --no_early_stop ^
    --lora_r 32 ^
    --label_smooth 0.1 ^
    -s ./checkpoints/multiview ^
    -cuda cuda:0
