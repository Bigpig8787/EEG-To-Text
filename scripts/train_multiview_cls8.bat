@echo off
REM Multi-View Conformer — CLS=8 per view (was 4), from-scratch full BART fine-tune.
REM   - n_cls_per_view=8 hardcoded in train_multiview.py / eval_multiview.py
REM   - CANNOT resume from cls=4 ckpt: cls_token + view_pos_embed shapes differ
REM   - Separate save dir to avoid collision with cls=4 baseline
REM   - 50 + 70 epochs (paper spec), label_smooth=0.1, no early stop
REM   - NO_LORA forced True in train_multiview.py -> full BART fine-tune
REM   - Output: ./checkpoints/multiview_cls8/best/<save_name>_full_merged.pt
python train_multiview.py ^
    --model_name MultiViewConformerTranslator ^
    --task_name task1_task2_task3_taskNRv2_taskTSRv2 ^
    --two_step ^
    --pretrained ^
    --not_load_step1_checkpoint ^
    --num_epoch_step1 50 ^
    --num_epoch_step2 70 ^
    -lr1 0.00005 ^
    -lr2 0.0000005 ^
    -b 4 ^
    --label_smooth 0.1 ^
    --no_early_stop ^
    --no_lora ^
    -s ./checkpoints/multiview_cls8 ^
    -cuda cuda:0
