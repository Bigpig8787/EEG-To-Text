@echo off
REM ANN side of the parameter-matched ANN-vs-SNN comparison.
REM
REM Identical to train_multiview.bat except --lora_r 8 (was 16), so BART
REM contributes exactly 2,359,296 trainable params — the same LoRA budget as the
REM matched SNN run (Spiking-EEG2TEXT\configs\multiview_snn_match_ann.json).
REM
REM Architecture is UNCHANGED (d_model=512, 4 encoder / 3 global layers,
REM tokens_per_view=64, n_cls_per_view=8): the ANN is the anchor, the SNN was
REM scaled up to meet it.
REM   ANN encoder side = 136,619,264   trainable = 138,978,560
REM   SNN encoder side = 140,107,904   trainable = 142,467,200   (+2.5%)
REM
REM --save_suffix _r8 keeps this run from overwriting the r=16 run's
REM config\decoding\<save_name>.json (lora_r is not part of save_name).
REM Save name written by train_multiview.py:
REM   task1_task2_task3_taskNRv2_taskTSRv2_multiview_2step_b4_50_70_5e-05_5e-07_unique_sent_r8

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
    --lora_r 8 ^
    --lora_targets q_proj k_proj v_proj out_proj ^
    --label_smooth 0.1 ^
    --no_early_stop ^
    --save_suffix _r8 ^
    -s ./checkpoints/multiview_lora_r8 ^
    -cuda cuda:0
