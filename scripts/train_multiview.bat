@echo off
REM ============================================================
REM  Two-step training (paper-spec hyperparameters):
REM    - encoder: 4 local / 3 global layers, dropout 0.1, weight_decay 0.05
REM    - LoRA r=16, alpha=r, LR2=5e-7 (paper spec)
REM    - EEG augmentation (amplitude scale + noise + time shift) during training
REM    - label_smooth=0.1, no early stopping, 50+70 epochs (doubled)
REM  Runs in the CURRENT terminal.
REM    - Live output shown in this window
REM    - Simultaneously written to train.log (overwritten each run)
REM    - Closing THIS terminal WILL stop training.
REM ============================================================

pushd "%~dp0\.."

REM Clear the log first so each run starts fresh.
type nul > train.log

powershell -NoProfile -Command "python train_multiview.py --model_name MultiViewConformerTranslator --task_name task1_task2_taskNRv2 --two_step --pretrained --not_load_step1_checkpoint --num_epoch_step1 50 --num_epoch_step2 70 -lr1 0.00005 -lr2 0.0000005 -b 4 --no_early_stop --lora_r 16 --label_smooth 0.1 -s ./checkpoints/multiview -cuda cuda:0 2>&1 | Tee-Object -FilePath train.log"

popd
