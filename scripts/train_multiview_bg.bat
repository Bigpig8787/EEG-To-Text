@echo off
REM ============================================================
REM  Background version of train_multiview.bat
REM  - Opens a detached cmd window that survives closing this one
REM  - All output goes to train.log (overwrites each run)
REM    Change  >  to  >>  below if you want to append instead.
REM  - Check progress with:  type train.log
REM    or follow live in PowerShell:  Get-Content train.log -Wait -Tail 50
REM ============================================================

REM Move to project root (this .bat lives in scripts\)
pushd "%~dp0\.."

echo Starting training in a new window. Log: %CD%\train.log

start "MultiView-Train" cmd /k "python train_multiview.py --model_name MultiViewConformerTranslator --task_name task1_task2_taskNRv2 --two_step --pretrained --not_load_step1_checkpoint --num_epoch_step1 50 --num_epoch_step2 70 -lr1 0.00005 -lr2 0.0000005 -b 4 --no_early_stop --lora_r 16 --label_smooth 0.1 -s ./checkpoints/multiview -cuda cuda:0 > train.log 2>&1"

popd
echo Done launching. You can close THIS window now.
