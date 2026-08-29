@echo off
REM Rescue an interrupted STEP 2 run of scripts\train_multiview_snn_matched.bat.
REM
REM train_multiview.py only writes the plain _merged.pt after the STEP 2 epoch
REM loop finishes (line 459/465). If STEP 2 was interrupted, best\<save_name>.pt
REM exists but still carries LoRA adapters, and eval_multiview.py cannot load it.
REM
REM This merges that checkpoint into the _merged.pt name eval expects. It does
REM not retrain anything -- the result is only as good as the epoch that
REM produced the best checkpoint, so check train.log for how many STEP 2 epochs
REM actually ran before trusting the numbers.

setlocal
pushd "%~dp0\.."

set RUN=task1_task2_task3_taskNRv2_taskTSRv2_multiview_2step_b4_50_70_5e-05_5e-06_unique_sent_snnmatch

python merge_lora_ckpt.py ^
    --config_path ./config/decoding/%RUN%.json ^
    --checkpoint_path ./checkpoints/multiview_snn_matched/best/%RUN%.pt

popd
endlocal
