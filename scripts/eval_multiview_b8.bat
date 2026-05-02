@echo off
REM Evaluate Multi-View Conformer trained with batch size 8.
set CKPT=./checkpoints/multiview_b8/best/task1_task2_taskNRv2_multiview_2step_b8_50_70_5e-05_5e-07_unique_sent_merged.pt
set CONF=./config/decoding/task1_task2_taskNRv2_multiview_2step_b8_50_70_5e-05_5e-07_unique_sent.json

echo [1/4] Teacher forcing + real EEG
python eval_multiview.py --checkpoint_path %CKPT% --config_path %CONF% -cuda cuda:0 -tf True -n False

echo [2/4] Teacher forcing + noise
python eval_multiview.py --checkpoint_path %CKPT% --config_path %CONF% -cuda cuda:0 -tf True -n True

echo [3/4] Free generation + real EEG (MAIN RESULT)
python eval_multiview.py --checkpoint_path %CKPT% --config_path %CONF% -cuda cuda:0 -tf False -n False

echo [4/4] Free generation + noise
python eval_multiview.py --checkpoint_path %CKPT% --config_path %CONF% -cuda cuda:0 -tf False -n True

pause
