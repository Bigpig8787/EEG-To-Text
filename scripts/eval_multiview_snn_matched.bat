@echo off
REM Evaluate the SNN-hyperparameter-matched ANN produced by
REM scripts\train_multiview_snn_matched.bat.
REM
REM eval_multiview.py reads the architecture out of %CONF%, so the geometry can
REM never drift from what was trained -- no flags to keep in sync here.
set CKPT=./checkpoints/multiview_snn_matched/best/task1_task2_task3_taskNRv2_taskTSRv2_multiview_2step_b4_50_70_5e-05_5e-06_unique_sent_snnmatch_merged.pt
set CONF=./config/decoding/task1_task2_task3_taskNRv2_taskTSRv2_multiview_2step_b4_50_70_5e-05_5e-06_unique_sent_snnmatch.json

echo [1/4] Teacher forcing + real EEG
python eval_multiview.py --checkpoint_path %CKPT% --config_path %CONF% -cuda cuda:0 -tf True -n False

echo [2/4] Teacher forcing + noise
python eval_multiview.py --checkpoint_path %CKPT% --config_path %CONF% -cuda cuda:0 -tf True -n True

echo [3/4] Free generation + real EEG (MAIN RESULT)
python eval_multiview.py --checkpoint_path %CKPT% --config_path %CONF% -cuda cuda:0 -tf False -n False

echo [4/4] Free generation + noise
python eval_multiview.py --checkpoint_path %CKPT% --config_path %CONF% -cuda cuda:0 -tf False -n True

pause
