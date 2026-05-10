@echo off
REM Evaluate Multi-View Conformer — full BART fine-tune merged checkpoint (cont_full)
set CKPT=./checkpoints/multiview/best/task1_task2_task3_taskNRv2_taskTSRv2_multiview_2step_b4_0_15_5e-05_1e-06_unique_sent_cont_full_merged.pt
set CONF=./config/decoding/task1_task2_task3_taskNRv2_taskTSRv2_multiview_2step_b4_0_15_5e-05_1e-06_unique_sent_cont_full.json

echo [1/4] Teacher forcing + real EEG
python eval_multiview.py --checkpoint_path %CKPT% --config_path %CONF% -cuda cuda:0 -tf True -n False

echo [2/4] Teacher forcing + noise
python eval_multiview.py --checkpoint_path %CKPT% --config_path %CONF% -cuda cuda:0 -tf True -n True

echo [3/4] Free generation + real EEG (MAIN RESULT)
python eval_multiview.py --checkpoint_path %CKPT% --config_path %CONF% -cuda cuda:0 -tf False -n False

echo [4/4] Free generation + noise
python eval_multiview.py --checkpoint_path %CKPT% --config_path %CONF% -cuda cuda:0 -tf False -n True

pause
