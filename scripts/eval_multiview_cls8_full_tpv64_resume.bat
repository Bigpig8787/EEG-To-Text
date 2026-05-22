@echo off
REM Evaluate Multi-View Conformer — CLS=8, tokens_per_view=64, pool_stride=50, FULL BART fine-tune.
REM   Matches train_multiview_cls8_full_tpv64_resume.bat output.
REM   - save_name suffix _cont_full (resume + no-lora) -> ckpt ..._unique_sent_cont_full_merged.pt
REM   - full-FT _merged.pt is a plain state_dict; eval needs NO peft.
REM   - eval_multiview.py must keep tokens_per_view=64 / pool_stride=50 to match training geometry.
set CKPT=./checkpoints/multiview_cls8_lora_tpv64/best/task1_task2_task3_taskNRv2_taskTSRv2_multiview_2step_b4_50_70_5e-05_5e-07_unique_sent_cont_full_merged.pt
set CONF=./config/decoding/task1_task2_task3_taskNRv2_taskTSRv2_multiview_2step_b4_50_70_5e-05_5e-07_unique_sent_cont_full.json

echo [1/4] Teacher forcing + real EEG
python eval_multiview.py --checkpoint_path %CKPT% --config_path %CONF% -cuda cuda:0 -tf True -n False

echo [2/4] Teacher forcing + noise
python eval_multiview.py --checkpoint_path %CKPT% --config_path %CONF% -cuda cuda:0 -tf True -n True

echo [3/4] Free generation + real EEG (MAIN RESULT)
python eval_multiview.py --checkpoint_path %CKPT% --config_path %CONF% -cuda cuda:0 -tf False -n False

echo [4/4] Free generation + noise
python eval_multiview.py --checkpoint_path %CKPT% --config_path %CONF% -cuda cuda:0 -tf False -n True

pause
