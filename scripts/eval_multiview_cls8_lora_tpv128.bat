@echo off
REM Evaluate Multi-View Conformer — CLS=8, tokens_per_view=128, LoRA r=16 merged checkpoint.
REM   LoRA path -> save_name has NO _full suffix: ..._unique_sent_merged.pt
REM   -s was ./checkpoints/multiview_cls8_lora_tpv128 ; config auto-written to ./config/decoding/<save_name>.json
REM   eval needs NO peft (LoRA already merged into plain state_dict at train end)
set CKPT=./checkpoints/multiview_cls8_lora_tpv128/best/task1_task2_task3_taskNRv2_taskTSRv2_multiview_2step_b4_50_70_5e-05_5e-07_unique_sent_merged.pt
set CONF=./config/decoding/task1_task2_task3_taskNRv2_taskTSRv2_multiview_2step_b4_50_70_5e-05_5e-07_unique_sent.json

echo [1/4] Teacher forcing + real EEG
python eval_multiview.py --checkpoint_path %CKPT% --config_path %CONF% -cuda cuda:0 -tf True -n False

echo [2/4] Teacher forcing + noise
python eval_multiview.py --checkpoint_path %CKPT% --config_path %CONF% -cuda cuda:0 -tf True -n True

echo [3/4] Free generation + real EEG (MAIN RESULT)
python eval_multiview.py --checkpoint_path %CKPT% --config_path %CONF% -cuda cuda:0 -tf False -n False

echo [4/4] Free generation + noise
python eval_multiview.py --checkpoint_path %CKPT% --config_path %CONF% -cuda cuda:0 -tf False -n True

pause
