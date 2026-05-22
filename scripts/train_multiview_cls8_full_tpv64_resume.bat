@echo off
REM Multi-View Conformer — CLS=8, tokens_per_view=64, pool_stride=50, FULL BART fine-tune (NO LoRA).
REM   Resumes from the tpv64 LoRA run's STEP-1 checkpoint (encoder warm-up) and runs STEP 2 only.
REM
REM   - --no_lora: now WIRED to NO_LORA in train_multiview.py (was a dead flag before).
REM       NO_LORA=True -> step2 unfreezes ALL BART params, no LoRA adapters.
REM   - --resume <_s1.pt>: loads the step-1 checkpoint, SKIPS step 1, SKIPS encoder_best.pt load.
REM       The _s1.pt is the encoder-warm-up ckpt saved by train_multiview_cls8_lora_tpv64.bat.
REM   - tokens_per_view=64 / pool_stride=50 hardcoded in train_multiview.py + models/multiview.py.
REM       Resume ckpt MUST be tpv64 geometry — DO NOT resume a tpv90/tpv32 _s1.pt (shape mismatch).
REM   - save_name picks up suffix _cont (resume) + _full (no-lora) -> ..._unique_sent_cont_full
REM       Output best:   ./checkpoints/multiview_cls8_lora_tpv64/best/<save_name>_cont_full.pt
REM       Output merged: ./checkpoints/multiview_cls8_lora_tpv64/best/<save_name>_cont_full_merged.pt
REM       (full FT path: _merged.pt is a plain state_dict, no LoRA merge)
REM   - peft NOT needed for this run (no LoRA).
REM   - STEP 1 epochs ignored under --resume; kept at 50 only so save_name matches the _s1 ckpt name.
REM   - step2 full-FT param LRs: enc = LR2*0.2 = 1e-7, BART = LR2*0.1 = 5e-8 (very low, anti-overfit).
python train_multiview.py ^
    --model_name MultiViewConformerTranslator ^
    --task_name task1_task2_task3_taskNRv2_taskTSRv2 ^
    --two_step ^
    --pretrained ^
    --no_lora ^
    --resume ./checkpoints/multiview_cls8_lora_tpv64/best/task1_task2_task3_taskNRv2_taskTSRv2_multiview_2step_b4_50_70_5e-05_5e-07_unique_sent_s1.pt ^
    --num_epoch_step1 50 ^
    --num_epoch_step2 70 ^
    -lr1 0.00005 ^
    -lr2 0.0000005 ^
    -b 4 ^
    --label_smooth 0.1 ^
    --patience 10 ^
    -s ./checkpoints/multiview_cls8_lora_tpv64 ^
    -cuda cuda:0
