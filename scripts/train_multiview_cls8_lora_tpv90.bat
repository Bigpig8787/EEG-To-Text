@echo off
REM Multi-View Conformer — CLS=8 per view, tokens_per_view=90, pool_stride=25, LoRA r=16.
REM   - tokens_per_view=90 / pool_stride=25 hardcoded in train_multiview.py + eval_multiview.py
REM       AvgPool 5000 -> 200, AdaptiveAvgPool 200 -> 90
REM   - WHY 90 not 128: BART-large encoder pos-emb max = 1024. Combined seq = V*(k+T),
REM       V=10 regions, k=8 cls. 10*(8+90)=980 <=1024 OK. 128 -> 1360 > 1024 = CUDA OOB crash.
REM   - LoRA ENABLED: NO_LORA=False in train_multiview.py (r=16, q/k/v/out_proj, alpha=r)
REM       step2 wraps BART in get_peft_model, merge_and_unload -> plain _merged.pt
REM   - NO --no_lora flag (ignored anyway; NO_LORA hardcoded False)
REM   - peft MUST be installed in this env: pip install peft
REM   - CANNOT resume from old tokens_per_view=32 ckpt: token geometry differs
REM   - Separate save dir to avoid collision with full-FT cls8 baseline
REM   - 50 + 70 epochs, label_smooth=0.1, no early stop, batch 4 (lower if OOM)
REM   - save_name has NO _full suffix (LoRA path): ..._unique_sent_merged.pt
REM   - Output: ./checkpoints/multiview_cls8_lora_tpv90/best/<save_name>_merged.pt
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
    --label_smooth 0.1 ^
    --no_early_stop ^
    --lora_r 16 ^
    -s ./checkpoints/multiview_cls8_lora_tpv90 ^
    -cuda cuda:0
