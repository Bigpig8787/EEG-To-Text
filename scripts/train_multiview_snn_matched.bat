@echo off
REM ANN run with every hyperparameter aligned to the SNN t=4 baseline
REM (Spiking-EEG2TEXT\configs\multiview_snn_v2.json).
REM
REM The SNN is the anchor here: it is not touched at all. The ANN is scaled DOWN
REM to it, so architecture AND sequence length are controlled at the same time.
REM
REM   SNN t=4 knob                 value   ANN flag
REM   local_dim[0]  t_conv out        64   --n_filters 64
REM   local_dim[1]  s_conv out        32   --n_spatial_filters 32
REM   local_dim[2]  attn dim         256   --d_model 256
REM   local_kernel_size (default)    250   --temporal_kernel 250
REM   local_pool_kernel_size         100   --pool_stride 100
REM   local_attn_length               32   --tokens_per_view 32
REM   n_class_per_view                 4   --n_cls_per_view 4
REM   local_num_attn_layers            2   --n_encoder_layers 2
REM   global_num_attn_layers           2   --n_global_layers 2
REM   local/global_num_heads           8   --n_heads 8
REM   global_out_dim                1024   decoder_embedding_size (fixed)
REM   learning_rate_step2           5e-6   -lr2 0.000005
REM   lora.lr_lora                  1e-4   --lr_lora 0.0001
REM   lora.lora_r                      8   --lora_r 8
REM
REM Already identical before this run, no flag needed: batch_size 4,
REM grad_accum_steps 2, 50+70 epochs, lr1 5e-5, warmup_ratio 0.2,
REM weight_decay 0.05, clip_norm 1.0, label_smooth 0.1, patience 10 /
REM no_early_stop, seed 312, lora_alpha=lora_r, lora_dropout 0.15,
REM lora_targets q/k/v/out_proj, FFN expansion 4x.
REM
REM Measured encoder-side parameters (BART excluded, identical on both sides):
REM   ANN matched   18,119,808
REM   SNN t=4       19,158,656      ANN is 5.4%% smaller
REM BART input sequence: 10*(4+32) = 360 tokens on both sides.
REM
REM The residual 5.4%% is structural, not a hyperparameter gap: the SNN carries a
REM learnable positional_embedding and an extra before_transformer_linear per
REM region that the ANN has no counterpart for (its PositionalEncoding is
REM sinusoidal and parameter-free, and its projection goes straight to d_model).
REM
REM WARNING: checkpoints\pretrain\encoder_best.pt was trained at the old geometry
REM (d_model 512, 40 filters, kernel 200, 4 layers) and will NOT load into this
REM model. load_pretrained_encoder skips every mismatched tensor and prints a
REM warning, so STEP 1 starts from random init. Re-run ANN MAE pre-training at
REM this geometry first if a warm start is wanted.

python train_multiview.py ^
    --model_name MultiViewConformerTranslator ^
    --task_name task1_task2_task3_taskNRv2_taskTSRv2 ^
    --two_step ^
    --pretrained ^
    --not_load_step1_checkpoint ^
    --d_model 256 ^
    --n_filters 64 ^
    --n_spatial_filters 32 ^
    --temporal_kernel 250 ^
    --pool_stride 100 ^
    --tokens_per_view 32 ^
    --n_cls_per_view 4 ^
    --n_heads 8 ^
    --n_encoder_layers 2 ^
    --n_global_layers 2 ^
    --dropout 0.1 ^
    --num_epoch_step1 50 ^
    --num_epoch_step2 70 ^
    -lr1 0.00005 ^
    -lr2 0.000005 ^
    --lr_lora 0.0001 ^
    -b 4 ^
    --lora_r 8 ^
    --lora_targets q_proj k_proj v_proj out_proj ^
    --label_smooth 0.1 ^
    --no_early_stop ^
    --save_suffix _snnmatch ^
    -s ./checkpoints/multiview_snn_matched ^
    -cuda cuda:0
