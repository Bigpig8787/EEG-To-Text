"""Merge an interrupted STEP 2 checkpoint into a plain state_dict for eval.

`train_multiview.py` wraps BART with `get_peft_model()` before STEP 2, so every
mid-loop save (`torch.save(model.state_dict(), checkpoint_path_best)`, line 187)
carries `lora_A` / `lora_B` keys. `eval_multiview.py` builds a plain
`MultiViewConformerTranslator` and calls `load_state_dict()` on it, so those
files are not loadable.

The plain `_merged.pt` that eval wants is only written after the STEP 2 epoch
loop returns (line 459/465). Ctrl+C, an OOM, or a crash anywhere inside STEP 2
means it never happens, and `best/<save_name>.pt` is stranded.

This rebuilds the model from the training config, re-wraps it with the same LoRA
configuration, loads the stranded checkpoint, calls `merge_and_unload()`, and
writes the plain checkpoint eval expects. Nothing is retrained.

The geometry and the LoRA hyper-parameters both come out of
`config/decoding/<save_name>.json`, which `train_multiview.py` writes before
training starts -- the same file `eval_multiview.py` reads -- so the rebuilt
model cannot drift from what was trained.

Usage:
    python merge_lora_ckpt.py ^
        --config_path ./config/decoding/<save_name>.json ^
        --checkpoint_path ./checkpoints/<run>/best/<save_name>.pt

Writes `<checkpoint_path>` with `.pt` replaced by `_merged.pt`, matching the
name `train_multiview.py` would have produced.
"""

import argparse
import json
import os

import torch
from transformers import BartForConditionalGeneration
from peft import LoraConfig, TaskType, get_peft_model

from models.multiview import MultiViewConformerTranslator


def build_model(config):
    """Same construction as `eval_multiview.py`, including the same fallbacks.

    The fallbacks matter: configs written before the architecture became
    configurable have no `d_model` / `n_filters` / ... keys, and the values here
    are what used to be hard-coded.
    """
    bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    return MultiViewConformerTranslator(
        bart,
        d_model=config.get('d_model', 512),
        n_filters=config.get('n_filters', 40),
        n_spatial_filters=config.get('n_spatial_filters'),
        temporal_kernel=config.get('temporal_kernel', 200),
        pool_stride=config.get('pool_stride', 50),
        tokens_per_view=config.get('tokens_per_view', 64),
        n_cls_per_view=config.get('n_cls_per_view', 8),
        n_heads=config.get('n_heads', 8),
        n_encoder_layers=config.get('n_encoder_layers', 4),
        n_global_layers=config.get('n_global_layers', 3),
        decoder_embedding_size=1024)


def looks_like_peft(state_dict):
    """True when the checkpoint still carries unmerged LoRA adapters."""
    return any('lora_' in key for key in state_dict)


def main():
    ap = argparse.ArgumentParser(
        description='Merge an interrupted STEP 2 LoRA checkpoint into a plain '
                    'state_dict that eval_multiview.py can load.')
    ap.add_argument('--config_path', required=True,
                    help='config/decoding/<save_name>.json written by train_multiview.py')
    ap.add_argument('--checkpoint_path', required=True,
                    help='the stranded STEP 2 checkpoint, e.g. best/<save_name>.pt')
    ap.add_argument('--output', default=None,
                    help='default: <checkpoint_path> with .pt -> _merged.pt, which is '
                         'exactly what train_multiview.py would have written')
    # train_multiview.py sets lora_alpha = lora_r and hard-codes dropout 0.15, so
    # the config JSON does not record either. Same defaults here; override only
    # if a run was patched to differ.
    ap.add_argument('--lora_alpha', type=int, default=None,
                    help='default: same as lora_r, matching train_multiview.py:228')
    ap.add_argument('--lora_dropout', type=float, default=0.15,
                    help='default: 0.15, matching train_multiview.py:401')
    args = ap.parse_args()

    with open(args.config_path, encoding='utf-8') as f:
        config = json.load(f)

    if config.get('no_lora', False):
        raise SystemExit(
            'This run was a full fine-tune (--no_lora), so its checkpoints are '
            'already plain state_dicts. Point eval at the checkpoint directly, '
            'or copy it to the _merged.pt name eval expects.')

    lora_r = config.get('lora_r', 16)
    lora_alpha = args.lora_alpha if args.lora_alpha is not None else lora_r
    lora_targets = config.get('lora_targets', ['q_proj', 'k_proj', 'v_proj', 'out_proj'])

    print(f'[INFO] config     : {args.config_path}')
    print(f'[INFO] checkpoint : {args.checkpoint_path}')
    print(f'[INFO] lora       : r={lora_r} alpha={lora_alpha} '
          f'dropout={args.lora_dropout} targets={lora_targets}')

    state_dict = torch.load(args.checkpoint_path, map_location='cpu')
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    if not looks_like_peft(state_dict):
        raise SystemExit(
            f'{args.checkpoint_path} has no lora_* keys — it is already a plain '
            'state_dict and needs no merging. If eval cannot load it, the problem '
            'is elsewhere (wrong architecture in the config, or a STEP 1 _s1.pt '
            'checkpoint, which has no BART weights to evaluate).')

    print('[INFO] building model from the training config...')
    model = build_model(config)

    print('[INFO] re-wrapping BART with the same LoRA configuration...')
    model.pretrained = get_peft_model(model.pretrained, LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=lora_targets,
    ))

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f'[INFO] load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected')
    if unexpected:
        # Unexpected keys mean the rebuilt model does not match what was saved,
        # which would silently produce a checkpoint with random weights in those
        # places. Refuse rather than write it.
        raise SystemExit(
            f'{len(unexpected)} unexpected keys — the rebuilt model does not match '
            f'the checkpoint. First 5: {unexpected[:5]}\n'
            'The config and the checkpoint are probably from different runs.')
    if missing:
        # PEFT leaves the frozen base weights in place, so a handful of missing
        # keys is normal only if they are LoRA bookkeeping. Anything else is not.
        real_missing = [k for k in missing if 'lora_' not in k]
        if real_missing:
            raise SystemExit(
                f'{len(real_missing)} weights are absent from the checkpoint, e.g. '
                f'{real_missing[:5]}\nMerging would fill them with random init.')

    print('[INFO] merge_and_unload() — collapsing LoRA into BART...')
    model.pretrained = model.pretrained.merge_and_unload()

    out = args.output or args.checkpoint_path.replace('.pt', '_merged.pt')
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    torch.save(model.state_dict(), out)
    print(f'[OK] merged checkpoint saved: {out}')
    print('[OK] eval_multiview.py can load this directly.')


if __name__ == '__main__':
    main()
