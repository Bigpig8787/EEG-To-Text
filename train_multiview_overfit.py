"""
Overfit Sanity Check — train MultiViewConformerTranslator on a SINGLE fixed batch
for many epochs to verify the model architecture is capable of learning.

Differences vs train_multiview.py:
  - Always uses the SAME batch every epoch (cached from first iteration)
  - No EEG augmentation, no early stopping, no LoRA / Step-2
  - Encoder + fc1 + (optionally BART) trained at constant LR
  - Logs train loss per epoch to overfit.log

Expected behavior:
  Loss should drop close to zero within ~100 epochs. If it does NOT, the
  encoder/decoder wiring is broken (gradient not flowing, shape bug, etc.).
"""

import os
import time
import json
import pickle
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import (BartConfig, BartForConditionalGeneration,
                          BartTokenizer)

from data.dataset import ZuCo_dataset
from models.multiview import MultiViewConformerTranslator

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
_LOG_PATH = 'overfit.log'


def log(msg=''):
    print(msg)
    try:
        with open(_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(str(msg) + '\n')
    except Exception:
        pass


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('-t', '--task_name', default='task1_task2_taskNRv2')
    p.add_argument('-ne', '--num_epochs', type=int, default=100)
    p.add_argument('-lr', '--learning_rate', type=float, default=5e-5)
    p.add_argument('-b', '--batch_size', type=int, default=4)
    p.add_argument('-cuda', default='cuda:0')
    p.add_argument('--unfreeze_bart', action='store_true',
                   help='also train BART params (default: BART frozen, encoder only)')
    p.add_argument('--no_pretrained_encoder', action='store_true',
                   help='skip loading checkpoints/pretrain/encoder_best.pt')
    p.add_argument('--clip', type=float, default=0.0,
                   help='grad clip max-norm (0 = disabled). default off so overfit '
                        'is not throttled by clipping.')
    p.add_argument('-s', '--save_path', default='./checkpoints/overfit')
    return vars(p.parse_args())


def main():
    args = get_args()

    with open(_LOG_PATH, 'w', encoding='utf-8') as f:
        f.write(f'=== train_multiview_overfit.py @ {time.strftime("%Y-%m-%d %H:%M:%S")} ===\n')
    for k, v in args.items():
        log(f'  {k}: {v}')

    seed = 312
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device(args['cuda'] if torch.cuda.is_available() else 'cpu')
    log(f'[INFO] device: {device}')

    # ── data ────────────────────────────────────────────────────────
    whole_dataset_dicts = []
    dd = os.path.join(PROJECT_ROOT, 'dataset', 'ZuCo')
    task_map = {
        'task1':    ('task1-SR',    'task1-SR-dataset.pickle'),
        'task2':    ('task2-NR',    'task2-NR-dataset.pickle'),
        'task3':    ('task3-TSR',   'task3-TSR-dataset.pickle'),
        'taskNRv2': ('task2-NR-2.0','task2-NR-2.0-dataset.pickle'),
    }
    for key, (task, fname) in task_map.items():
        if key in args['task_name']:
            with open(os.path.join(dd, task, 'pickle', fname), 'rb') as f:
                whole_dataset_dicts.append(pickle.load(f))

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    train_set = ZuCo_dataset(whole_dataset_dicts, 'train', tokenizer,
                             subject='ALL', eeg_type='GD',
                             bands=['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'],
                             setting='unique_sent')

    # Pick a deterministic subset: the first BATCH_SIZE indices that yield a
    # populated raw_views dict. Keeps the same sample on every run.
    chosen = []
    for i in range(len(train_set)):
        sample = train_set[i]
        raw_views = sample[-1]
        if raw_views:
            chosen.append(i)
        if len(chosen) >= args['batch_size']:
            break
    log(f'[INFO] fixed indices used for overfit: {chosen}')

    fixed_subset = Subset(train_set, chosen)
    loader = DataLoader(fixed_subset, batch_size=args['batch_size'],
                        shuffle=False, num_workers=0)

    # Materialize the one and only batch ONCE; reuse it every epoch.
    fixed_batch = next(iter(loader))
    (input_emb, seq_len, masks, mask_inv, target_ids,
     target_mask, sentiment, sent_eeg, raw_views) = fixed_batch
    log(f'[INFO] fixed batch shapes: input_emb={tuple(input_emb.shape)}, '
        f'target_ids={tuple(target_ids.shape)}, raw_views keys={list(raw_views.keys())}')

    # Move to device once (raw_views is a dict of tensors)
    target_ids_d = target_ids.to(device)
    masks_d = masks.to(device)
    mask_inv_d = mask_inv.to(device)
    raw_views_d = {k: v.to(device).float() for k, v in raw_views.items()}
    target_ids_d[target_ids_d == tokenizer.pad_token_id] = -100

    # Decode the target text so we can verify what the model is supposed to output
    label_for_print = target_ids.clone()
    label_for_print[label_for_print == -100] = tokenizer.pad_token_id
    decoded_targets = [tokenizer.decode(t, skip_special_tokens=True)
                       for t in label_for_print]
    for i, txt in enumerate(decoded_targets):
        log(f'[TARGET {i}] {txt}')

    # ── model ───────────────────────────────────────────────────────
    bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    model = MultiViewConformerTranslator(
        bart, d_model=512, n_filters=40, temporal_kernel=25,
        pool_stride=10, tokens_per_view=100, n_heads=8,
        n_encoder_layers=4, n_global_layers=3, dropout=0.1,
        decoder_embedding_size=1024,
    )

    encoder_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'pretrain', 'encoder_best.pt')
    if args['no_pretrained_encoder']:
        log('[INFO] --no_pretrained_encoder set; training encoder from scratch')
    elif os.path.exists(encoder_path):
        model.load_pretrained_encoder(encoder_path)
        log(f'[INFO] loaded pretrained encoder: {encoder_path}')
    else:
        log(f'[WARN] no pretrained encoder, training from scratch')

    if not args['unfreeze_bart']:
        for p in model.pretrained.parameters():
            p.requires_grad = False
        log('[INFO] BART frozen — training encoder + global_transformer + fc1 only')
    else:
        log('[INFO] BART unfrozen — full model trained')

    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    log(f'[INFO] trainable params: {sum(p.numel() for p in trainable):,}')

    optimizer = optim.AdamW(trainable, lr=args['learning_rate'], weight_decay=0.0)

    os.makedirs(args['save_path'], exist_ok=True)

    # ── overfit loop ────────────────────────────────────────────────
    log('\n' + '=' * 60)
    log(f'OVERFIT — single batch, {args["num_epochs"]} epochs, lr={args["learning_rate"]}')
    log('=' * 60)

    since = time.time()
    losses = []

    for epoch in range(args['num_epochs']):
        model.train()
        optimizer.zero_grad()

        # Run in fp32 (no autocast/GradScaler): single-batch overfit doesn't
        # need fp16 speed and AMP makes the loss numerically unstable.
        output = model(raw_views_d, masks_d, mask_inv_d, target_ids_d)
        loss = output.loss
        loss.backward()

        # Compute grad norm BEFORE clipping so we can see if grads are flowing.
        total_norm = 0.0
        for p in trainable:
            if p.grad is not None:
                total_norm += p.grad.data.float().norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        if args['clip'] > 0:
            nn.utils.clip_grad_norm_(trainable, max_norm=args['clip'])

        optimizer.step()

        losses.append(loss.item())
        log(f'epoch {epoch:3d}/{args["num_epochs"]-1} | loss: {loss.item():.6f} '
            f'| grad_norm: {total_norm:.4f}')

    elapsed = time.time() - since
    log(f'\nDone in {elapsed//60:.0f}m {elapsed%60:.0f}s')
    log(f'first loss: {losses[0]:.6f} | final loss: {losses[-1]:.6f} '
        f'| min loss: {min(losses):.6f}')

    # quick generation check at end
    model.eval()
    generated = model.generate(
        raw_views_d, masks_d, mask_inv_d, target_ids_d,
        max_length=56, num_beams=1,
    )
    decoded = [tokenizer.decode(g, skip_special_tokens=True) for g in generated]
    for i, txt in enumerate(decoded):
        log(f'[GEN  {i}] {txt}')

    ckpt = os.path.join(args['save_path'], 'overfit.pt')
    torch.save(model.state_dict(), ckpt)
    log(f'[INFO] saved final overfit checkpoint: {ckpt}')

    with open(os.path.join(args['save_path'], 'overfit_losses.json'), 'w') as f:
        json.dump(losses, f)


if __name__ == '__main__':
    main()
