"""CLIP-style contrastive pre-training: align EEG encoder with BART text encoder.

Loads an existing pre-trained encoder checkpoint (encoder_best.pt) and continues
training with a symmetric InfoNCE loss between EEG embeddings and BART
sentence embeddings. The text encoder (BART-large) is frozen.

Output: encoder_clip_best.pt (state dict of ConformerEncoder, drop-in
replacement for encoder_best.pt in train_multiview.py).
"""

import os
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import BartTokenizer, BartModel

from models.pretrain_clip_model import ConformerCLIPModel
from models.pretrain_model import create_remask
from data.pretrain_clip_dataset import EEGPretrainCLIPDataset, RAW_EEG_MAX_LEN

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ne', '--num_epochs', type=int, default=30)
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-5)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-cuda', '--cuda', default='cuda:0')
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_filters', type=int, default=40)
    parser.add_argument('--temporal_kernel', type=int, default=200)
    parser.add_argument('--pool_stride', type=int, default=100)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_transformer_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--mask_ratio', type=float, default=0.15,
                        help='Re-Masked augmentation on EEG input (0 to disable)')
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--proj_dim', type=int, default=1024,
                        help='matches BART-large hidden size')
    parser.add_argument('--text_model', default='facebook/bart-large')
    parser.add_argument('--max_text_len', type=int, default=64)
    parser.add_argument('--encoder_init_path', default='./checkpoints/pretrain/encoder_best.pt',
                        help='path to existing encoder weights to continue from')
    parser.add_argument('-s', '--save_path', default='./checkpoints/pretrain')
    parser.add_argument('--strict_load', action='store_true',
                        help='strict state_dict match for encoder init')
    return vars(parser.parse_args())


def make_collate(tokenizer, max_text_len):
    def collate(batch):
        eegs, lens, texts = zip(*batch)
        eeg = torch.stack(eegs, dim=0)
        lens = torch.tensor(lens, dtype=torch.long)
        tok = tokenizer(list(texts), padding=True, truncation=True,
                        max_length=max_text_len, return_tensors='pt')
        return eeg, lens, tok['input_ids'], tok['attention_mask']
    return collate


def text_pool(hidden, attention_mask):
    # Mean-pool BART encoder hidden states over non-pad tokens.
    mask = attention_mask.unsqueeze(-1).type_as(hidden)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


def info_nce(z_eeg, z_txt, temperature):
    # both already L2-normalised
    logits = z_eeg @ z_txt.t() / temperature
    targets = torch.arange(z_eeg.size(0), device=z_eeg.device)
    loss_e = F.cross_entropy(logits, targets)
    loss_t = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_e + loss_t), logits


def accuracy_at_k(logits, k=1):
    targets = torch.arange(logits.size(0), device=logits.device)
    topk = logits.topk(k, dim=1).indices
    return (topk == targets.unsqueeze(1)).any(dim=1).float().mean().item()


def main():
    args = get_args()
    print('=' * 60)
    print('EEG↔Text Contrastive Pre-Training (InfoNCE)')
    print('=' * 60)
    for k, v in args.items():
        print(f'  {k}: {v}')
    print('=' * 60)

    device = torch.device(args['cuda'] if torch.cuda.is_available() else 'cpu')
    os.makedirs(args['save_path'], exist_ok=True)
    np.random.seed(312)
    torch.manual_seed(312)

    # ── data ─────────────────────────────────────────────────────────
    pickle_paths = []
    dataset_dir = os.path.join(PROJECT_ROOT, 'dataset', 'ZuCo')
    for task, fname in [('task1-SR', 'task1-SR-dataset.pickle'),
                        ('task2-NR', 'task2-NR-dataset.pickle'),
                        ('task3-TSR', 'task3-TSR-dataset.pickle'),
                        ('task2-NR-2.0', 'task2-NR-2.0-dataset.pickle'),
                        ('task2-TSR-2.0', 'task2-TSR-2.0-dataset.pickle')]:
        p = os.path.join(dataset_dir, task, 'pickle', fname)
        if os.path.isfile(p):
            pickle_paths.append(p)
        else:
            print(f'[WARN] missing pickle: {p}')

    train_ds = EEGPretrainCLIPDataset(pickle_paths, split='train')
    val_ds = EEGPretrainCLIPDataset(pickle_paths, split='dev')

    tokenizer = BartTokenizer.from_pretrained(args['text_model'])
    collate = make_collate(tokenizer, args['max_text_len'])

    train_loader = DataLoader(train_ds, batch_size=args['batch_size'], shuffle=True,
                              num_workers=0, drop_last=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args['batch_size'], shuffle=False,
                            num_workers=0, drop_last=True, collate_fn=collate)

    # ── text encoder (frozen) ────────────────────────────────────────
    bart = BartModel.from_pretrained(args['text_model'])
    text_encoder = bart.get_encoder().to(device)
    for p in text_encoder.parameters():
        p.requires_grad = False
    text_encoder.eval()

    # ── EEG model ────────────────────────────────────────────────────
    model = ConformerCLIPModel(
        n_channels=105, d_model=args['d_model'], n_filters=args['n_filters'],
        temporal_kernel=args['temporal_kernel'], pool_stride=args['pool_stride'],
        n_heads=args['n_heads'], n_transformer_layers=args['n_transformer_layers'],
        dropout=args['dropout'], proj_dim=args['proj_dim'],
    ).to(device)

    if os.path.isfile(args['encoder_init_path']):
        missing, unexpected = model.load_encoder_weights(
            args['encoder_init_path'], strict=args['strict_load'])
        print(f'[INFO] loaded encoder init: {args["encoder_init_path"]}')
        print(f'  missing={len(missing)} unexpected={len(unexpected)}')
    else:
        print(f'[WARN] no encoder init at {args["encoder_init_path"]}; training from scratch')

    print(f'[INFO] EEG model params: {sum(p.numel() for p in model.parameters()):,}')
    print(f'[INFO] EEG model trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

    optimizer = optim.AdamW(model.parameters(), lr=args['learning_rate'], weight_decay=0.05)
    scaler = torch.cuda.amp.GradScaler()

    best_val_loss = float('inf')

    for epoch in range(args['num_epochs']):
        # ── train ────────────────────────────────────────────────────
        model.train()
        total_loss, total_acc1, n = 0.0, 0.0, 0

        pbar = tqdm(train_loader, desc=f'train e{epoch}')
        for eeg, _, input_ids, attn in pbar:
            eeg = eeg.to(device, non_blocking=True)
            input_ids = input_ids.to(device, non_blocking=True)
            attn = attn.to(device, non_blocking=True)

            if args['mask_ratio'] > 0:
                eeg_in, mask = create_remask(eeg, mask_ratio=args['mask_ratio'])
            else:
                eeg_in, mask = eeg, None

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                z_eeg = model(eeg_in, mask=mask)
                with torch.no_grad():
                    txt_h = text_encoder(input_ids=input_ids, attention_mask=attn).last_hidden_state
                    z_txt = F.normalize(text_pool(txt_h, attn), dim=-1)
                loss, logits = info_nce(z_eeg, z_txt, args['temperature'])

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            bs = eeg.size(0)
            total_loss += loss.item() * bs
            total_acc1 += accuracy_at_k(logits.detach(), 1) * bs
            n += bs
            pbar.set_postfix({'loss': f'{total_loss/n:.4f}',
                              'acc@1': f'{total_acc1/n:.3f}'})

        train_loss = total_loss / n
        train_acc1 = total_acc1 / n

        # ── val ──────────────────────────────────────────────────────
        model.eval()
        v_loss, v_acc1, vn = 0.0, 0.0, 0
        with torch.no_grad():
            for eeg, _, input_ids, attn in val_loader:
                eeg = eeg.to(device); input_ids = input_ids.to(device); attn = attn.to(device)
                with torch.cuda.amp.autocast():
                    z_eeg = model(eeg, mask=None)
                    txt_h = text_encoder(input_ids=input_ids, attention_mask=attn).last_hidden_state
                    z_txt = F.normalize(text_pool(txt_h, attn), dim=-1)
                    loss, logits = info_nce(z_eeg, z_txt, args['temperature'])
                bs = eeg.size(0)
                v_loss += loss.item() * bs
                v_acc1 += accuracy_at_k(logits, 1) * bs
                vn += bs
        val_loss = v_loss / max(vn, 1)
        val_acc1 = v_acc1 / max(vn, 1)

        print(f'Epoch {epoch}/{args["num_epochs"]-1} | '
              f'train: {train_loss:.4f} acc@1={train_acc1:.3f} | '
              f'val: {val_loss:.4f} acc@1={val_acc1:.3f} | '
              f'lr: {optimizer.param_groups[0]["lr"]:.2e}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            full_path = os.path.join(args['save_path'], 'pretrain_clip_best.pt')
            enc_path = os.path.join(args['save_path'], 'encoder_clip_best.pt')
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_loss': val_loss, 'val_acc1': val_acc1, 'args': args}, full_path)
            torch.save(model.encoder.state_dict(), enc_path)
            print(f'  → saved best (val_loss={val_loss:.4f}, acc@1={val_acc1:.3f})')


if __name__ == '__main__':
    main()
