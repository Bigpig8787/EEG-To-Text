"""Pre-training with INPUT vs OUTPUT logging.

Identical to train_pretrain.py except: at the end of each epoch one validation
sample is run through the model and stored to disk so input EEG and the
reconstructed EEG can be compared after training.

Per epoch outputs:
  pretrain_io_samples/epoch_{N}.pt   — dict with input/recon/mask tensors
  pretrain_io.log                    — appended stats line per epoch

The .pt file format:
  {
      'epoch':       int,
      'input':       (105, 5000) tensor   — raw EEG fed into the model
      'recon':       (105, 5000) tensor   — model reconstruction
      'mask':        (105, 5000) bool     — which positions were masked
      'val_loss':    float,
      'mse_overall': float,
      'mse_masked':  float,
  }
"""

import os
import argparse
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.pretrain_model import ConformerPreTrainModel, create_remask
from data.pretrain_dataset import EEGPretrainDataset, RAW_EEG_MAX_LEN

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
_LOG_PATH = 'pretrain_io.log'
_SAMPLES_DIR = 'pretrain_io_samples'


def log(msg=''):
    print(msg)
    try:
        with open(_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(str(msg) + '\n')
    except Exception:
        pass


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ne', '--num_epochs', type=int, default=50)
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-5)
    parser.add_argument('-b', '--batch_size', type=int, default=4)
    parser.add_argument('-cuda', '--cuda', default='cuda:0')
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_filters', type=int, default=40)
    parser.add_argument('--temporal_kernel', type=int, default=200)
    parser.add_argument('--pool_stride', type=int, default=100)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_transformer_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--mask_ratio', type=float, default=0.15)
    parser.add_argument('-s', '--save_path', default='./checkpoints/pretrain_io')
    return vars(parser.parse_args())


def save_io_sample(model, val_loader, device, epoch, mask_ratio):
    """Run a single validation sample through the model and persist
    (input, recon, mask, stats) to disk + write a one-line summary to log."""
    model.eval()
    with torch.no_grad():
        batch_eeg, _ = next(iter(val_loader))
        batch_eeg = batch_eeg.to(device)
        x_masked, mask = create_remask(batch_eeg, mask_ratio)
        x_recon = model(x_masked)
        mask_exp = mask.expand_as(batch_eeg)

        # take just the first sample of the batch
        inp = batch_eeg[0].cpu()
        rec = x_recon[0].cpu()
        msk = mask_exp[0].cpu()

        mse_overall = F.mse_loss(rec, inp).item()
        mse_masked = (F.mse_loss(rec[msk], inp[msk]).item()
                      if msk.any() else float('nan'))

    os.makedirs(_SAMPLES_DIR, exist_ok=True)
    out_path = os.path.join(_SAMPLES_DIR, f'epoch_{epoch:03d}.pt')
    torch.save({
        'epoch': epoch,
        'input': inp,
        'recon': rec,
        'mask': msk,
        'mse_overall': mse_overall,
        'mse_masked': mse_masked,
    }, out_path)

    log(f'[IO  e{epoch:03d}] '
        f'in  μ={inp.mean():+.4f} σ={inp.std():.4f}  '
        f'rec μ={rec.mean():+.4f} σ={rec.std():.4f}  '
        f'mse_all={mse_overall:.6f} mse_masked={mse_masked:.6f}  '
        f'saved={out_path}')


def main():
    args = get_args()

    with open(_LOG_PATH, 'w', encoding='utf-8') as f:
        f.write(f'=== train_pretrain_io_log.py @ {time.strftime("%Y-%m-%d %H:%M:%S")} ===\n')

    log('=' * 60)
    log('EEG Pre-Training (I/O logged)')
    log('=' * 60)
    for k, v in args.items():
        log(f'  {k}: {v}')

    seed = 312
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device(args['cuda'] if torch.cuda.is_available() else 'cpu')
    log(f'[INFO] device: {device}')

    pickle_paths = []
    dataset_dir = os.path.join(PROJECT_ROOT, 'dataset', 'ZuCo')
    for task, fname in [('task1-SR','task1-SR-dataset.pickle'),
                        ('task2-NR','task2-NR-dataset.pickle'),
                        ('task3-TSR','task3-TSR-dataset.pickle'),
                        ('task2-NR-2.0','task2-NR-2.0-dataset.pickle')]:
        p = os.path.join(dataset_dir, task, 'pickle', fname)
        if os.path.exists(p):
            pickle_paths.append(p)

    train_ds = EEGPretrainDataset(pickle_paths, split='train')
    val_ds = EEGPretrainDataset(pickle_paths, split='dev')
    train_loader = DataLoader(train_ds, batch_size=args['batch_size'], shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args['batch_size'], shuffle=False,
                            num_workers=0)
    # deterministic single-sample loader for I/O snapshots (shuffle=False so
    # the SAME validation sample is logged every epoch — easier to compare)
    io_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = ConformerPreTrainModel(
        n_channels=105, d_model=args['d_model'], n_filters=args['n_filters'],
        temporal_kernel=args['temporal_kernel'], pool_stride=args['pool_stride'],
        n_heads=args['n_heads'], n_transformer_layers=args['n_transformer_layers'],
        dropout=args['dropout'], target_T=RAW_EEG_MAX_LEN,
    ).to(device)

    log(f'[INFO] params: {sum(p.numel() for p in model.parameters()):,}')

    optimizer = optim.AdamW(model.parameters(), lr=args['learning_rate'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['num_epochs'])

    os.makedirs(args['save_path'], exist_ok=True)
    best_val_loss = float('inf')
    best_wts = None

    # Snapshot at epoch -1 (untrained model) for a baseline comparison.
    save_io_sample(model, io_loader, device, epoch=-1, mask_ratio=args['mask_ratio'])

    for epoch in range(args['num_epochs']):
        # train
        model.train()
        total_loss, n = 0.0, 0
        for batch_eeg, _ in tqdm(train_loader, desc=f'Epoch {epoch}'):
            batch_eeg = batch_eeg.to(device)
            x_masked, mask = create_remask(batch_eeg, args['mask_ratio'])
            x_recon = model(x_masked)
            mask_exp = mask.expand_as(batch_eeg)
            loss = (F.mse_loss(x_recon[mask_exp], batch_eeg[mask_exp])
                    if mask_exp.any() else F.mse_loss(x_recon, batch_eeg))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * batch_eeg.size(0)
            n += batch_eeg.size(0)
        train_loss = total_loss / n

        # validate
        model.eval()
        total_loss, n = 0.0, 0
        with torch.no_grad():
            for batch_eeg, _ in val_loader:
                batch_eeg = batch_eeg.to(device)
                x_masked, mask = create_remask(batch_eeg, args['mask_ratio'])
                x_recon = model(x_masked)
                mask_exp = mask.expand_as(batch_eeg)
                loss = (F.mse_loss(x_recon[mask_exp], batch_eeg[mask_exp])
                        if mask_exp.any() else F.mse_loss(x_recon, batch_eeg))
                total_loss += loss.item() * batch_eeg.size(0)
                n += batch_eeg.size(0)
        val_loss = total_loss / n

        scheduler.step()
        log(f'Epoch {epoch}/{args["num_epochs"]-1} | train: {train_loss:.6f} '
            f'| val: {val_loss:.6f} | lr: {optimizer.param_groups[0]["lr"]:.2e}')

        # I/O snapshot — fixed validation sample, this epoch's weights
        save_io_sample(model, io_loader, device, epoch=epoch,
                       mask_ratio=args['mask_ratio'])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_wts = copy.deepcopy(model.state_dict())
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_loss': val_loss, 'args': args},
                       os.path.join(args['save_path'], 'pretrain_best.pt'))
            log(f'  → saved best (val_loss={val_loss:.6f})')

    model.load_state_dict(best_wts)
    torch.save(model.encoder.state_dict(),
               os.path.join(args['save_path'], 'encoder_best.pt'))
    log(f'Saved encoder: {os.path.join(args["save_path"], "encoder_best.pt")}')


if __name__ == '__main__':
    main()
