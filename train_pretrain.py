"""Pre-training: Conformer + Re-Masked Token Prediction."""

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
from data.pickle_paths import (DEFAULT_PICKLE_SUFFIX, default_dataset_root,
                               existing_pickle_paths)

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ne', '--num_epochs', type=int, default=50)
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-5)
    parser.add_argument('-b', '--batch_size', type=int, default=4)
    parser.add_argument('-cuda', '--cuda', default='cuda:0')
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_filters', type=int, default=40)
    parser.add_argument('--n_spatial_filters', type=int, default=None,
                        help='spatial conv output channels; None = n_filters. Must equal '
                             'the fine-tune value or `projection` will not transfer')
    parser.add_argument('--temporal_kernel', type=int, default=200)
    parser.add_argument('--pool_stride', type=int, default=100)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_transformer_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--mask_ratio', type=float, default=0.15)
    parser.add_argument('-s', '--save_path', default='./checkpoints/pretrain')
    parser.add_argument('--dataset_root', default=None,
                        help='folder holding <task>/pickle/; default <repo>/dataset/ZuCo')
    parser.add_argument('--pickle_suffix', default=DEFAULT_PICKLE_SUFFIX,
                        help="'-dataset.pickle' (500 Hz) or '-dataset-200hz.pickle'")
    parser.add_argument('--raw_eeg_max_len', type=int, default=RAW_EEG_MAX_LEN,
                        help='pad/truncate length; 5000 for 500 Hz, 2000 for 200 Hz')
    return vars(parser.parse_args())


def main():
    args = get_args()
    print('=' * 60)
    print('EEG Pre-Training')
    print('=' * 60)
    for k, v in args.items():
        print(f'  {k}: {v}')

    seed = 312
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device(args['cuda'] if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] device: {device}')

    # dataset
    dataset_dir = args['dataset_root'] or default_dataset_root(PROJECT_ROOT)
    pickle_paths = existing_pickle_paths(dataset_dir, args['pickle_suffix'])
    if not pickle_paths:
        raise FileNotFoundError(
            f'no *{args["pickle_suffix"]} pickle under {dataset_dir}/<task>/pickle/')
    max_len = args['raw_eeg_max_len']

    train_ds = EEGPretrainDataset(pickle_paths, split='train', raw_eeg_max_len=max_len)
    val_ds = EEGPretrainDataset(pickle_paths, split='dev', raw_eeg_max_len=max_len)
    train_loader = DataLoader(train_ds, batch_size=args['batch_size'], shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args['batch_size'], shuffle=False, num_workers=0)

    # model
    model = ConformerPreTrainModel(
        n_channels=105, d_model=args['d_model'], n_filters=args['n_filters'],
        temporal_kernel=args['temporal_kernel'], pool_stride=args['pool_stride'],
        n_heads=args['n_heads'], n_transformer_layers=args['n_transformer_layers'],
        dropout=args['dropout'], target_T=max_len,
        n_spatial_filters=args['n_spatial_filters'],
    ).to(device)

    print(f'[INFO] params: {sum(p.numel() for p in model.parameters()):,}')

    optimizer = optim.AdamW(model.parameters(), lr=args['learning_rate'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['num_epochs'])

    os.makedirs(args['save_path'], exist_ok=True)
    best_val_loss = float('inf')
    best_wts = None

    for epoch in range(args['num_epochs']):
        # train
        model.train()
        total_loss, n = 0.0, 0
        for batch_eeg, _ in tqdm(train_loader, desc=f'Epoch {epoch}'):
            batch_eeg = batch_eeg.to(device)
            x_masked, mask = create_remask(batch_eeg, args['mask_ratio'],
                                           block_size=args['pool_stride'])
            x_recon = model(x_masked, mask)
            mask_exp = mask.expand_as(batch_eeg)
            loss = F.mse_loss(x_recon[mask_exp], batch_eeg[mask_exp]) if mask_exp.any() else F.mse_loss(x_recon, batch_eeg)
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
                x_masked, mask = create_remask(batch_eeg, args['mask_ratio'],
                                               block_size=args['pool_stride'])
                x_recon = model(x_masked, mask)
                mask_exp = mask.expand_as(batch_eeg)
                loss = F.mse_loss(x_recon[mask_exp], batch_eeg[mask_exp]) if mask_exp.any() else F.mse_loss(x_recon, batch_eeg)
                total_loss += loss.item() * batch_eeg.size(0)
                n += batch_eeg.size(0)
        val_loss = total_loss / n

        scheduler.step()
        print(f'Epoch {epoch}/{args["num_epochs"]-1} | train: {train_loss:.6f} | val: {val_loss:.6f} | lr: {optimizer.param_groups[0]["lr"]:.2e}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_wts = copy.deepcopy(model.state_dict())
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_loss': val_loss, 'args': args},
                       os.path.join(args['save_path'], 'pretrain_best.pt'))
            print(f'  → saved best (val_loss={val_loss:.6f})')

    # save encoder only
    model.load_state_dict(best_wts)
    torch.save(model.encoder.state_dict(), os.path.join(args['save_path'], 'encoder_best.pt'))
    print(f'Saved encoder: {os.path.join(args["save_path"], "encoder_best.pt")}')


if __name__ == '__main__':
    main()
