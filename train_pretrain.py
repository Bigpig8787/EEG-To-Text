"""
Train script for EEG pre-training with Conformer + Re-Masked Token Prediction.

Usage:
    python train_pretrain.py -cuda cuda:0 -b 8 -ne 50 -lr 0.0001

Pre-training uses all ZuCo data (no train/dev/test split) to maximize
the amount of EEG data the encoder sees.
After pre-training, the encoder weights are saved and used to initialize
the multi-view transformer for EEG-to-text decoding.
"""

import os
import argparse
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_pretrain import ConformerPreTrainModel, create_remask
from data_pretrain import EEGPretrainDataset, RAW_EEG_MAX_LEN

# ---- 路徑設定 ----
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


def get_pretrain_args():
    parser = argparse.ArgumentParser(description='EEG Pre-Training with Conformer')
    
    # training
    parser.add_argument('-ne', '--num_epochs', type=int, default=50)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-b', '--batch_size', type=int, default=8)
    parser.add_argument('-cuda', '--cuda', default='cuda:0')
    
    # model
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_filters', type=int, default=40)
    parser.add_argument('--temporal_kernel', type=int, default=25)
    parser.add_argument('--pool_stride', type=int, default=50)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_transformer_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # masking
    parser.add_argument('--mask_ratio', type=float, default=0.15)
    
    # save
    parser.add_argument('-s', '--save_path', default='./checkpoints/pretrain')
    
    return vars(parser.parse_args())


def train_one_epoch(model, dataloader, optimizer, device, mask_ratio, epoch):
    """
    Train one epoch with re-masking (new random mask for each batch).
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    for batch_eeg, batch_actual_len in tqdm(dataloader, desc=f'Epoch {epoch}'):
        batch_eeg = batch_eeg.to(device)  # (batch, 105, T)
        
        # ---- Re-Masked Token Prediction: new mask every batch ----
        x_masked, mask = create_remask(batch_eeg, mask_ratio=mask_ratio)
        
        # forward
        x_recon = model(x_masked)  # (batch, 105, T)
        
        # loss: MSE only on the masked positions
        # mask: (batch, 1, T) → expand to (batch, 105, T)
        mask_expanded = mask.expand_as(batch_eeg)
        
        # only compute loss where mask is True (masked positions)
        if mask_expanded.any():
            loss = F.mse_loss(x_recon[mask_expanded], batch_eeg[mask_expanded])
        else:
            loss = F.mse_loss(x_recon, batch_eeg)
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        batch_size = batch_eeg.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    
    avg_loss = total_loss / total_samples
    return avg_loss


@torch.no_grad()
def validate(model, dataloader, device, mask_ratio):
    """Validate with fixed mask."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    for batch_eeg, batch_actual_len in dataloader:
        batch_eeg = batch_eeg.to(device)
        
        x_masked, mask = create_remask(batch_eeg, mask_ratio=mask_ratio)
        x_recon = model(x_masked)
        
        mask_expanded = mask.expand_as(batch_eeg)
        if mask_expanded.any():
            loss = F.mse_loss(x_recon[mask_expanded], batch_eeg[mask_expanded])
        else:
            loss = F.mse_loss(x_recon, batch_eeg)
        
        batch_size = batch_eeg.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    
    avg_loss = total_loss / total_samples
    return avg_loss


def main():
    args = get_pretrain_args()
    
    print('=' * 60)
    print('EEG Pre-Training with Conformer + Re-Masked Token Prediction')
    print('=' * 60)
    for k, v in args.items():
        print(f'  {k}: {v}')
    print()
    
    # ---- random seed ----
    seed = 312
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # ---- device ----
    if torch.cuda.is_available():
        device = torch.device(args['cuda'])
    else:
        device = torch.device('cpu')
    print(f'[INFO] using device: {device}')
    
    # ---- dataset ----
    pickle_paths = []
    dataset_dir = os.path.join(PROJECT_ROOT, 'dataset', 'ZuCo')
    
    task_pickles = [
        ('task1-SR', 'task1-SR-dataset.pickle'),
        ('task2-NR', 'task2-NR-dataset.pickle'),
        ('task3-TSR', 'task3-TSR-dataset.pickle'),
        ('task2-NR-2.0', 'task2-NR-2.0-dataset.pickle'),
    ]
    
    for task_name, pickle_name in task_pickles:
        p = os.path.join(dataset_dir, task_name, 'pickle', pickle_name)
        if os.path.exists(p):
            pickle_paths.append(p)
            print(f'[INFO] found: {p}')
        else:
            print(f'[WARN] not found: {p}')
    
    # use 90% for train, 10% for validation
    train_dataset = EEGPretrainDataset(pickle_paths, split='train')
    val_dataset = EEGPretrainDataset(pickle_paths, split='dev')
    
    print(f'[INFO] train samples: {len(train_dataset)}')
    print(f'[INFO] val samples:   {len(val_dataset)}')
    
    train_loader = DataLoader(
        train_dataset, batch_size=args['batch_size'],
        shuffle=True, num_workers=0, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args['batch_size'],
        shuffle=False, num_workers=0
    )
    
    # ---- model ----
    model = ConformerPreTrainModel(
        n_channels=105,
        d_model=args['d_model'],
        n_filters=args['n_filters'],
        temporal_kernel=args['temporal_kernel'],
        pool_stride=args['pool_stride'],
        n_heads=args['n_heads'],
        n_transformer_layers=args['n_transformer_layers'],
        dropout=args['dropout'],
        target_T=RAW_EEG_MAX_LEN,
    )
    model.to(device)
    
    # print model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[INFO] total parameters: {total_params:,}')
    print(f'[INFO] trainable parameters: {trainable_params:,}')
    print()
    
    # ---- optimizer & scheduler ----
    optimizer = optim.AdamW(model.parameters(), lr=args['learning_rate'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['num_epochs'])
    
    # ---- save path ----
    save_path = args['save_path']
    os.makedirs(save_path, exist_ok=True)
    
    # ---- training loop ----
    best_val_loss = float('inf')
    best_model_wts = None
    
    import torch.nn.functional as F
    
    print('=== Start Pre-Training ===')
    since = time.time()
    
    for epoch in range(args['num_epochs']):
        # train (re-mask happens inside train_one_epoch for each batch)
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            args['mask_ratio'], epoch
        )
        
        # validate
        val_loss = validate(model, val_loader, device, args['mask_ratio'])
        
        # scheduler step
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch}/{args["num_epochs"]-1} | '
              f'train_loss: {train_loss:.6f} | val_loss: {val_loss:.6f} | '
              f'lr: {current_lr:.2e}')
        
        # save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            
            best_path = os.path.join(save_path, 'conformer_pretrain_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'args': args,
            }, best_path)
            print(f'  → saved best checkpoint: {best_path} (val_loss={val_loss:.6f})')
        
        print()
    
    # save last
    last_path = os.path.join(save_path, 'conformer_pretrain_last.pt')
    torch.save({
        'epoch': args['num_epochs'] - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'args': args,
    }, last_path)
    print(f'saved last checkpoint: {last_path}')
    
    # save encoder only (for downstream use)
    encoder_path = os.path.join(save_path, 'conformer_encoder_best.pt')
    model.load_state_dict(best_model_wts)
    torch.save(model.encoder.state_dict(), encoder_path)
    print(f'saved encoder-only weights: {encoder_path}')
    
    time_elapsed = time.time() - since
    print(f'\nPre-training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
    print(f'Best val loss: {best_val_loss:.6f}')


if __name__ == '__main__':
    main()
