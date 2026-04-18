import os
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.pretrain_model import ConformerPreTrainModel, create_remask


@dataclass
class PretrainConfig:
    num_epochs: int = 50
    learning_rate: float = 5e-5
    batch_size: int = 4
    patience: int = 10
    d_model: int = 512
    n_filters: int = 40
    temporal_kernel: int = 25
    pool_stride: int = 10
    n_heads: int = 8
    n_transformer_layers: int = 4
    dropout: float = 0.1
    mask_ratio: float = 0.15
    save_path: str = './checkpoints/pretrain'
    resume_from: Optional[str] = None
    use_amp: bool = True


class PretrainPipeline:
    """Pre-trains the Conformer encoder via masked EEG reconstruction (MSE)."""

    def __init__(self, pickle_paths: list, config: PretrainConfig, device):
        self.pickle_paths = pickle_paths
        self.cfg = config
        self.device = device

    def run(self) -> str:
        """Run pre-training. Returns path to saved encoder checkpoint."""
        cfg = self.cfg
        os.makedirs(cfg.save_path, exist_ok=True)

        from data.pretrain_dataset import EEGPretrainDataset
        train_ds = EEGPretrainDataset(self.pickle_paths, split='train')
        dev_ds   = EEGPretrainDataset(self.pickle_paths, split='dev')
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  drop_last=True)
        dev_loader   = DataLoader(dev_ds,   batch_size=cfg.batch_size, shuffle=False)

        model = ConformerPreTrainModel(
            n_channels=105, d_model=cfg.d_model, n_filters=cfg.n_filters,
            temporal_kernel=cfg.temporal_kernel, pool_stride=cfg.pool_stride,
            n_heads=cfg.n_heads, n_transformer_layers=cfg.n_transformer_layers,
            dropout=cfg.dropout,
        ).to(self.device)

        opt = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.num_epochs)
        scaler = GradScaler() if cfg.use_amp else None
        mse = nn.MSELoss()

        start_epoch, best_loss = 0, float('inf')
        pretrain_path = os.path.join(cfg.save_path, 'pretrain_best.pt')
        encoder_path  = os.path.join(cfg.save_path, 'encoder_best.pt')

        if cfg.resume_from and os.path.isfile(cfg.resume_from):
            ckpt = torch.load(cfg.resume_from, map_location=self.device)
            model.load_state_dict(ckpt['model_state'])
            opt.load_state_dict(ckpt['optimizer_state'])
            start_epoch = ckpt['epoch'] + 1
            best_loss   = ckpt.get('best_loss', float('inf'))
            print(f'[Pretrain] resumed from epoch {ckpt["epoch"]}')

        patience_ctr = 0
        for epoch in range(start_epoch, cfg.num_epochs):
            for phase, loader in (('train', train_loader), ('dev', dev_loader)):
                is_train = (phase == 'train')
                model.train() if is_train else model.eval()
                total_loss = n = 0

                pbar = tqdm(loader, desc=f'pretrain {phase} e{epoch+1}')
                for raw_eeg, actual_len in pbar:
                    raw_eeg = raw_eeg.to(self.device).float()
                    masked, mask = create_remask(raw_eeg, mask_ratio=cfg.mask_ratio)

                    with torch.set_grad_enabled(is_train):
                        if is_train and scaler:
                            with autocast():
                                recon = model(masked)
                                loss  = mse(recon[mask], raw_eeg[mask])
                            scaler.scale(loss).backward()
                            scaler.unscale_(opt)
                            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            scaler.step(opt)
                            scaler.update()
                            opt.zero_grad()
                        else:
                            with torch.no_grad():
                                recon = model(masked)
                                loss  = mse(recon[mask], raw_eeg[mask])

                    total_loss += loss.item() * raw_eeg.size(0)
                    n += raw_eeg.size(0)
                    pbar.set_postfix({'loss': f'{total_loss/n:.6f}'})

                epoch_loss = total_loss / max(n, 1)
                print(f'  {phase} loss: {epoch_loss:.6f}')

                if phase == 'dev':
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        patience_ctr = 0
                        torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                                    'optimizer_state': opt.state_dict(), 'best_loss': best_loss,
                                    'args': vars(cfg)}, pretrain_path)
                        torch.save(model.encoder.state_dict(), encoder_path)
                        print(f'  -> saved best encoder (loss={best_loss:.6f})')
                    else:
                        patience_ctr += 1
                        if patience_ctr >= cfg.patience:
                            print(f'  early stopping at epoch {epoch+1}')
                            return encoder_path
            sch.step()

        return encoder_path
