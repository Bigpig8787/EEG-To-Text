import os
import copy
import time
from dataclasses import dataclass, field
from typing import Optional, Dict

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm


@dataclass
class TrainerConfig:
    num_epochs: int = 30
    grad_accum_steps: int = 2
    patience: int = 10
    no_early_stop: bool = False
    label_smooth: float = 0.1
    max_grad_norm: float = 1.0
    use_amp: bool = True
    save_best: bool = True
    save_every: int = 0          # 0=off; N=also save every N epochs
    checkpoint_dir: str = './checkpoints'
    checkpoint_name: str = 'model'
    resume_from: Optional[str] = None   # path to checkpoint to resume


class Trainer:
    """Generic training loop with checkpointing, AMP, early stopping, and resume."""

    def __init__(self, model: nn.Module, dataloaders: Dict, optimizer, scheduler,
                 tokenizer, config: TrainerConfig, device):
        self.model = model
        self.loaders = dataloaders
        self.opt = optimizer
        self.sch = scheduler
        self.tok = tokenizer
        self.cfg = config
        self.device = device
        self.scaler = GradScaler() if config.use_amp else None

        self.best_path = os.path.join(config.checkpoint_dir, 'best', f'{config.checkpoint_name}.pt')
        self.last_path = os.path.join(config.checkpoint_dir, 'last', f'{config.checkpoint_name}.pt')
        os.makedirs(os.path.dirname(self.best_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.last_path), exist_ok=True)

        self.start_epoch = 0
        self.best_loss = float('inf')

        if config.resume_from and os.path.isfile(config.resume_from):
            self._resume(config.resume_from)

    # ── public ─────────────────────────────────────────────────────────

    def train(self) -> nn.Module:
        cfg = self.cfg
        eff_patience = 9999 if cfg.no_early_stop else cfg.patience
        patience_ctr = 0
        best_wts = copy.deepcopy(self.model.state_dict())

        for epoch in range(self.start_epoch, cfg.num_epochs):
            print(f'\nEpoch {epoch+1}/{cfg.num_epochs}')
            for phase in ('train', 'dev'):
                if phase not in self.loaders:
                    continue
                epoch_loss = self._run_phase(phase, epoch)

                if phase == 'dev':
                    is_best = epoch_loss < self.best_loss
                    if is_best:
                        self.best_loss = epoch_loss
                        best_wts = copy.deepcopy(self.model.state_dict())
                        patience_ctr = 0
                    else:
                        patience_ctr += 1
                        print(f'  no improvement ({patience_ctr}/{eff_patience})')
                    self._save(epoch, epoch_loss, is_best)
                    if patience_ctr >= eff_patience:
                        print(f'\nEarly stopping at epoch {epoch+1}')
                        break
            else:
                continue
            break  # inner break propagated

        self.model.load_state_dict(best_wts)
        return self.model

    # ── private ────────────────────────────────────────────────────────

    def _compute_loss(self, output, target_ids):
        if self.cfg.label_smooth > 0.0 and hasattr(output, 'logits'):
            fct = nn.CrossEntropyLoss(label_smoothing=self.cfg.label_smooth, ignore_index=-100)
            return fct(output.logits.view(-1, output.logits.size(-1)), target_ids.view(-1))
        return output.loss

    def _run_phase(self, phase: str, epoch: int) -> float:
        is_train = (phase == 'train')
        self.model.train() if is_train else self.model.eval()
        cfg = self.cfg
        running_loss = n = 0
        self.opt.zero_grad()
        pad_id = self.tok.pad_token_id

        pbar = tqdm(self.loaders[phase], desc=phase)
        for step, batch in enumerate(pbar):
            (_, _, attn_mask, attn_mask_inv,
             target_ids, _, _, _, raw_views) = batch

            attn_mask     = attn_mask.to(self.device)
            attn_mask_inv = attn_mask_inv.to(self.device)
            target_ids    = target_ids.to(self.device)
            target_ids[target_ids == pad_id] = -100
            view_inputs   = {k: v.to(self.device).float() for k, v in raw_views.items()}

            with torch.set_grad_enabled(is_train):
                if is_train and self.scaler:
                    with autocast():
                        out  = self.model(view_inputs, attn_mask, attn_mask_inv, target_ids)
                        loss = self._compute_loss(out, target_ids) / cfg.grad_accum_steps
                    self.scaler.scale(loss).backward()
                    if (step + 1) % cfg.grad_accum_steps == 0 or (step + 1) == len(self.loaders[phase]):
                        self.scaler.unscale_(self.opt)
                        nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                        self.scaler.step(self.opt)
                        self.scaler.update()
                        self.opt.zero_grad()
                        if self.sch:
                            self.sch.step()
                else:
                    with torch.no_grad():
                        out  = self.model(view_inputs, attn_mask, attn_mask_inv, target_ids)
                        loss = self._compute_loss(out, target_ids) / cfg.grad_accum_steps

            bs = target_ids.size(0)
            running_loss += loss.item() * cfg.grad_accum_steps * bs
            n += bs
            pbar.set_postfix({'loss': f'{running_loss/n:.4f}'})

        epoch_loss = running_loss / max(n, 1)
        lr = self.opt.param_groups[0]['lr']
        print(f'{phase} Loss: {epoch_loss:.4f} | lr: {lr:.2e}')
        return epoch_loss

    def _save(self, epoch: int, val_loss: float, is_best: bool):
        state = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.opt.state_dict(),
            'best_loss': self.best_loss,
            'val_loss': val_loss,
        }
        if self.sch:
            state['scheduler_state'] = self.sch.state_dict()
        torch.save(state, self.last_path)
        if is_best:
            torch.save(state, self.best_path)
            print(f'  -> saved best (val_loss={val_loss:.4f})')
        if self.cfg.save_every > 0 and (epoch + 1) % self.cfg.save_every == 0:
            ep_path = self.last_path.replace('.pt', f'_ep{epoch+1:03d}.pt')
            torch.save(state, ep_path)

    def _resume(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.opt.load_state_dict(ckpt['optimizer_state'])
        if self.sch and 'scheduler_state' in ckpt:
            self.sch.load_state_dict(ckpt['scheduler_state'])
        self.start_epoch = ckpt['epoch'] + 1
        self.best_loss   = ckpt.get('best_loss', float('inf'))
        print(f'[Trainer] resumed from epoch {ckpt["epoch"]}, best_loss={self.best_loss:.4f}')
