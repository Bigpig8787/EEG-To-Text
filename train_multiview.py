"""
Train Multi-View Conformer Translator for EEG-to-Text.

Improvements:
1. LoRA on BART (only ~2M trainable params in BART, prevents overfitting)
2. Mixed Precision (AMP) for 1.5-2x speedup and 30-40% VRAM savings
3. Warmup + Cosine scheduler for stable initial training
4. Gradient Accumulation (effective batch_size without extra VRAM)
5. Early Stopping (stop when dev loss plateaus)
6. All 10 view encoders train together (no freeze/unfreeze instability)
7. Single-stage training (no Step 1/2 needed since BART uses LoRA)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import pickle
import json
import time
import copy
from tqdm import tqdm
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig, get_cosine_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType

from data.dataset import ZuCo_dataset
from models.multiview import MultiViewConformerTranslator
from config import get_config

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def train_model(dataloaders, device, model, tokenizer, optimizer, scheduler, scaler,
                num_epochs=30, grad_accum_steps=2, patience=10,
                checkpoint_path_best='./checkpoints/best/temp.pt',
                checkpoint_path_last='./checkpoints/last/temp.pt'):
    os.makedirs(os.path.dirname(checkpoint_path_best), exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_path_last), exist_ok=True)
    since = time.time()

    best_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs-1}')
        print('-' * 40)

        for phase in ['train', 'dev']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            n_samples = 0
            optimizer.zero_grad()

            dataloader = dataloaders[phase]
            pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=phase)

            for step, (input_emb, seq_len, masks, mask_inv, target_ids,
                       target_mask, sentiment, sent_eeg, raw_views) in pbar:

                if not raw_views:
                    continue

                target_ids_batch = target_ids.to(device)
                masks_batch = masks.to(device)
                mask_inv_batch = mask_inv.to(device)
                view_inputs = {k: v.to(device).float() for k, v in raw_views.items()}

                target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100

                with torch.set_grad_enabled(phase == 'train'):
                    with autocast():
                        output = model(view_inputs, masks_batch, mask_inv_batch, target_ids_batch)
                        loss = output.loss / grad_accum_steps

                    if phase == 'train':
                        scaler.scale(loss).backward()

                        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
                            scaler.unscale_(optimizer)
                            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                            scheduler.step()

                bs = target_ids.size(0)
                running_loss += loss.item() * grad_accum_steps * bs
                n_samples += bs

                if n_samples > 0:
                    pbar.set_postfix({'loss': f'{running_loss/n_samples:.4f}'})

            epoch_loss = running_loss / max(n_samples, 1)
            current_lr = optimizer.param_groups[0]['lr']
            print(f'{phase} Loss: {epoch_loss:.4f} | lr: {current_lr:.2e}')

            if phase == 'dev':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), checkpoint_path_best)
                    print(f'  → saved best (val_loss={epoch_loss:.4f})')
                    patience_counter = 0
                else:
                    patience_counter += 1
                    print(f'  no improvement ({patience_counter}/{patience})')

        if patience_counter >= patience:
            print(f'\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)')
            break

        print()

    elapsed = time.time() - since
    print(f'\nTraining complete in {elapsed//60:.0f}m {elapsed%60:.0f}s')
    print(f'Best val loss: {best_loss:.4f}')
    torch.save(model.state_dict(), checkpoint_path_last)
    model.load_state_dict(best_wts)
    return model


if __name__ == '__main__':
    args = get_config('train_decoding')

    dataset_setting = 'unique_sent'
    num_epochs = args['num_epoch_step2']
    lr = args['learning_rate_step1']
    batch_size = args['batch_size']
    task_name = args['task_name']
    save_path = args['save_path']
    use_random_init = args['use_random_init']
    subject_choice = args['subjects']
    eeg_type_choice = args['eeg_type']
    bands_choice = args['eeg_bands']

    GRAD_ACCUM_STEPS = 2
    LORA_R = 16
    LORA_ALPHA = 32
    WARMUP_RATIO = 0.1
    PATIENCE = 7

    save_name = f'{task_name}_multiview_lora_r{LORA_R}_b{batch_size}_lr{lr}_{dataset_setting}'

    save_best = os.path.join(save_path, 'best')
    save_last = os.path.join(save_path, 'last')
    os.makedirs(save_best, exist_ok=True)
    os.makedirs(save_last, exist_ok=True)
    ckpt_best = os.path.join(save_best, f'{save_name}.pt')
    ckpt_last = os.path.join(save_last, f'{save_name}.pt')

    print('=' * 60)
    print('Multi-View Conformer + LoRA Training')
    print('=' * 60)
    print(f'  task: {task_name}')
    print(f'  lr: {lr}')
    print(f'  batch_size: {batch_size} (effective: {batch_size * GRAD_ACCUM_STEPS})')
    print(f'  epochs: {num_epochs}')
    print(f'  LoRA rank: {LORA_R}, alpha: {LORA_ALPHA}')
    print(f'  warmup: {WARMUP_RATIO*100:.0f}%')
    print(f'  patience: {PATIENCE}')
    print(f'  AMP: enabled')

    np.random.seed(312)
    torch.manual_seed(312)
    torch.cuda.manual_seed_all(312)

    device = torch.device(args['cuda'] if torch.cuda.is_available() else 'cpu')
    print(f'  device: {device}')

    # data
    whole_dataset_dicts = []
    dd = os.path.join(PROJECT_ROOT, 'dataset', 'ZuCo')
    for key, (task, fname) in {'task1': ('task1-SR','task1-SR-dataset.pickle'),
                                'task2': ('task2-NR','task2-NR-dataset.pickle'),
                                'task3': ('task3-TSR','task3-TSR-dataset.pickle'),
                                'taskNRv2': ('task2-NR-2.0','task2-NR-2.0-dataset.pickle')}.items():
        if key in task_name:
            p = os.path.join(dd, task, 'pickle', fname)
            with open(p, 'rb') as f:
                whole_dataset_dicts.append(pickle.load(f))

    cfg_dir = './config/decoding/'
    os.makedirs(cfg_dir, exist_ok=True)
    with open(os.path.join(cfg_dir, f'{save_name}.json'), 'w') as f:
        json.dump(args, f, indent=4)

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    train_set = ZuCo_dataset(whole_dataset_dicts, 'train', tokenizer,
                             subject=subject_choice, eeg_type=eeg_type_choice,
                             bands=bands_choice, setting=dataset_setting)
    dev_set = ZuCo_dataset(whole_dataset_dicts, 'dev', tokenizer,
                           subject=subject_choice, eeg_type=eeg_type_choice,
                           bands=bands_choice, setting=dataset_setting)

    print(f'  train: {len(train_set)}, dev: {len(dev_set)}')

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(dev_set, batch_size=1, shuffle=False, num_workers=0)
    dataloaders = {'train': train_loader, 'dev': val_loader}

    # model
    if use_random_init:
        bart_config = BartConfig.from_pretrained('facebook/bart-large')
        pretrained_bart = BartForConditionalGeneration(bart_config)
    else:
        pretrained_bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

    # LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )
    pretrained_bart = get_peft_model(pretrained_bart, lora_config)
    print('\n[INFO] LoRA applied to BART:')
    pretrained_bart.print_trainable_parameters()

    model = MultiViewConformerTranslator(
        pretrained_bart, d_model=512, n_filters=40, temporal_kernel=25,
        pool_stride=10, tokens_per_view=100, n_heads=8,
        n_encoder_layers=4, n_global_layers=3, dropout=0.1,
        decoder_embedding_size=1024,
    )

    # load pre-trained encoder
    encoder_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'pretrain', 'encoder_best.pt')
    if os.path.exists(encoder_path):
        model.load_pretrained_encoder(encoder_path)
    else:
        print(f'[WARN] No pre-trained encoder at {encoder_path}')

    model.to(device)

    total, trainable = count_params(model)
    print(f'\n[INFO] Total params: {total:,}')
    print(f'[INFO] Trainable params: {trainable:,} ({trainable/total*100:.1f}%)')

    # optimizer with param groups
    encoder_params = []
    lora_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'view_encoders' in name:
            encoder_params.append(param)
        elif 'lora' in name:
            lora_params.append(param)
        else:
            other_params.append(param)

    optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': lr},
        {'params': lora_params, 'lr': lr * 0.1},
        {'params': other_params, 'lr': lr * 0.5},
    ], weight_decay=0.01)

    print(f'\n[INFO] Param groups:')
    print(f'  encoders: {sum(p.numel() for p in encoder_params):,} (lr={lr})')
    print(f'  lora: {sum(p.numel() for p in lora_params):,} (lr={lr*0.1})')
    print(f'  other: {sum(p.numel() for p in other_params):,} (lr={lr*0.5})')

    # scheduler
    total_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * num_epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f'[INFO] Total steps: {total_steps}, warmup: {warmup_steps}')

    # AMP
    scaler = GradScaler()

    # train
    print('\n=== Training ===')
    trained_model = train_model(
        dataloaders, device, model, tokenizer,
        optimizer, scheduler, scaler,
        num_epochs=num_epochs,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        patience=PATIENCE,
        checkpoint_path_best=ckpt_best,
        checkpoint_path_last=ckpt_last,
    )