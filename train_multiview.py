"""
Train Multi-View Conformer Translator for EEG-to-Text.

Key fixes from previous version:
1. Does NOT rebuild optimizer each epoch (was causing momentum reset → instability)
2. Proper two-step training (Step 1 freezes BART, Step 2 unfreezes)
3. freeze/unfreeze only sets requires_grad, optimizer updates all params but frozen ones get zero grad
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import pickle
import json
import time
import copy
from tqdm import tqdm
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

from data.dataset import ZuCo_dataset
from models.multiview import MultiViewConformerTranslator
from config import get_config

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


def train_model(dataloaders, dataset_sizes, device, model, tokenizer,
                optimizer, scheduler, num_epochs=25, n_active_views=3,
                checkpoint_path_best='./checkpoints/best/temp.pt',
                checkpoint_path_last='./checkpoints/last/temp.pt'):
    os.makedirs(os.path.dirname(checkpoint_path_best), exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_path_last), exist_ok=True)
    since = time.time()
    best_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(num_epochs):
        # set active views (freeze 7, unfreeze 3)
        active = model.set_active_views(n_active=n_active_views)
        print(f'Epoch {epoch}/{num_epochs-1} | active: {active}')
        print('-' * 40)

        for phase in ['train', 'dev']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            n_samples = 0

            for (input_emb, seq_len, masks, mask_inv, target_ids,
                 target_mask, sentiment, sent_eeg, raw_views) in tqdm(dataloaders[phase], desc=phase):

                if not raw_views:
                    continue

                target_ids_batch = target_ids.to(device)
                masks_batch = masks.to(device)
                mask_inv_batch = mask_inv.to(device)
                view_inputs = {k: v.to(device).float() for k, v in raw_views.items()}

                target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100

                with torch.set_grad_enabled(phase == 'train'):
                    output = model(view_inputs, masks_batch, mask_inv_batch, target_ids_batch)
                    loss = output.loss

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                bs = target_ids.size(0)
                running_loss += loss.item() * bs
                n_samples += bs

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / max(n_samples, 1)
            print(f'{phase} Loss: {epoch_loss:.4f}')

            if phase == 'dev' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), checkpoint_path_best)
                print(f'  → saved best (val_loss={epoch_loss:.4f})')
        print()

    elapsed = time.time() - since
    print(f'Training complete in {elapsed//60:.0f}m {elapsed%60:.0f}s')
    print(f'Best val loss: {best_loss:.4f}')
    torch.save(model.state_dict(), checkpoint_path_last)
    model.load_state_dict(best_wts)
    return model


if __name__ == '__main__':
    args = get_config('train_decoding')

    dataset_setting = 'unique_sent'
    num_epochs_step1 = args['num_epoch_step1']
    num_epochs_step2 = args['num_epoch_step2']
    step1_lr = args['learning_rate_step1']
    step2_lr = args['learning_rate_step2']
    batch_size = args['batch_size']
    task_name = args['task_name']
    save_path = args['save_path']
    skip_step_one = args['skip_step_one']
    use_random_init = args['use_random_init']

    subject_choice = args['subjects']
    eeg_type_choice = args['eeg_type']
    bands_choice = args['eeg_bands']

    # save name
    step_label = 'skipstep1' if skip_step_one else '2step'
    save_name = f'{task_name}_multiview_{step_label}_b{batch_size}_{num_epochs_step1}_{num_epochs_step2}_{step1_lr}_{step2_lr}_{dataset_setting}'

    save_best = os.path.join(save_path, 'best')
    save_last = os.path.join(save_path, 'last')
    os.makedirs(save_best, exist_ok=True)
    os.makedirs(save_last, exist_ok=True)
    ckpt_best = os.path.join(save_best, f'{save_name}.pt')
    ckpt_last = os.path.join(save_last, f'{save_name}.pt')

    print(f'[INFO] task: {task_name}')
    print(f'[INFO] subjects: {subject_choice}')
    print(f'[INFO] skip_step_one: {skip_step_one}')

    # seed
    np.random.seed(312)
    torch.manual_seed(312)
    torch.cuda.manual_seed_all(312)

    device = torch.device(args['cuda'] if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] device: {device}')

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

    # save config
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

    dataset_sizes = {'train': len(train_set), 'dev': len(dev_set)}
    print(f'[INFO] train: {len(train_set)}, dev: {len(dev_set)}')

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(dev_set, batch_size=1, shuffle=False, num_workers=0)
    dataloaders = {'train': train_loader, 'dev': val_loader}

    # model
    if use_random_init:
        bart_config = BartConfig.from_pretrained('facebook/bart-large')
        pretrained_bart = BartForConditionalGeneration(bart_config)
    else:
        pretrained_bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

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
        print(f'[WARN] No pre-trained encoder found at {encoder_path}')

    model.to(device)
    print(f'[INFO] params: {sum(p.numel() for p in model.parameters()):,}')

    # ====== Step 1: freeze most BART, train encoders ======
    if not skip_step_one:
        # freeze BART except embedding + first encoder layer
        for name, param in model.named_parameters():
            if 'pretrained' in name:
                if ('shared' in name) or ('embed_positions' in name) or ('encoder.layers.0' in name):
                    continue
                param.requires_grad = False

        optimizer1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=step1_lr, weight_decay=0.01)
        scheduler1 = lr_scheduler.CosineAnnealingLR(optimizer1, T_max=num_epochs_step1)

        print('\n=== Step 1: Train encoders + partial BART ===')
        model = train_model(dataloaders, dataset_sizes, device, model, tokenizer,
                            optimizer1, scheduler1, num_epochs=num_epochs_step1,
                            n_active_views=3,
                            checkpoint_path_best=ckpt_best, checkpoint_path_last=ckpt_last)
    else:
        print('[INFO] Skipping Step 1')

    # ====== Step 2: unfreeze all BART, fine-tune ======
    for param in model.parameters():
        param.requires_grad = True

    optimizer2 = optim.AdamW(model.parameters(), lr=step2_lr, weight_decay=0.01)
    scheduler2 = lr_scheduler.CosineAnnealingLR(optimizer2, T_max=num_epochs_step2)

    print('\n=== Step 2: Fine-tune entire model ===')
    model = train_model(dataloaders, dataset_sizes, device, model, tokenizer,
                        optimizer2, scheduler2, num_epochs=num_epochs_step2,
                        n_active_views=3,
                        checkpoint_path_best=ckpt_best, checkpoint_path_last=ckpt_last)
