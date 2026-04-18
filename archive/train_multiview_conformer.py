"""
Train Multi-View Conformer Translator for EEG-to-Text decoding.

Usage:
    python train_multiview_conformer.py -cuda cuda:0

This script:
1. Loads pre-trained Conformer encoder weights
2. Initializes 10 regional encoders (temporal conv from pre-trained, spatial conv random)
3. Two-step training: freeze most BART params → unfreeze all
4. Uses sentence-level raw EEG split by brain regions
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pickle
import json
import time
import copy
from tqdm import tqdm
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

from data import ZuCo_dataset
from model_multiview import MultiViewConformerTranslator
from config import get_config

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


def freeze_unfreeze_view_encoders(model, n_active=3):
    """
    Randomly select n_active view encoders to train, freeze the rest.
    Following EEG2TEXT paper Table 7 strategy.
    
    Args:
        model: MultiViewConformerTranslator
        n_active: number of encoders to unfreeze (paper best = 3)
    
    Returns:
        list of active region names
    """
    import random
    region_names = list(model.view_encoders.keys())
    
    # randomly select n_active regions to train
    active_regions = random.sample(region_names, n_active)
    
    for region in region_names:
        encoder = model.view_encoders[region]
        requires_grad = (region in active_regions)
        for param in encoder.parameters():
            param.requires_grad = requires_grad
    
    # global transformer and fc1 always trainable
    for param in model.global_transformer.parameters():
        param.requires_grad = True
    for param in model.fc1.parameters():
        param.requires_grad = True
    
    return active_regions


def train_model(dataloaders, device, model, tokenizer, model_name, optimizer, scheduler,
                num_epochs=25,
                checkpoint_path_best='./checkpoints/multiview/best/temp.pt',
                checkpoint_path_last='./checkpoints/multiview/last/temp.pt',
                n_active_views=3):
    os.makedirs(os.path.dirname(checkpoint_path_best), exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_path_last), exist_ok=True)
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(num_epochs):
        # ---- freeze/unfreeze view encoders each epoch ----
        active_regions = freeze_unfreeze_view_encoders(model, n_active=n_active_views)
        print(f'Epoch {epoch}/{num_epochs - 1} | active views: {active_regions}')
        print('-' * 40)

        # rebuild optimizer to only include trainable params
        current_lr = optimizer.param_groups[0]['lr']
        optimizer = type(optimizer)(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=current_lr, momentum=0.9
        )

        for phase in ['train', 'dev']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            num_samples = 0

            for (input_embeddings, seq_len, input_masks, input_mask_invert,
                 target_ids, target_mask, sentiment_labels, sent_level_EEG, raw_eeg_views) in tqdm(dataloaders[phase], desc=f'{phase}'):

                # skip if no raw_eeg_views
                if not raw_eeg_views:
                    continue

                target_ids_batch = target_ids.to(device)
                input_masks_batch = input_masks.to(device)
                input_mask_invert_batch = input_mask_invert.to(device)

                # move view inputs to device
                view_inputs = {k: v.to(device).float() for k, v in raw_eeg_views.items()}

                # replace padding ids with -100
                target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    output = model(view_inputs, input_masks_batch,
                                   input_mask_invert_batch, target_ids_batch)
                    loss = output.loss

                    if phase == 'train':
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                batch_size = target_ids.size(0)
                running_loss += loss.item() * batch_size
                num_samples += batch_size

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / max(num_samples, 1)
            print(f'{phase} Loss: {epoch_loss:.4f}')

            if phase == 'dev' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), checkpoint_path_best)
                print(f'  → saved best checkpoint (val_loss={epoch_loss:.4f})')

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val loss: {best_loss:.4f}')
    torch.save(model.state_dict(), checkpoint_path_last)
    print(f'Saved last checkpoint: {checkpoint_path_last}')

    model.load_state_dict(best_model_wts)
    return model


def show_require_grad_layers(model):
    print('\n require_grad layers:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(' ', name)


if __name__ == '__main__':
    args = get_config('train_decoding')

    dataset_setting = 'unique_sent'
    num_epochs_step1 = args['num_epoch_step1']
    num_epochs_step2 = args['num_epoch_step2']
    step1_lr = args['learning_rate_step1']
    step2_lr = args['learning_rate_step2']
    batch_size = args['batch_size']
    model_name = args['model_name']
    task_name = args['task_name']
    save_path = args['save_path']
    skip_step_one = args['skip_step_one']
    use_random_init = args['use_random_init']

    print(f'[INFO] model: {model_name}')
    print(f'[INFO] task: {task_name}')

    # ---- save name ----
    if skip_step_one:
        save_name = f'{task_name}_multiview_conformer_skipstep1_b{batch_size}_{num_epochs_step1}_{num_epochs_step2}_{step1_lr}_{step2_lr}_{dataset_setting}'
    else:
        save_name = f'{task_name}_multiview_conformer_2step_b{batch_size}_{num_epochs_step1}_{num_epochs_step2}_{step1_lr}_{step2_lr}_{dataset_setting}'

    save_path_best = os.path.join(save_path, 'best')
    save_path_last = os.path.join(save_path, 'last')
    os.makedirs(save_path_best, exist_ok=True)
    os.makedirs(save_path_last, exist_ok=True)
    output_checkpoint_best = os.path.join(save_path_best, f'{save_name}.pt')
    output_checkpoint_last = os.path.join(save_path_last, f'{save_name}.pt')

    subject_choice = args['subjects']
    eeg_type_choice = args['eeg_type']
    bands_choice = args['eeg_bands']
    print(f'[INFO] subjects: {subject_choice}')
    print(f'[INFO] eeg_type: {eeg_type_choice}')
    print(f'[INFO] bands: {bands_choice}')

    # ---- seed ----
    seed_val = 312
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # ---- device ----
    if torch.cuda.is_available():
        dev = args['cuda']
    else:
        dev = 'cpu'
    device = torch.device(dev)
    print(f'[INFO] device: {device}')

    # ---- load data ----
    whole_dataset_dicts = []
    dataset_dir = os.path.join(PROJECT_ROOT, 'dataset', 'ZuCo')

    if 'task1' in task_name:
        p = os.path.join(dataset_dir, 'task1-SR', 'pickle', 'task1-SR-dataset.pickle')
        with open(p, 'rb') as f:
            whole_dataset_dicts.append(pickle.load(f))
    if 'task2' in task_name:
        p = os.path.join(dataset_dir, 'task2-NR', 'pickle', 'task2-NR-dataset.pickle')
        with open(p, 'rb') as f:
            whole_dataset_dicts.append(pickle.load(f))
    if 'task3' in task_name:
        p = os.path.join(dataset_dir, 'task3-TSR', 'pickle', 'task3-TSR-dataset.pickle')
        with open(p, 'rb') as f:
            whole_dataset_dicts.append(pickle.load(f))
    if 'taskNRv2' in task_name:
        p = os.path.join(dataset_dir, 'task2-NR-2.0', 'pickle', 'task2-NR-2.0-dataset.pickle')
        with open(p, 'rb') as f:
            whole_dataset_dicts.append(pickle.load(f))

    # ---- save config ----
    cfg_dir = './config/decoding/'
    os.makedirs(cfg_dir, exist_ok=True)
    with open(os.path.join(cfg_dir, f'{save_name}.json'), 'w') as f:
        json.dump(args, f, indent=4)

    # ---- tokenizer & dataset ----
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    train_set = ZuCo_dataset(whole_dataset_dicts, 'train', tokenizer,
                             subject=subject_choice, eeg_type=eeg_type_choice,
                             bands=bands_choice, setting=dataset_setting)
    dev_set = ZuCo_dataset(whole_dataset_dicts, 'dev', tokenizer,
                           subject=subject_choice, eeg_type=eeg_type_choice,
                           bands=bands_choice, setting=dataset_setting)

    dataset_sizes = {'train': len(train_set), 'dev': len(dev_set)}
    print(f'[INFO] train_set: {len(train_set)}')
    print(f'[INFO] dev_set: {len(dev_set)}')

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(dev_set, batch_size=1, shuffle=False, num_workers=0)
    dataloaders = {'train': train_loader, 'dev': val_loader}

    # ---- model ----
    if use_random_init:
        config = BartConfig.from_pretrained('facebook/bart-large')
        pretrained_bart = BartForConditionalGeneration(config)
    else:
        pretrained_bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

    model = MultiViewConformerTranslator(
        pretrained_bart,
        d_model=512,
        n_filters=40,
        temporal_kernel=25,
        pool_stride=10,
        tokens_per_view=100,
        n_heads=8,
        n_encoder_layers=4,
        n_global_layers=3,
        dropout=0.1,
        decoder_embedding_size=1024,
    )

    # ---- load pre-trained encoder ----
    pretrained_encoder_path = os.path.join(
        PROJECT_ROOT, 'checkpoints', 'pretrain_s10_d512', 'conformer_encoder_best.pt'
    )
    if os.path.exists(pretrained_encoder_path):
        model.load_pretrained_encoder(pretrained_encoder_path)
    else:
        print(f'[WARN] Pre-trained encoder not found: {pretrained_encoder_path}')
        print('[WARN] Training from scratch without pre-trained weights')

    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[INFO] total params: {total_params:,}')
    print(f'[INFO] trainable params: {trainable_params:,}')

    # ====== Step 1: freeze most BART params ======
    for name, param in model.named_parameters():
        if param.requires_grad and 'pretrained' in name:
            if ('shared' in name) or ('embed_positions' in name) or ('encoder.layers.0' in name):
                continue
            else:
                param.requires_grad = False

    if skip_step_one:
        print('[INFO] Skipping step 1')
    else:
        optimizer_step1 = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=step1_lr, momentum=0.9
        )
        scheduler_step1 = lr_scheduler.StepLR(optimizer_step1, step_size=20, gamma=0.1)
        criterion = nn.CrossEntropyLoss()

        print('\n=== Step 1: Train encoders + partial BART ===')
        show_require_grad_layers(model)
        model = train_model(
            dataloaders, device, model, tokenizer, model_name,
            optimizer_step1, scheduler_step1,
            num_epochs=num_epochs_step1,
            checkpoint_path_best=output_checkpoint_best,
            checkpoint_path_last=output_checkpoint_last,
            n_active_views=3,
        )

    # ====== Step 2: unfreeze all, fine-tune ======
    for name, param in model.named_parameters():
        param.requires_grad = True

    optimizer_step2 = optim.SGD(model.parameters(), lr=step2_lr, momentum=0.9)
    scheduler_step2 = lr_scheduler.StepLR(optimizer_step2, step_size=30, gamma=0.1)

    print('\n=== Step 2: Fine-tune entire model ===')
    show_require_grad_layers(model)
    trained_model = train_model(
        dataloaders, device, model, tokenizer, model_name,
        optimizer_step2, scheduler_step2,
        num_epochs=num_epochs_step2,
        checkpoint_path_best=output_checkpoint_best,
        checkpoint_path_last=output_checkpoint_last,
        n_active_views=3,
    )
