"""
Train Multi-View Conformer Translator — two-step training matching the paper.

Step 1 (encoder warm-up):
  - BART fully frozen
  - View encoders + global transformer + fc1 trained at high LR (5e-5)
  - 20 epochs

Step 2 (LoRA fine-tune):
  - LoRA applied to BART (q_proj, v_proj)
  - All trainable params at lower LR
  - 30 epochs

After training the LoRA weights are merged into BART and a plain checkpoint is
saved so eval_multiview.py can load it without PEFT installed.
"""

import os
import copy
import json
import pickle
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (BartConfig, BartForConditionalGeneration,
                          BartTokenizer, get_cosine_schedule_with_warmup)
from peft import LoraConfig, TaskType, get_peft_model

from config import get_config
from data.dataset import ZuCo_dataset
from models.multiview import MultiViewConformerTranslator


def augment_eeg_views(view_inputs: dict, is_train: bool = True) -> dict:
    """EEG data augmentation applied per-batch during training.

    Three transforms applied independently with independent probabilities:
      - Amplitude scaling  (p=0.7): multiply by uniform(0.80, 1.20)
      - Gaussian noise     (p=0.5): add N(0, 0.03) noise
      - Time shift         (p=0.5): circular roll ±5 % of T
    """
    if not is_train:
        return view_inputs
    augmented = {}
    for region, eeg in view_inputs.items():
        # eeg: (B, C, T)
        B, C, T = eeg.shape
        # amplitude scale
        if random.random() < 0.7:
            scale = torch.empty(B, 1, 1, device=eeg.device).uniform_(0.80, 1.20)
            eeg = eeg * scale
        # Gaussian noise
        if random.random() < 0.5:
            eeg = eeg + torch.randn_like(eeg) * 0.03
        # time shift
        if random.random() < 0.5:
            shift = random.randint(-T // 20, T // 20)
            if shift != 0:
                eeg = torch.roll(eeg, shift, dims=2)
        augmented[region] = eeg
    return augmented

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


_LOG_PATH = 'train.log'


def log(msg=''):
    """Print to stdout AND append to train.log. Used for important events
    (epoch/loss/step headers/save notifications) so the log stays clean
    even when tqdm progress bars spam stdout."""
    print(msg)
    try:
        with open(_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(str(msg) + '\n')
    except Exception:
        pass


def train_model(dataloaders, device, model, tokenizer, optimizer, scheduler, scaler,
                num_epochs=30, grad_accum_steps=2, patience=10, label_smooth=0.0,
                checkpoint_path_best='./checkpoints/best/temp.pt',
                checkpoint_path_last='./checkpoints/last/temp.pt'):
    os.makedirs(os.path.dirname(checkpoint_path_best), exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_path_last), exist_ok=True)
    since = time.time()

    best_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        log(f'Epoch {epoch}/{num_epochs - 1}')
        log('-' * 40)

        for phase in ['train', 'dev']:
            model.train() if phase == 'train' else model.eval()

            running_loss = 0.0
            n_samples = 0
            optimizer.zero_grad()

            pbar = tqdm(enumerate(dataloaders[phase]), total=len(dataloaders[phase]), desc=phase)
            for step, (input_emb, seq_len, masks, mask_inv, target_ids,
                       target_mask, sentiment, sent_eeg, raw_views) in pbar:

                if not raw_views:
                    continue

                target_ids_batch = target_ids.to(device)
                masks_batch = masks.to(device)
                mask_inv_batch = mask_inv.to(device)
                view_inputs = {k: v.to(device).float() for k, v in raw_views.items()}
                target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100

                view_inputs = augment_eeg_views(view_inputs, is_train=(phase == 'train'))

                with torch.set_grad_enabled(phase == 'train'):
                    with autocast():
                        output = model(view_inputs, masks_batch, mask_inv_batch, target_ids_batch)
                        if label_smooth > 0.0:
                            lm_logits = output.logits
                            labels = target_ids_batch.clone()
                            loss_fct = nn.CrossEntropyLoss(
                                label_smoothing=label_smooth, ignore_index=-100)
                            loss = loss_fct(
                                lm_logits.view(-1, lm_logits.size(-1)),
                                labels.view(-1)) / grad_accum_steps
                        else:
                            loss = output.loss / grad_accum_steps

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloaders[phase]):
                            scaler.unscale_(optimizer)
                            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                            if scheduler is not None:
                                scheduler.step()

                bs = target_ids.size(0)
                running_loss += loss.item() * grad_accum_steps * bs
                n_samples += bs
                pbar.set_postfix({'loss': f'{running_loss / n_samples:.4f}'})

            epoch_loss = running_loss / max(n_samples, 1)
            lr_now = optimizer.param_groups[0]['lr']
            log(f'{phase} Loss: {epoch_loss:.4f} | lr: {lr_now:.2e}')

            if phase == 'dev':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), checkpoint_path_best)
                    log(f'  → saved best (val_loss={epoch_loss:.4f})')
                    patience_counter = 0
                else:
                    patience_counter += 1
                    log(f'  no improvement ({patience_counter}/{patience})')

        if patience_counter >= patience:
            log(f'\nEarly stopping at epoch {epoch}')
            break
        print()

    elapsed = time.time() - since
    log(f'\nTraining complete in {elapsed // 60:.0f}m {elapsed % 60:.0f}s')
    log(f'Best val loss: {best_loss:.4f}')
    torch.save(model.state_dict(), checkpoint_path_last)
    model.load_state_dict(best_wts)
    return model


if __name__ == '__main__':
    args = get_config('train_decoding')

    with open(_LOG_PATH, 'w', encoding='utf-8') as _f:
        _f.write(f'=== train_multiview.py run @ {time.strftime("%Y-%m-%d %H:%M:%S")} ===\n')
    log(f'[INFO] Logging important events to {_LOG_PATH}')

    # ── hyper-params ────────────────────────────────────────────────
    STEP1_EPOCHS   = args['num_epoch_step1']   # freeze BART, warm up encoders
    STEP2_EPOCHS   = args['num_epoch_step2']   # LoRA fine-tune
    LR1            = args['learning_rate_step1']
    LR2            = args['learning_rate_step2']
    BATCH_SIZE     = args['batch_size']
    TASK_NAME      = args['task_name']
    SAVE_PATH      = args['save_path']
    SUBJECT_CHOICE = args['subjects']
    EEG_TYPE       = args['eeg_type']
    BANDS_CHOICE   = args['eeg_bands']

    GRAD_ACCUM     = 2
    LORA_R         = args.get('lora_r', 16)
    LORA_ALPHA     = LORA_R
    LORA_TARGETS   = args.get('lora_targets', ['q_proj', 'k_proj', 'v_proj', 'out_proj'])
    LABEL_SMOOTH   = args.get('label_smooth', 0.1)
    WARMUP_RATIO   = 0.2
    PATIENCE       = 9999 if args.get('no_early_stop', False) else args.get('patience', 10)

    save_name = f'{TASK_NAME}_multiview_2step_b{BATCH_SIZE}_{STEP1_EPOCHS}_{STEP2_EPOCHS}_{LR1}_{LR2}_unique_sent'

    np.random.seed(312)
    torch.manual_seed(312)
    torch.cuda.manual_seed_all(312)

    device = torch.device(args['cuda'] if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

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
        if key in TASK_NAME:
            with open(os.path.join(dd, task, 'pickle', fname), 'rb') as f:
                whole_dataset_dicts.append(pickle.load(f))

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    train_set = ZuCo_dataset(whole_dataset_dicts, 'train', tokenizer,
                             subject=SUBJECT_CHOICE, eeg_type=EEG_TYPE,
                             bands=BANDS_CHOICE, setting='unique_sent')
    dev_set = ZuCo_dataset(whole_dataset_dicts, 'dev', tokenizer,
                           subject=SUBJECT_CHOICE, eeg_type=EEG_TYPE,
                           bands=BANDS_CHOICE, setting='unique_sent')
    print(f'train: {len(train_set)}, dev: {len(dev_set)}')

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(dev_set,   batch_size=1,          shuffle=False, num_workers=0)
    dataloaders  = {'train': train_loader, 'dev': val_loader}

    cfg_dir = './config/decoding/'
    os.makedirs(cfg_dir, exist_ok=True)
    with open(os.path.join(cfg_dir, f'{save_name}.json'), 'w') as f:
        json.dump(args, f, indent=4)

    # ── model (plain BART, no LoRA yet) ─────────────────────────────
    if args['use_random_init']:
        bart = BartForConditionalGeneration(BartConfig.from_pretrained('facebook/bart-large'))
    else:
        bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

    model = MultiViewConformerTranslator(
        bart, d_model=512, n_filters=40, temporal_kernel=25,
        pool_stride=10, tokens_per_view=100, n_heads=8,
        n_encoder_layers=4, n_global_layers=3, dropout=0.1,
        decoder_embedding_size=1024,
    )

    encoder_path = os.path.join(PROJECT_ROOT, 'checkpoints', 'pretrain', 'encoder_best.pt')
    if os.path.exists(encoder_path):
        model.load_pretrained_encoder(encoder_path)
    else:
        print(f'[WARN] No pre-trained encoder at {encoder_path}; training from scratch')

    model.to(device)
    scaler = GradScaler()

    os.makedirs(os.path.join(SAVE_PATH, 'best'), exist_ok=True)
    os.makedirs(os.path.join(SAVE_PATH, 'last'), exist_ok=True)
    ckpt_best = os.path.join(SAVE_PATH, 'best', f'{save_name}.pt')
    ckpt_last = os.path.join(SAVE_PATH, 'last', f'{save_name}.pt')

    # ════════════════════════════════════════════════════════════════
    # STEP 1: freeze BART, warm up EEG encoders at high LR
    # ════════════════════════════════════════════════════════════════
    log('\n' + '=' * 60)
    log(f'STEP 1 — encoder warm-up ({STEP1_EPOCHS} epochs, lr={LR1})')
    log('=' * 60)

    for p in model.pretrained.parameters():
        p.requires_grad = False

    enc_params = (list(model.view_encoders.parameters()) +
                  list(model.global_transformer.parameters()) +
                  [model.view_pos_embed] +
                  list(model.fc1.parameters()))
    total_s1 = (len(train_loader) // GRAD_ACCUM) * STEP1_EPOCHS
    warm_s1  = int(total_s1 * WARMUP_RATIO)
    opt1     = optim.AdamW(enc_params, lr=LR1, weight_decay=0.05)
    sch1     = get_cosine_schedule_with_warmup(opt1, warm_s1, total_s1)

    print(f'[INFO] Step-1 trainable params: {sum(p.numel() for p in enc_params):,}')

    if STEP1_EPOCHS > 0:
        model = train_model(dataloaders, device, model, tokenizer, opt1, sch1, scaler,
                            num_epochs=STEP1_EPOCHS, grad_accum_steps=GRAD_ACCUM,
                            patience=PATIENCE, label_smooth=LABEL_SMOOTH,
                            checkpoint_path_best=ckpt_best.replace('.pt', '_s1.pt'),
                            checkpoint_path_last=ckpt_last.replace('.pt', '_s1.pt'))

    # ════════════════════════════════════════════════════════════════
    # STEP 2: apply LoRA, fine-tune all at low LR
    # ════════════════════════════════════════════════════════════════
    log('\n' + '=' * 60)
    log(f'STEP 2 — LoRA fine-tune ({STEP2_EPOCHS} epochs, enc lr={LR2})')
    log('=' * 60)

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.15,
        target_modules=LORA_TARGETS,
    )
    model.pretrained = get_peft_model(model.pretrained, lora_cfg)
    model.pretrained.print_trainable_parameters()
    model.to(device)

    enc_p2, lora_p2, other_p2 = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if ('view_encoders' in name or 'global_transformer' in name
                or 'fc1' in name or 'view_pos_embed' in name):
            enc_p2.append(param)
        elif 'lora' in name.lower():
            lora_p2.append(param)
        else:
            other_p2.append(param)

    # LoRA adapters need LR >= encoder LR to adapt BART effectively.
    # The previous LR2*0.1 was 10× too slow and is the primary cause of poor free-gen scores.
    opt2 = optim.AdamW([
        {'params': enc_p2,   'lr': LR2 * 0.2},  # protect warmed-up encoder from drift
        {'params': lora_p2,  'lr': LR2 * 2.0},  # LoRA higher than enc: fast BART adaptation
        {'params': other_p2, 'lr': LR2 * 0.1},  # keep pre-trained BART weights stable
    ], weight_decay=0.05)

    total_s2 = (len(train_loader) // GRAD_ACCUM) * STEP2_EPOCHS
    warm_s2  = int(total_s2 * WARMUP_RATIO)
    sch2     = get_cosine_schedule_with_warmup(opt2, warm_s2, total_s2)

    print(f'[INFO] Step-2 trainable: enc={sum(p.numel() for p in enc_p2):,} '
          f'lora={sum(p.numel() for p in lora_p2):,} other={sum(p.numel() for p in other_p2):,}')

    if STEP2_EPOCHS > 0:
        model = train_model(dataloaders, device, model, tokenizer, opt2, sch2, scaler,
                            num_epochs=STEP2_EPOCHS, grad_accum_steps=GRAD_ACCUM,
                            patience=PATIENCE, label_smooth=LABEL_SMOOTH,
                            checkpoint_path_best=ckpt_best,
                            checkpoint_path_last=ckpt_last)

    # ════════════════════════════════════════════════════════════════
    # Merge LoRA → save plain checkpoint (compatible with eval_multiview.py)
    # ════════════════════════════════════════════════════════════════
    log('\n[INFO] Merging LoRA weights into BART...')
    model.pretrained = model.pretrained.merge_and_unload()
    merged_path = ckpt_best.replace('.pt', '_merged.pt')
    torch.save(model.state_dict(), merged_path)
    log(f'[INFO] Merged checkpoint saved: {merged_path}')
