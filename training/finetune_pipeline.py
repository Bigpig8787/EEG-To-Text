import os
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import torch
import torch.optim as optim
from transformers import BartTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, TaskType, get_peft_model

from models.base import BaseEEGModel
from training.trainer import Trainer, TrainerConfig


@dataclass
class FinetuneConfig:
    # Step 1 -- encoder warm-up (BART frozen)
    step1_epochs: int = 20
    lr1: float = 5e-5

    # Step 2 -- LoRA fine-tune
    step2_epochs: int = 30
    lr2: float = 5e-6

    # LoRA
    lora_r: int = 32
    lora_dropout: float = 0.1
    lora_targets: List[str] = field(default_factory=lambda: ['q_proj', 'k_proj', 'v_proj', 'out_proj'])

    # Common
    batch_size: int = 4
    grad_accum_steps: int = 2
    patience: int = 10
    no_early_stop: bool = True
    label_smooth: float = 0.1
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.2
    use_amp: bool = True
    save_every: int = 0

    checkpoint_dir: str = './checkpoints/multiview'
    checkpoint_name: str = 'model'
    pretrained_encoder_path: Optional[str] = None  # path to encoder_best.pt from pretrain

    # Resume from a specific step checkpoint (set either, not both)
    resume_step1: Optional[str] = None
    resume_step2: Optional[str] = None


class FinetunePipeline:
    """Two-step fine-tuning pipeline: encoder warm-up -> LoRA fine-tune -> merge & save.

    Usage:
        pipeline = FinetunePipeline(model, dataloaders, tokenizer, config, device)
        merged_path = pipeline.run()
    """

    def __init__(self, model: BaseEEGModel, dataloaders: Dict, tokenizer,
                 config: FinetuneConfig, device):
        self.model = model
        self.loaders = dataloaders
        self.tok = tokenizer
        self.cfg = config
        self.device = device
        self.model.to(device)

        if config.pretrained_encoder_path:
            self.model.load_pretrained_encoder(config.pretrained_encoder_path)

        os.makedirs(config.checkpoint_dir, exist_ok=True)

    def run(self) -> str:
        """Execute full two-step training. Returns path to merged checkpoint."""
        self._step1()
        self._step2()
        return self._merge_and_save()

    # ── Step 1 ─────────────────────────────────────────────────────────────

    def _step1(self):
        cfg = self.cfg
        if cfg.step1_epochs <= 0:
            print('[Finetune] skipping step 1')
            return

        print('\n' + '=' * 60)
        print(f'STEP 1 -- encoder warm-up ({cfg.step1_epochs} epochs, lr={cfg.lr1})')
        print('=' * 60)

        self.model.freeze_language_model()

        enc_params = (list(self.model.eeg_encoder.parameters()) +
                      (list(self.model.global_transformer.parameters())
                       if self.model.global_transformer else []) +
                      ([self.model.view_pos_embed]
                       if hasattr(self.model, 'view_pos_embed') else []) +
                      list(self.model.fc1.parameters()))

        opt = optim.AdamW(enc_params, lr=cfg.lr1, weight_decay=0.01)
        total_steps = (len(self.loaders['train']) // cfg.grad_accum_steps) * cfg.step1_epochs
        warm_steps  = int(total_steps * cfg.warmup_ratio)
        sch = get_cosine_schedule_with_warmup(opt, warm_steps, total_steps)

        t_cfg = TrainerConfig(
            num_epochs=cfg.step1_epochs, grad_accum_steps=cfg.grad_accum_steps,
            patience=cfg.patience, no_early_stop=cfg.no_early_stop,
            label_smooth=cfg.label_smooth, max_grad_norm=cfg.max_grad_norm,
            use_amp=cfg.use_amp, save_every=cfg.save_every,
            checkpoint_dir=cfg.checkpoint_dir,
            checkpoint_name=cfg.checkpoint_name + '_s1',
            resume_from=cfg.resume_step1,
        )
        trainer = Trainer(self.model, self.loaders, opt, sch, self.tok, t_cfg, self.device)
        self.model = trainer.train()

    # ── Step 2 ─────────────────────────────────────────────────────────────

    def _step2(self):
        cfg = self.cfg
        if cfg.step2_epochs <= 0:
            print('[Finetune] skipping step 2')
            return

        print('\n' + '=' * 60)
        print(f'STEP 2 -- LoRA fine-tune ({cfg.step2_epochs} epochs, lr={cfg.lr2})')
        print('=' * 60)

        # Unfreeze LM then apply LoRA
        self.model.unfreeze_language_model()
        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=cfg.lora_r, lora_alpha=cfg.lora_r,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_targets,
        )
        self.model.pretrained = get_peft_model(self.model.pretrained, lora_cfg)
        self.model.pretrained.print_trainable_parameters()
        self.model.to(self.device)

        # 3-group optimizer
        enc_p, lora_p, other_p = [], [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(k in name for k in ('view_encoders', 'global_transformer', 'fc1',
                                        '_global_transformer', 'eeg_encoder',
                                        'view_pos_embed')):
                enc_p.append(param)
            elif 'lora' in name.lower():
                lora_p.append(param)
            else:
                other_p.append(param)

        opt = optim.AdamW([
            {'params': enc_p,   'lr': cfg.lr2 * 0.2},
            {'params': lora_p,  'lr': cfg.lr2 * 3.0},
            {'params': other_p, 'lr': cfg.lr2 * 0.1},
        ], weight_decay=0.01)

        total_steps = (len(self.loaders['train']) // cfg.grad_accum_steps) * cfg.step2_epochs
        warm_steps  = int(total_steps * cfg.warmup_ratio)
        sch = get_cosine_schedule_with_warmup(opt, warm_steps, total_steps)

        t_cfg = TrainerConfig(
            num_epochs=cfg.step2_epochs, grad_accum_steps=cfg.grad_accum_steps,
            patience=cfg.patience, no_early_stop=cfg.no_early_stop,
            label_smooth=cfg.label_smooth, max_grad_norm=cfg.max_grad_norm,
            use_amp=cfg.use_amp, save_every=cfg.save_every,
            checkpoint_dir=cfg.checkpoint_dir,
            checkpoint_name=cfg.checkpoint_name + '_s2',
            resume_from=cfg.resume_step2,
        )
        trainer = Trainer(self.model, self.loaders, opt, sch, self.tok, t_cfg, self.device)
        self.model = trainer.train()

    # ── Merge & Save ───────────────────────────────────────────────────────

    def _merge_and_save(self) -> str:
        cfg = self.cfg
        merged_name = cfg.checkpoint_name + '_merged.pt'
        merged_path = os.path.join(cfg.checkpoint_dir, 'best', merged_name)

        print('\n[Finetune] merging LoRA weights into BART...')
        if hasattr(self.model.pretrained, 'merge_and_unload'):
            self.model.pretrained = self.model.pretrained.merge_and_unload()

        torch.save(self.model.state_dict(), merged_path)
        print(f'[Finetune] merged checkpoint -> {merged_path}')
        return merged_path
