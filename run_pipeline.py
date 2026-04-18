"""
run_pipeline.py -- complete EEG-to-text training & evaluation pipeline.

Usage:
    python run_pipeline.py --task task1_task2_taskNRv2 --cuda cuda:0
    python run_pipeline.py --skip_pretrain --resume_step2 ./checkpoints/multiview/last/model_s2.pt
"""
import argparse
import json
import os

import torch
from transformers import BartTokenizer

from data.channel_mapping import BRAIN_REGION_CHANNEL_COUNT
from data.pipeline import EEGDataPipeline, PICKLE_PATHS, TASK_KEYS
from models.multiview import MultiViewConformerTranslator
from training.pretrain_pipeline import PretrainPipeline, PretrainConfig
from training.finetune_pipeline import FinetunePipeline, FinetuneConfig
from eval.evaluator import Evaluator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--task',        default='task1_task2_taskNRv2')
    p.add_argument('--subject',     default='ALL')
    p.add_argument('--eeg_type',    default='GD')
    p.add_argument('--batch_size',  type=int, default=4)
    p.add_argument('--cuda',        default='cuda:0')

    # Pretrain
    p.add_argument('--skip_pretrain', action='store_true')
    p.add_argument('--pretrain_epochs', type=int, default=50)
    p.add_argument('--pretrained_encoder', default=None,
                   help='path to existing encoder_best.pt (skips pretrain)')

    # Finetune
    p.add_argument('--step1_epochs', type=int, default=20)
    p.add_argument('--step2_epochs', type=int, default=30)
    p.add_argument('--lr1',  type=float, default=5e-5)
    p.add_argument('--lr2',  type=float, default=5e-6)
    p.add_argument('--lora_r', type=int, default=32)
    p.add_argument('--no_early_stop', action='store_true')
    p.add_argument('--label_smooth', type=float, default=0.1)
    p.add_argument('--resume_step1', default=None)
    p.add_argument('--resume_step2', default=None)
    p.add_argument('--checkpoint_dir', default='./checkpoints/pipeline')
    p.add_argument('--checkpoint_name', default='multiview')

    # Eval only
    p.add_argument('--eval_only',   action='store_true')
    p.add_argument('--eval_ckpt',   default=None, help='merged checkpoint for eval-only mode')

    return p.parse_args()


def build_model(device):
    """Construct MultiViewConformerTranslator from channel map."""
    from transformers import BartForConditionalGeneration
    pretrained_bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    return MultiViewConformerTranslator(
        pretrained_bart=pretrained_bart,
        d_model=512, n_filters=40, temporal_kernel=25, pool_stride=10,
        tokens_per_view=100, n_heads=8, n_encoder_layers=4, n_global_layers=3,
        dropout=0.1, decoder_embedding_size=1024,
    ).to(device)


def main():
    args = parse_args()
    device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    # ── Data ──────────────────────────────────────────────────────────────
    data_pipeline = EEGDataPipeline(
        task_name=args.task, subject=args.subject,
        eeg_type=args.eeg_type, batch_size=args.batch_size, tokenizer=tokenizer,
    )
    loaders = data_pipeline.build()

    if args.eval_only:
        # ── Eval-only ──────────────────────────────────────────────────────
        assert args.eval_ckpt, '--eval_ckpt required in --eval_only mode'
        model = build_model(device)
        ev = Evaluator.from_checkpoint(
            args.eval_ckpt, type(model),
            {'pretrained_bart': model.pretrained},
            tokenizer, device,
        )
        results = ev.evaluate_all(loaders['test'])
        out_path = os.path.join(args.checkpoint_dir, 'eval_results.json')
        ev.save_results(results, out_path)
        return

    # ── Pre-training ───────────────────────────────────────────────────────
    encoder_path = args.pretrained_encoder
    if not args.skip_pretrain and encoder_path is None:
        pickle_paths = [PICKLE_PATHS[k] for k in TASK_KEYS[args.task]]
        pretrain_cfg = PretrainConfig(
            num_epochs=args.pretrain_epochs,
            batch_size=args.batch_size,
            save_path=os.path.join(args.checkpoint_dir, 'pretrain'),
        )
        pretrain_pipeline = PretrainPipeline(pickle_paths, pretrain_cfg, device)
        encoder_path = pretrain_pipeline.run()

    # ── Fine-tuning ────────────────────────────────────────────────────────
    model = build_model(device)

    ft_cfg = FinetuneConfig(
        step1_epochs=args.step1_epochs, lr1=args.lr1,
        step2_epochs=args.step2_epochs, lr2=args.lr2,
        lora_r=args.lora_r,
        batch_size=args.batch_size,
        no_early_stop=args.no_early_stop,
        label_smooth=args.label_smooth,
        checkpoint_dir=os.path.join(args.checkpoint_dir, 'finetune'),
        checkpoint_name=args.checkpoint_name,
        pretrained_encoder_path=encoder_path,
        resume_step1=args.resume_step1,
        resume_step2=args.resume_step2,
    )
    ft_pipeline = FinetunePipeline(model, loaders, tokenizer, ft_cfg, device)
    merged_path = ft_pipeline.run()

    # ── Evaluation ─────────────────────────────────────────────────────────
    print('\n' + '=' * 60 + '\nFINAL EVALUATION\n' + '=' * 60)
    ev = Evaluator(ft_pipeline.model, tokenizer, device)
    results = ev.evaluate_all(loaders['test'])
    out_path = os.path.join(args.checkpoint_dir, 'eval_results.json')
    ev.save_results(results, out_path)
    print(f'\nDone. Merged checkpoint: {merged_path}')
    print(f'Eval results: {out_path}')


if __name__ == '__main__':
    main()
