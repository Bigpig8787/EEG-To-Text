import os
import json
from typing import Optional, Dict, Any

import torch
from tqdm import tqdm

from metrics import compute_metrics


class Evaluator:
    """Evaluate a trained EEG-to-text model. Standalone or callable as a module.

    Usage (standalone):
        ev = Evaluator.from_checkpoint(ckpt_path, model_class, model_kwargs, tokenizer, device)
        results = ev.evaluate(test_loader, teacher_forcing=False)
        ev.save_results(results, './results/eval.json')

    Usage (in pipeline):
        ev = Evaluator(model, tokenizer, device)
        results = ev.evaluate_all(test_loader)
    """

    GEN_DEFAULTS = dict(max_new_tokens=50, num_beams=5,
                        repetition_penalty=1.5, no_repeat_ngram_size=3,
                        early_stopping=True)

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tok   = tokenizer
        self.device = device
        self.model.to(device)
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, model_class, model_kwargs: dict,
                        tokenizer, device):
        """Load model from a plain state_dict checkpoint (merged, no PEFT needed)."""
        model = model_class(**model_kwargs)
        state = torch.load(checkpoint_path, map_location=device)
        # Support both raw state_dict and wrapped checkpoint dicts
        if isinstance(state, dict) and 'model_state' in state:
            state = state['model_state']
        model.load_state_dict(state)
        print(f'[Evaluator] loaded checkpoint <- {checkpoint_path}')
        return cls(model, tokenizer, device)

    def evaluate(self, dataloader, teacher_forcing: bool = False,
                 use_noise: bool = False, gen_kwargs: Optional[dict] = None) -> dict:
        """Run one evaluation condition, return metrics dict."""
        kw = {**self.GEN_DEFAULTS, **(gen_kwargs or {})}
        forced_bos = self.tok.convert_tokens_to_ids(['<s>'])[0]
        pad_id = self.tok.pad_token_id

        preds, targets = [], []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f'eval tf={teacher_forcing} noise={use_noise}'):
                (_, _, attn_mask, attn_mask_inv,
                 target_ids, _, _, _, raw_views) = batch

                if not raw_views:
                    continue

                attn_mask     = attn_mask.to(self.device)
                attn_mask_inv = attn_mask_inv.to(self.device)
                target_ids    = target_ids.to(self.device)

                if use_noise:
                    view_inputs = {k: torch.randn_like(v).to(self.device)
                                   for k, v in raw_views.items()}
                else:
                    view_inputs = {k: v.to(self.device).float() for k, v in raw_views.items()}

                if teacher_forcing:
                    tf_ids = target_ids.clone()
                    tf_ids[tf_ids == pad_id] = -100
                    out = self.model(view_inputs, attn_mask, attn_mask_inv, tf_ids)
                    pred_ids = out.logits.argmax(-1)
                else:
                    target_ids_for_gen = target_ids.clone()
                    target_ids_for_gen[target_ids_for_gen == pad_id] = -100
                    pred_ids = self.model.generate(
                        view_inputs, attn_mask, attn_mask_inv,
                        target_ids_batch=target_ids_for_gen,
                        forced_bos_token_id=forced_bos, **kw)

                for pred, tgt in zip(pred_ids, target_ids):
                    pred_text = self.tok.decode(pred, skip_special_tokens=True)
                    tgt_text  = self.tok.decode(tgt[tgt != pad_id], skip_special_tokens=True)
                    preds.append(pred_text)
                    targets.append(tgt_text)

        metrics = compute_metrics(preds, targets)
        metrics['n_samples'] = len(preds)
        return metrics

    def evaluate_all(self, dataloader) -> Dict[str, dict]:
        """Run all 4 conditions: (tf, noise) x {True, False}."""
        conditions = {
            'tf_real':   (True,  False),
            'tf_noise':  (True,  True),
            'gen_real':  (False, False),
            'gen_noise': (False, True),
        }
        results = {}
        for name, (tf, noise) in conditions.items():
            print(f'\n[{name}] teacher_forcing={tf}, noise={noise}')
            results[name] = self.evaluate(dataloader, teacher_forcing=tf, use_noise=noise)
            self._print_metrics(results[name])
        return results

    def save_results(self, results: dict, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'[Evaluator] results saved -> {path}')

    @staticmethod
    def _print_metrics(m: dict):
        for k, v in m.items():
            if k != 'n_samples':
                print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')
