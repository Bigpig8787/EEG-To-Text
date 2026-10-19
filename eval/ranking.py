"""Sentence-ranking evaluation: the torch side, shared by ANN and SNN.

See `eval/ranking_core.py` for what is measured and why. The model-specific
part is one callable, `encode_fn(view_inputs, raw_eeg_lens) -> (embeds, mask)`,
which returns what the model feeds to BART as `inputs_embeds` /
`attention_mask` (and resets SNN state itself if it needs to).
"""

import torch
from tqdm import tqdm
from transformers.modeling_outputs import BaseModelOutput

from eval.ranking_core import (MODES, candidate_indices, rank_of,
                               shuffle_partner, summarize, unique_keys)


def _sample(dataset, i, device):
    item = dataset[i]
    target_ids, raw_views, raw_len = item[4], item[8], item[9]
    views = {k: v.unsqueeze(0).to(device).float() for k, v in raw_views.items()}
    lens = torch.as_tensor([int(raw_len)], device=device)
    return target_ids, views, lens


@torch.no_grad()
def candidate_nll(bart, embeds, attn_mask, labels, chunk=25):
    """Summed NLL and token count of each candidate under one EEG encoding.

    BART's encoder runs once; only the decoder sees the K candidates.

    Args:
        bart: BartForConditionalGeneration (merged, no LoRA wrappers).
        embeds, attn_mask: (1, L, D) / (1, L) from `encode_fn`.
        labels: (K, T) target ids with pad already set to -100.

    Returns:
        (nll_sum (K,), n_tokens (K,)) as float / long CPU tensors.
    """
    enc = bart.get_encoder()(inputs_embeds=embeds, attention_mask=attn_mask,
                             return_dict=True).last_hidden_state
    sums, counts = [], []
    for s in range(0, labels.size(0), chunk):
        lab = labels[s:s + chunk]
        n = lab.size(0)
        out = bart(encoder_outputs=BaseModelOutput(last_hidden_state=enc.expand(n, -1, -1)),
                   attention_mask=attn_mask.expand(n, -1), labels=lab, return_dict=True)
        logp = out.logits.float().log_softmax(-1)
        valid = lab != -100
        tok = logp.gather(-1, lab.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        sums.append(-(tok * valid).sum(1).cpu())
        counts.append(valid.sum(1).cpu())
    return torch.cat(sums), torch.cat(counts)


def run_ranking(dataset, encode_fn, bart, tokenizer, device, modes=MODES, k=50,
                seed=312, n_samples=0, chunk=25):
    """Rank the true sentence among K test-set candidates, per input mode.

    Modes:
        real     the sample's own EEG
        rand     `torch.rand_like` — the same control as `eval_multiview.py -n True`
        shuffle  EEG of another test sample with a different sentence
                 (distribution-preserving; the stricter control)

    Candidates are ranked by NLL per token (length-normalised), so short
    sentences do not win just by being short.

    Returns:
        {mode: summary dict from `ranking_core.summarize`}
    """
    for m in modes:
        if m not in MODES:
            raise ValueError(f'unknown mode {m!r}; choose from {MODES}')

    # Every sample with raw EEG, in dataset order; eval_multiview skips the rest too.
    idx = [i for i in range(len(dataset)) if dataset[i][8]]
    if n_samples:
        idx = idx[:n_samples]
    keys = [tuple(dataset[i][4].tolist()) for i in idx]
    uniq = unique_keys(keys)
    pos = {key: u for u, key in enumerate(uniq)}
    labels_all = torch.tensor(uniq, dtype=torch.long)
    labels_all[labels_all == tokenizer.pad_token_id] = -100
    k = min(k, len(uniq))
    print(f'[ranking] samples={len(idx)} unique sentences={len(uniq)} K={k} modes={list(modes)}')

    results = {}
    for mode in modes:
        ranks, nll_sum, n_tok = [], 0.0, 0
        for local, i in enumerate(tqdm(idx, desc=f'rank[{mode}]')):
            _, views, lens = _sample(dataset, i, device)
            if mode == 'rand':
                views = {r: torch.rand_like(v) for r, v in views.items()}
            elif mode == 'shuffle':
                j = idx[shuffle_partner(keys, local, seed)]
                _, views, lens = _sample(dataset, j, device)
            embeds, attn = encode_fn(views, lens)
            cand = candidate_indices(len(uniq), pos[keys[local]], k, seed, local)
            sums, counts = candidate_nll(bart, embeds, attn, labels_all[cand].to(device), chunk)
            score = (sums / counts.clamp(min=1)).tolist()
            ranks.append(rank_of(score[0], score[1:]))
            nll_sum += float(sums[0])
            n_tok += int(counts[0])
        results[mode] = summarize(ranks, k, nll_sum, n_tok)
    return results


def format_table(results) -> str:
    rows = [f'{"mode":<9}{"top1":>8}{"top5":>8}{"MRR":>8}{"rank":>8}{"NLL":>8}{"PPL":>9}']
    for mode, r in results.items():
        rows.append(f'{mode:<9}{r["top1"] * 100:7.2f}%{r["top5"] * 100:7.2f}%{r["mrr"]:8.4f}'
                    f'{r["mean_rank"]:8.2f}{r["nll"]:8.4f}{r["ppl"]:9.2f}')
    any_r = next(iter(results.values()))
    rows.append(f'{"chance":<9}{any_r["chance_top1"] * 100:7.2f}%{any_r["chance_top5"] * 100:7.2f}%'
                f'{any_r["chance_mrr"]:8.4f}{(any_r["k"] + 1) / 2:8.2f}')
    return '\n'.join(rows)
