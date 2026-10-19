"""Sentence-ranking evaluation: the parts that need no torch.

The question this evaluation answers is "given this EEG, can the model tell
which sentence was read?". For every test sample the true sentence and K-1
distractor sentences from the test set are scored by the model's per-token NLL
under that EEG; the rank of the true sentence is the result. A model that
ignores the EEG ranks at chance (top-1 = 1/K) no matter how fluent BART is,
because every candidate goes through the same BART.

Everything that decides *which* candidates and *which* shuffled EEG a sample
gets lives here and depends only on (seed, sample index, sentence keys). The
ANN entry (`eval_ranking.py`) and the SNN entry
(`Spiking-EEG2TEXT/EEGSNN/cfg_runner/eval_ranking_snn_cfg.py`) build the same
test set from the same pickles, so both see identical candidates.

Standard library only, so it can be tested without torch.
"""

import math
import random

MODES = ('real', 'rand', 'shuffle')


def unique_keys(keys):
    """Sorted distinct sentence keys (e.g. tuples of target token ids)."""
    return sorted(set(keys))


def candidate_indices(n_unique: int, true_idx: int, k: int, seed: int, sample_idx: int) -> list:
    """Indices into the unique-sentence list: true sentence first, then K-1 distractors.

    Deterministic in (seed, sample_idx). K is capped at n_unique.
    """
    k = min(int(k), n_unique)
    rng = random.Random(seed * 1_000_003 + sample_idx)
    pool = [u for u in range(n_unique) if u != true_idx]
    return [true_idx] + rng.sample(pool, k - 1)


def shuffle_partner(keys, sample_idx: int, seed: int) -> int:
    """Index of another sample whose sentence differs, to borrow its EEG.

    This is the distribution-preserving control: real EEG, wrong sentence.
    """
    rng = random.Random(seed * 7_919 + sample_idx + 1)
    n = len(keys)
    if len(set(keys)) < 2:
        raise ValueError('shuffle needs at least two distinct sentences')
    while True:
        j = rng.randrange(n)
        if keys[j] != keys[sample_idx]:
            return j


def rank_of(true_score: float, distractor_scores) -> int:
    """1-based rank of the true candidate; lower score = better.

    A distractor that ties the true sentence counts as ranked above it, so a
    model that scores everything the same lands at the bottom, not the top.
    """
    return 1 + sum(1 for s in distractor_scores if s <= true_score)


def pick(scores) -> int:
    """Position of the candidate the model picks (lowest score) in `scores`.

    `scores[0]` is the true sentence. Ties go to a distractor, the same rule as
    `rank_of`, so `pick(scores) == 0` exactly when the rank is 1.
    """
    order = list(range(1, len(scores))) + [0]
    return min(order, key=lambda j: scores[j])


def chance_pick(k: int, seed: int, sample_idx: int) -> int:
    """Position a uniform guesser picks among the K candidates (retrieval chance level)."""
    return random.Random(seed * 104_729 + sample_idx).randrange(k)


def summarize(ranks, k: int, nll_sum: float, n_tokens: int) -> dict:
    """Aggregate one mode.

    Args:
        ranks: 1-based rank of the true sentence per sample.
        k: candidates per sample (after capping).
        nll_sum, n_tokens: summed NLL and token count of the *true* sentences,
            for the corpus-level loss / perplexity.
    """
    n = len(ranks)
    if n == 0:
        raise ValueError('no samples')
    top = lambda t: sum(r <= t for r in ranks) / n
    mean_nll = nll_sum / max(n_tokens, 1)
    return {
        'n': n,
        'k': k,
        'top1': top(1),
        'top5': top(5),
        'mrr': sum(1.0 / r for r in ranks) / n,
        'mean_rank': sum(ranks) / n,
        'nll': mean_nll,
        'ppl': math.exp(mean_nll),
        'chance_top1': 1.0 / k,
        'chance_top5': min(5, k) / k,
        'chance_mrr': sum(1.0 / r for r in range(1, k + 1)) / k,
        'ranks': list(ranks),
    }
