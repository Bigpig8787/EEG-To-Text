"""`eval/ranking_core.py`. Standard library only.

    python -m pytest eval/tests/test_ranking_core.py -q
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from eval.ranking_core import (candidate_indices, chance_pick, pick, rank_of,  # noqa: E402
                               shuffle_partner, summarize, unique_keys)


def test_candidates_put_the_true_sentence_first_and_are_distinct():
    c = candidate_indices(100, 7, 50, seed=312, sample_idx=3)
    assert c[0] == 7 and len(c) == 50 and len(set(c)) == 50


def test_candidates_are_deterministic_and_differ_per_sample():
    a = candidate_indices(100, 7, 50, 312, 3)
    assert a == candidate_indices(100, 7, 50, 312, 3)
    assert a != candidate_indices(100, 7, 50, 312, 4)
    assert a != candidate_indices(100, 7, 50, 313, 3)


def test_k_is_capped_at_the_number_of_sentences():
    assert sorted(candidate_indices(5, 2, 50, 312, 0)) == [0, 1, 2, 3, 4]


def test_unique_keys_sorted_and_deduplicated():
    assert unique_keys([(2,), (1,), (2,)]) == [(1,), (2,)]


def test_shuffle_partner_reads_a_different_sentence():
    keys = ['a', 'a', 'b', 'c', 'a']
    for i in range(len(keys)):
        j = shuffle_partner(keys, i, 312)
        assert keys[j] != keys[i]
        assert j == shuffle_partner(keys, i, 312)


def test_shuffle_partner_needs_two_sentences():
    with pytest.raises(ValueError):
        shuffle_partner(['a', 'a'], 0, 312)


def test_rank_of_lower_is_better_and_ties_count_against():
    assert rank_of(1.0, [2.0, 3.0]) == 1
    assert rank_of(2.5, [2.0, 3.0]) == 2
    assert rank_of(2.0, [2.0, 2.0]) == 3


def test_summarize_metrics_and_chance():
    r = summarize([1, 2, 10], k=10, nll_sum=6.0, n_tokens=3)
    assert r['top1'] == pytest.approx(1 / 3)
    assert r['top5'] == pytest.approx(2 / 3)
    assert r['mrr'] == pytest.approx((1 + 0.5 + 0.1) / 3)
    assert r['nll'] == pytest.approx(2.0)
    assert r['ppl'] == pytest.approx(math.exp(2.0))
    assert r['chance_top1'] == pytest.approx(0.1)
    assert r['chance_top5'] == pytest.approx(0.5)
    assert r['ranks'] == [1, 2, 10]


def test_pick_agrees_with_rank_and_gives_ties_to_a_distractor():
    assert pick([1.0, 2.0, 3.0]) == 0 and rank_of(1.0, [2.0, 3.0]) == 1
    assert pick([2.0, 1.0, 3.0]) == 1
    assert pick([1.0, 3.0, 1.0]) == 2 and rank_of(1.0, [3.0, 1.0]) == 2


def test_chance_pick_is_deterministic_and_in_range():
    got = [chance_pick(50, 312, i) for i in range(200)]
    assert got == [chance_pick(50, 312, i) for i in range(200)]
    assert all(0 <= g < 50 for g in got) and len(set(got)) > 20
