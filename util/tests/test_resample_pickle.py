"""resample_pickle 的單元測試。用合成的 dataset dict，不需要真的 ZuCo 資料。

    python -m pytest EEG-To-Text/util/tests/ -q
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resample_pickle import resample_dataset_dict, verify_report      # noqa: E402


def _sent(n_samples=5000, with_raw=True):
    """最小的 sent_obj，欄位名與 construct_dataset_mat_to_pickle_v1.py 一致。"""
    obj = {
        'content': 'the quick brown fox',
        'sentence_level_EEG': {'mean_t1': np.zeros(105, dtype=np.float32)},
        'word': [{'content': 'the', 'nFixations': 1}],
        'word_tokens_all': ['the', 'quick', 'brown', 'fox'],
        'word_tokens_has_fixation': ['the'],
        'word_tokens_with_mask': ['the', '[MASK]', '[MASK]', '[MASK]'],
    }
    if with_raw:
        obj['rawData'] = np.random.randn(105, n_samples).astype(np.float32)
    return obj


# ── rawData 被重取樣 ──────────────────────────────────────────────────
def test_raw_data_is_resampled_to_the_target_length():
    d = {'ZAB': [_sent(5000)]}
    out = resample_dataset_dict(d, 500, 200)
    assert out['ZAB'][0]['rawData'].shape == (105, 2000)


def test_raw_data_stays_float32():
    d = {'ZAB': [_sent(5000)]}
    out = resample_dataset_dict(d, 500, 200)
    assert out['ZAB'][0]['rawData'].dtype == np.float32


# ── 其餘欄位原樣保留 ──────────────────────────────────────────────────
def test_non_raw_fields_are_carried_over_untouched():
    d = {'ZAB': [_sent(5000)]}
    out = resample_dataset_dict(d, 500, 200)
    got = out['ZAB'][0]
    assert got['content'] == 'the quick brown fox'
    assert got['word_tokens_all'] == ['the', 'quick', 'brown', 'fox']
    assert got['word'][0]['nFixations'] == 1


def test_sentence_level_band_features_are_not_resampled():
    """頻帶特徵是 ZuCo 預先算好的統計量，不是時序訊號，長度必須不動。"""
    d = {'ZAB': [_sent(5000)]}
    out = resample_dataset_dict(d, 500, 200)
    assert out['ZAB'][0]['sentence_level_EEG']['mean_t1'].shape == (105,)


# ── 索引對齊（ZuCo_dataset 按 index range 切 train/dev/test）──────────
def test_none_entries_keep_their_index():
    d = {'ZAB': [_sent(5000), None, _sent(5000)]}
    out = resample_dataset_dict(d, 500, 200)
    assert len(out['ZAB']) == 3
    assert out['ZAB'][1] is None


def test_sentences_without_raw_data_are_kept():
    d = {'ZAB': [_sent(5000, with_raw=False)]}
    out = resample_dataset_dict(d, 500, 200)
    assert len(out['ZAB']) == 1
    assert 'rawData' not in out['ZAB'][0]


def test_subject_keys_and_order_are_preserved():
    d = {'ZAB': [_sent(5000)], 'ZDM': [_sent(5000)]}
    out = resample_dataset_dict(d, 500, 200)
    assert list(out.keys()) == ['ZAB', 'ZDM']


def test_input_dict_is_not_mutated():
    d = {'ZAB': [_sent(5000)]}
    resample_dataset_dict(d, 500, 200)
    assert d['ZAB'][0]['rawData'].shape == (105, 5000)


# ── ragged 長度 ───────────────────────────────────────────────────────
def test_short_sentences_resample_without_error():
    d = {'ZAB': [_sent(731)]}
    out = resample_dataset_dict(d, 500, 200)
    assert out['ZAB'][0]['rawData'].shape == (105, 293)


# ── verify 報表 ───────────────────────────────────────────────────────
def test_verify_report_records_both_lengths():
    before = np.random.randn(105, 5000).astype(np.float32)
    after = np.random.randn(105, 2000).astype(np.float32)
    rep = verify_report(before, after, 500, 200)
    assert rep['len_before'] == 5000
    assert rep['len_after'] == 2000


def test_verify_report_shows_the_stopband_is_gone():
    t = np.arange(5000) / 500
    before = np.sin(2 * np.pi * 150 * t).astype(np.float32)[None, :].repeat(105, axis=0)
    from eeg_resample import resample_eeg
    after = resample_eeg(before, 500, 200)
    rep = verify_report(before, after, 500, 200)
    assert rep['band_ratio']['>100'] < 1e-3
