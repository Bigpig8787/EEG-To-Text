"""raw_eeg_max_len 參數化的測試。

需要 torch（`ZuCo_dataset` 回傳 tensor），所以只能在訓練機上跑：

    python -m pytest EEG-To-Text/data/tests/ -q
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from data.dataset import RAW_EEG_MAX_LEN, get_input_sample             # noqa: E402


class _StubTokenizer:
    """只回傳固定長度的 ids/mask，避免測試依賴 HuggingFace 下載。"""

    def __call__(self, text, padding=None, max_length=56, truncation=True,
                 return_tensors='pt', return_attention_mask=True):
        import torch
        return {
            'input_ids': torch.zeros(1, max_length, dtype=torch.long),
            'attention_mask': torch.ones(1, max_length, dtype=torch.long),
        }


def _sent(n_samples):
    return {
        'content': 'the quick brown fox',
        'sentence_level_EEG': {f'mean_{b}': np.zeros(105, dtype=np.float32)
                               for b in ['t1', 't2', 'a1', 'a2', 'b1', 'b2', 'g1', 'g2']},
        'word': [{'content': 'the', 'nFixations': 1}],
        'rawData': np.random.randn(105, n_samples).astype(np.float32),
    }


def test_default_is_still_5000_so_existing_callers_do_not_change():
    assert RAW_EEG_MAX_LEN == 5000


def test_raw_views_are_padded_to_the_requested_length():
    sample = get_input_sample(_sent(1500), _StubTokenizer(), raw_eeg_max_len=2000)
    total = sum(v.shape[1] for v in sample['raw_eeg_views'].values())
    assert all(v.shape[1] == 2000 for v in sample['raw_eeg_views'].values())
    assert total == 2000 * len(sample['raw_eeg_views'])


def test_raw_views_are_truncated_to_the_requested_length():
    sample = get_input_sample(_sent(5000), _StubTokenizer(), raw_eeg_max_len=2000)
    assert all(v.shape[1] == 2000 for v in sample['raw_eeg_views'].values())


def test_raw_eeg_len_reports_the_clamped_actual_length():
    sample = get_input_sample(_sent(5000), _StubTokenizer(), raw_eeg_max_len=2000)
    assert sample['raw_eeg_len'] == 2000


def test_raw_eeg_len_reports_the_unpadded_length_when_shorter():
    sample = get_input_sample(_sent(1500), _StubTokenizer(), raw_eeg_max_len=2000)
    assert sample['raw_eeg_len'] == 1500


def test_missing_raw_data_still_emits_zero_views_at_the_requested_length():
    sent = _sent(2000)
    del sent['rawData']
    sample = get_input_sample(sent, _StubTokenizer(), raw_eeg_max_len=2000)
    assert all(v.shape[1] == 2000 for v in sample['raw_eeg_views'].values())
    assert sample['raw_eeg_len'] == 0


def test_region_channel_counts_are_unaffected_by_the_length_change():
    sample = get_input_sample(_sent(2000), _StubTokenizer(), raw_eeg_max_len=2000)
    total_channels = sum(v.shape[0] for v in sample['raw_eeg_views'].values())
    assert total_channels == 105
