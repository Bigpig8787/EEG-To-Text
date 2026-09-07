"""eeg_resample 的單元測試。

不需要 torch、不需要 GPU、不需要 ZuCo 原始資料：

    python -m pytest EEG-To-Text/util/tests/ -q
"""

import math
import os
import sys

import numpy as np
import pytest
from scipy import signal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eeg_resample import (                                           # noqa: E402
    resample_eeg,
    resample_ratio,
    resampled_length,
)


def _band_energy(x, fs, lo, hi):
    """Welch PSD 在 [lo, hi) 的能量總和。"""
    freqs, power = signal.welch(x, fs=fs, nperseg=min(1024, len(x)))
    mask = (freqs >= lo) & (freqs < hi)
    return power[mask].sum()


def _tone(freq_hz, n_samples, fs):
    t = np.arange(n_samples) / fs
    return np.sin(2 * np.pi * freq_hz * t).astype(np.float32)


# ── 比值換算 ──────────────────────────────────────────────────────────
def test_500_to_200_is_up2_down5():
    assert resample_ratio(500, 200) == (2, 5)


def test_ratio_is_reduced_to_lowest_terms():
    assert resample_ratio(1000, 400) == (2, 5)


def test_equal_rates_are_identity():
    assert resample_ratio(200, 200) == (1, 1)


# ── 長度換算 ──────────────────────────────────────────────────────────
@pytest.mark.parametrize('n_in,expected', [(5000, 2000), (4237, 1695), (731, 293), (1, 1)])
def test_resampled_length_is_ceil_of_the_ratio(n_in, expected):
    assert resampled_length(n_in, 500, 200) == expected


def test_resampled_length_matches_scipy_for_a_ragged_input():
    raw = np.random.randn(105, 4237).astype(np.float32)
    out = resample_eeg(raw)
    assert out.shape[1] == resampled_length(4237, 500, 200)


# ── 形狀與 dtype ──────────────────────────────────────────────────────
def test_channel_count_is_preserved():
    raw = np.random.randn(105, 5000).astype(np.float32)
    assert resample_eeg(raw).shape == (105, 2000)


def test_output_is_float32():
    raw = np.random.randn(105, 5000).astype(np.float64)
    assert resample_eeg(raw).dtype == np.float32


def test_non_2d_input_is_rejected():
    with pytest.raises(ValueError, match='2-D'):
        resample_eeg(np.random.randn(5000).astype(np.float32))


# ── 頻率內容：通帶留下、止帶砍掉 ─────────────────────────────────────
@pytest.mark.parametrize('freq_hz', [10, 40])
def test_passband_tones_survive(freq_hz):
    """40Hz 以下是本專案用到的頻帶（最高 g2 = 40~49.5Hz），必須保留。"""
    raw = _tone(freq_hz, 5000, 500)[None, :]
    out = resample_eeg(raw)
    e_in = _band_energy(raw[0], 500, 0, 250)
    e_out = _band_energy(out[0], 200, 0, 100)
    # PSD 是密度，取樣率從 500 降到 200 會讓密度乘上 2.5；留寬鬆下限即可
    assert e_out / e_in > 2.0


@pytest.mark.parametrize('freq_hz', [150, 200])
def test_stopband_tones_are_removed(freq_hz):
    """>100Hz（200Hz 的 Nyquist）必須在降頻前被砍掉，否則會折返成假低頻。"""
    raw = _tone(freq_hz, 5000, 500)[None, :]
    out = resample_eeg(raw)
    e_in = _band_energy(raw[0], 500, 0, 250)
    e_out = _band_energy(out[0], 200, 0, 100)
    assert e_out / e_in < 1e-3


def test_a_150hz_tone_does_not_alias_into_the_beta_band():
    """150Hz 若沒濾掉會折返到 50Hz。直接檢查 40~60Hz 沒有能量堆積。"""
    raw = _tone(150, 5000, 500)[None, :]
    out = resample_eeg(raw)
    total = _band_energy(out[0], 200, 0, 100)
    beta = _band_energy(out[0], 200, 40, 60)
    assert total < 1e-6 or beta / total < 0.5
