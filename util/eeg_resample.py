"""ZuCo raw EEG 的取樣率轉換。

500Hz -> 200Hz 是 x0.4，**不是整數降頻倍率**，所以 `scipy.signal.decimate`
（只吃整數 q）不能用。這裡走 `resample_poly(up=2, down=5)`：內部先升取樣
2 倍到 1000Hz、套 anti-alias FIR、再降 5 倍到 200Hz。

那個 FIR 的截止點自動落在**較低取樣率的 Nyquist = 100Hz**，正是需要的值：
目標 fs=200Hz 只能無失真表示 <100Hz 的成分，100~200Hz 若不先砍掉，降頻後
會折返（aliasing）成 0~100Hz 的假訊號，且事後無法分離。例如 150Hz 會變成
50Hz，直接汙染 b2 頻帶。

本專案用到的最高頻帶是 g2 = 40~49.5Hz，100Hz 截止有 2 倍餘裕。
"""

import math

import numpy as np
from scipy import signal

DEFAULT_FS_IN = 500
DEFAULT_FS_OUT = 200


def resample_ratio(fs_in: int, fs_out: int) -> tuple:
    """回傳 `resample_poly` 要的 `(up, down)`，已約分到最簡分數。

    Args:
        fs_in: 原始取樣率 (Hz)。
        fs_out: 目標取樣率 (Hz)。

    Returns:
        `(up, down)`，滿足 `fs_out / fs_in == up / down`。
    """
    fs_in, fs_out = int(fs_in), int(fs_out)
    if fs_in <= 0 or fs_out <= 0:
        raise ValueError(f'sampling rates must be positive, got {fs_in} -> {fs_out}')
    divisor = math.gcd(fs_in, fs_out)
    return fs_out // divisor, fs_in // divisor


def resampled_length(n_samples: int, fs_in: int = DEFAULT_FS_IN,
                     fs_out: int = DEFAULT_FS_OUT) -> int:
    """`resample_poly` 的輸出長度 = `ceil(n_samples * up / down)`。"""
    up, down = resample_ratio(fs_in, fs_out)
    return math.ceil(int(n_samples) * up / down)


def resample_eeg(raw: np.ndarray, fs_in: int = DEFAULT_FS_IN,
                 fs_out: int = DEFAULT_FS_OUT) -> np.ndarray:
    """把 `(channels, samples)` 的 EEG 重取樣到 `fs_out`。

    Args:
        raw: 形狀 `(channels, samples)` 的陣列，時間在最後一軸。
        fs_in: 原始取樣率 (Hz)。ZuCo 是 500。
        fs_out: 目標取樣率 (Hz)。

    Returns:
        形狀 `(channels, resampled_length(samples, fs_in, fs_out))` 的 float32。

    Raises:
        ValueError: `raw` 不是二維。
    """
    if raw.ndim != 2:
        raise ValueError(
            f'raw must be 2-D (channels, samples), got shape {raw.shape}')

    up, down = resample_ratio(fs_in, fs_out)
    if (up, down) == (1, 1):
        return raw.astype(np.float32, copy=True)

    # float64 運算再降回 float32：FIR 是長濾波器，累加誤差在 float32 下看得出來。
    out = signal.resample_poly(raw.astype(np.float64), up=up, down=down, axis=1)
    return out.astype(np.float32)
