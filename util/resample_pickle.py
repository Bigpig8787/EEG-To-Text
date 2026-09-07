"""把已建好的 ZuCo pickle 從 500Hz 降到 200Hz。

只改 `sent_obj['rawData']`（時序訊號）；word-level 與 sentence-level 的頻帶
特徵是 ZuCo 預先算好的統計量，不是時序，原樣搬過去。

輸入  dataset/ZuCo/<task>/pickle/<task>-dataset.pickle
輸出  dataset/ZuCo/<task>/pickle/<task>-dataset-200hz.pickle

用法：
    python util/resample_pickle.py -t task1-SR
    python util/resample_pickle.py -t task2-NR-2.0 --verify
    python util/resample_pickle.py --all
"""

import argparse
import os
import pickle

import numpy as np
from scipy import signal
from tqdm import tqdm

from eeg_resample import DEFAULT_FS_IN, DEFAULT_FS_OUT, resample_eeg

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

TASKS = ['task1-SR', 'task2-NR', 'task3-TSR', 'task2-NR-2.0', 'task2-TSR-2.0']

VERIFY_BANDS = [('0-40', 0, 40), ('40-90', 40, 90), ('90-100', 90, 100), ('>100', 100, None)]


def resample_dataset_dict(dataset_dict: dict, fs_in: int = DEFAULT_FS_IN,
                          fs_out: int = DEFAULT_FS_OUT, progress: bool = False) -> dict:
    """回傳一份新的 dataset dict，其中每個 `rawData` 都被重取樣過。

    輸入不會被就地修改。`None` 的句子保留 `None`，沒有 `rawData` 的句子原樣
    搬過去 —— `ZuCo_dataset` 的 train/dev/test 是按 index range 切的
    (`data/dataset.py:158-166`)，少一筆或錯位都會讓切分跑掉。

    Args:
        dataset_dict: `{subject: [sent_obj | None, ...]}`。
        fs_in: 原始取樣率 (Hz)。
        fs_out: 目標取樣率 (Hz)。
        progress: 顯示 tqdm 進度條。

    Returns:
        同結構的新 dict。
    """
    out = {}
    for subject, sentences in dataset_dict.items():
        iterator = tqdm(sentences, desc=subject, leave=False) if progress else sentences
        new_sentences = []
        for sent in iterator:
            if sent is None:
                new_sentences.append(None)
                continue

            new_sent = dict(sent)
            raw = sent.get('rawData', None)
            if raw is not None:
                new_sent['rawData'] = resample_eeg(np.asarray(raw), fs_in, fs_out)
            new_sentences.append(new_sent)
        out[subject] = new_sentences
    return out


def verify_report(before: np.ndarray, after: np.ndarray,
                  fs_in: int = DEFAULT_FS_IN, fs_out: int = DEFAULT_FS_OUT) -> dict:
    """比對重取樣前後的長度與各頻帶能量比。

    `band_ratio[name]` = 重取樣後該頻帶能量 / 重取樣前全頻帶能量。
    `>100` 應該趨近 0（那是 200Hz 的 Nyquist 以上，必須被砍掉）；
    `0-40` 應該保留（PSD 是密度，取樣率下降會讓比值大於 1，屬正常）。

    Args:
        before: `(channels, samples)`，重取樣前。
        after: `(channels, samples)`，重取樣後。
        fs_in: `before` 的取樣率 (Hz)。
        fs_out: `after` 的取樣率 (Hz)。

    Returns:
        `{'len_before': int, 'len_after': int, 'band_ratio': {name: float}}`
    """
    def band_energy(x, fs, lo, hi):
        freqs, power = signal.welch(x, fs=fs, nperseg=min(1024, len(x)))
        mask = freqs >= lo
        if hi is not None:
            mask &= freqs < hi
        return float(power[mask].sum())

    ch_before = np.asarray(before)[0]
    ch_after = np.asarray(after)[0]
    total_before = band_energy(ch_before, fs_in, 0, None)
    if total_before == 0:
        total_before = 1.0

    return {
        'len_before': int(np.asarray(before).shape[1]),
        'len_after': int(np.asarray(after).shape[1]),
        'band_ratio': {
            name: band_energy(ch_after, fs_out, lo, hi) / total_before
            for name, lo, hi in VERIFY_BANDS
        },
    }


def _pickle_paths(task_name: str, fs_out: int):
    pickle_dir = os.path.join(PROJECT_ROOT, 'dataset', 'ZuCo', task_name, 'pickle')
    src = os.path.join(pickle_dir, f'{task_name}-dataset.pickle')
    dst = os.path.join(pickle_dir, f'{task_name}-dataset-{fs_out}hz.pickle')
    return src, dst


def _first_raw(dataset_dict):
    """找出第一個有 rawData 的句子，給 --verify 用。"""
    for sentences in dataset_dict.values():
        for sent in sentences:
            if sent is not None and 'rawData' in sent:
                return np.asarray(sent['rawData'])
    return None


def process_task(task_name: str, fs_in: int, fs_out: int, verify: bool) -> None:
    src, dst = _pickle_paths(task_name, fs_out)
    if not os.path.exists(src):
        print(f'[SKIP] {task_name}: not found -> {src}')
        return

    print(f'[{task_name}] loading {src}')
    with open(src, 'rb') as handle:
        dataset_dict = pickle.load(handle)

    sample_before = _first_raw(dataset_dict) if verify else None
    if verify and sample_before is None:
        print(f'[WARN] {task_name}: no sentence carries rawData; '
              f'the resampled pickle will be identical to the source')

    out = resample_dataset_dict(dataset_dict, fs_in, fs_out, progress=True)

    if verify and sample_before is not None:
        report = verify_report(sample_before, _first_raw(out), fs_in, fs_out)
        print(f'  length : {report["len_before"]} -> {report["len_after"]}')
        for name, ratio in report['band_ratio'].items():
            flag = '  <- must be ~0' if name == '>100' else ''
            print(f'  band {name:>7}: {ratio:.6f}{flag}')

    with open(dst, 'wb') as handle:
        pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'[{task_name}] saved {dst}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-t', '--task_name', choices=TASKS,
                        help='single task to convert')
    parser.add_argument('--all', action='store_true', help='convert every task in TASKS')
    parser.add_argument('--fs_in', type=int, default=DEFAULT_FS_IN)
    parser.add_argument('--fs_out', type=int, default=DEFAULT_FS_OUT)
    parser.add_argument('--verify', action='store_true',
                        help='print length and per-band energy for the first sentence')
    args = parser.parse_args()

    if not args.all and not args.task_name:
        parser.error('give -t <task> or --all')

    targets = TASKS if args.all else [args.task_name]
    for task_name in targets:
        process_task(task_name, args.fs_in, args.fs_out, args.verify)


if __name__ == '__main__':
    main()
