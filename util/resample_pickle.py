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
import tempfile

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


def _pickle_paths(task_name: str, fs_out: int, project_root: str = PROJECT_ROOT):
    """算出 `task_name` 的來源/目的 pickle 路徑。

    `project_root` 可覆寫（測試用合成的 tmp 目錄），預設是本檔案所在的
    `EEG-To-Text/` repo 根目錄。
    """
    pickle_dir = os.path.join(project_root, 'dataset', 'ZuCo', task_name, 'pickle')
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


def _atomic_pickle_dump(obj, dst: str) -> None:
    """把 `obj` 寫進 `dst`，中途失敗不會在 `dst` 留下半成品檔案。

    直接 `open(dst, 'wb')` 一開檔就把舊檔案截斷了 —— 幾 GB 的 dump 跑到一半
    當機或被中斷，`dst` 這個路徑上就會躺著一個讀不回來的殘缺 pickle，而且
    檔名跟正常輸出一模一樣，沒有任何標記說它是壞的。

    改成先寫到同一個目錄下的暫存檔，dump 完全成功才用 `os.replace` 換過去
    （`os.replace` 只在同一個檔案系統內才是原子操作，所以暫存檔必須跟
    `dst` 同目錄，不能放到系統預設的 temp 資料夾）。dump 途中若拋例外，
    暫存檔會被刪掉，不會留垃圾在資料夾裡。
    """
    directory = os.path.dirname(dst) or '.'
    fd, tmp_path = tempfile.mkstemp(
        dir=directory, prefix=os.path.basename(dst) + '.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, dst)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def process_task(task_name: str, fs_in: int, fs_out: int, verify: bool,
                 project_root: str = PROJECT_ROOT) -> None:
    src, dst = _pickle_paths(task_name, fs_out, project_root)
    if dst == src:
        # 不該發生：`_pickle_paths` 的 dst 一定帶 `-{fs_out}hz` 後綴。留這道檢查是
        # 因為一旦這裡的路徑樣板被改壞，後果是原始資料被覆寫，代價太高不能只靠測試擋。
        raise RuntimeError(
            f'{task_name}: resolved destination equals source ({dst}); '
            f'refusing to write, this would destroy the original pickle')

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

    existed_before = os.path.exists(dst)
    _atomic_pickle_dump(out, dst)
    verb = 'overwrote' if existed_before else 'saved'
    print(f'[{task_name}] {verb} {dst}')


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
