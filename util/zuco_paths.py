"""ZuCo 資料集在磁碟上的目錄名 ↔ 程式碼用的正規名。

磁碟上的目錄名帶空格、而且分 v1/ v2/ 兩層；程式碼從頭到尾用的是不帶空格的
正規名（`task1-SR` 等），因為那五個字串已經被 SNN 端的
`EEGSNN/cfg_runner/pickle_naming.py` 與 `util/resample_pickle.py` 使用。
把兩邊的對照集中在這裡，轉檔器就不必各自散落一份路徑樣板。

刻意只用標準函式庫：這樣它在沒有 numpy/scipy/h5py/torch 的機器上也能被測。
"""

import os
import tempfile

# 正規名 -> (資料集版本目錄, 磁碟上的 task 目錄名)
#
# 磁碟名是實地看過的，不是推測：`task1- SR` 的 '-' 後面有一個空格，
# 另外四個的 '-' 兩側都有空格，而 `task2-TSR-2.0` 的磁碟名是
# `task2 - TSR` —— 少了 `-2.0`。
TASK_LAYOUT = {
    'task1-SR':      ('v1', 'task1- SR'),
    'task2-NR':      ('v1', 'task2 - NR'),
    'task3-TSR':     ('v1', 'task3 - TSR'),
    'task2-NR-2.0':  ('v2', 'task2 - NR-2.0'),
    'task2-TSR-2.0': ('v2', 'task2 - TSR'),
}

CANONICAL_TASKS = ('task1-SR', 'task2-NR', 'task3-TSR',
                   'task2-NR-2.0', 'task2-TSR-2.0')

MAT_SUBDIR = 'Matlab files'


def _layout(task: str):
    try:
        return TASK_LAYOUT[task]
    except KeyError:
        raise KeyError(
            '{!r} is not a ZuCo task. Valid names: {}'.format(
                task, ', '.join(CANONICAL_TASKS))) from None


def mat_dir(dataset_root: str, task: str) -> str:
    """該 task 的 `.mat` 目錄，例如 `<root>/v2/task2 - TSR/Matlab files`。"""
    version, on_disk = _layout(task)
    return os.path.join(dataset_root, version, on_disk, MAT_SUBDIR)


def pickle_dir(output_root: str, task: str) -> str:
    """該 task 的 pickle 輸出目錄。

    刻意用**正規名**且**不帶 v1/v2 層** —— SNN 端的
    `pickle_naming.pickle_file_name()` 假設 task 平鋪在 dataset_root 底下，
    維持這個假設，轉檔完成後 SNN 端一行都不用改。
    """
    return os.path.join(output_root, task, 'pickle')


def pickle_name(task: str, fs_out: int = None) -> str:
    """pickle 檔名。`fs_out=None` 是 500Hz 原始版本。

    `'-dataset-200hz.pickle'` 是 `data.pickle_suffix` 的契約值。
    """
    _layout(task)
    if fs_out is None:
        return '{}-dataset.pickle'.format(task)
    return '{}-dataset-{}hz.pickle'.format(task, fs_out)


def pickle_path(output_root: str, task: str, fs_out: int = None) -> str:
    return os.path.join(pickle_dir(output_root, task), pickle_name(task, fs_out))


def assert_writable(path: str) -> None:
    """確認 `path` 可以寫，必要時把父目錄建出來。

    掛載進來的資料集唯讀是常態。這道檢查要在解析任何 `.mat` 之前跑 ——
    幾十 GB 的轉檔跑了幾小時才在寫檔時發現不能寫，代價太高。
    """
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            '{} is not writable ({}). Pass --output_root pointing somewhere '
            'you can write.'.format(path, exc)) from None
    try:
        with tempfile.NamedTemporaryFile(dir=path, prefix='.writetest-'):
            pass
    except OSError as exc:
        raise RuntimeError(
            '{} is not writable ({}). Mounted datasets are often read-only; '
            'pass --output_root pointing somewhere you can write.'.format(
                path, exc)) from None
