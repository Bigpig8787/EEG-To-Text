"""`convert()`/`main()` 共用的轉檔骨架。

v1、v2 兩支轉檔器除了「怎麼解析一份 `.mat`」（`build_dataset_dict` /
`_build_sentence`——ZuCo 1.0 與 2.0 的檔案格式真的不同，故意不合併）以外，
其餘約 60 行完全相同：目錄檢查、寫檔迴圈、summary、verify 區塊、以及四個
共用的 CLI flag。這支模組把這些抽出來，讓兩支轉檔器各自只留 task 專屬的
`build_dataset_dict` 與 argparse 的 task 專屬部分（`-t/--task_name` 的
`choices=` 不一樣，留給呼叫端自己加）。

**峰值記憶體**：`run()` 對每個 `fs_out` 都呼叫一次
`resample_dataset_dict(dataset_dict, ...)`，而該函式刻意不就地修改輸入
（見 `resample_pickle.py` 的 `test_input_dict_is_not_mutated`），所以重取樣
呼叫期間，500Hz 的原始 dict 與正在建構的新採樣率 dict 會同時存在於記憶體
裡。以預設的 `--fs_out 500 200` 為例，峰值大約是 500Hz payload 的 1.4 倍
（500Hz 本身的 1.0 + 200Hz 的 0.4）——`task2-NR-2.0` 的 500Hz payload 是
GB 等級，1.4 倍換算下來大約 11 GB 常駐記憶體。多小時解析完才被 OOM-kill、
什麼都沒寫出去，代價非常高。`run()` 會在 500Hz 那份已經落盤、且正在建構
「最後一個」採樣率之後，立刻釋放 500Hz dict 的參照，讓接下來的
`_atomic_pickle_dump` 不必再背一份完整的 500Hz——但重取樣呼叫本身仍然需要
讀完整個來源 dict 才能建出新的一份，所以呼叫期間的瞬時尖峰無法完全避免，
這裡做的只是縮短它存在的時間。
"""

import os
import shutil

import zuco_paths
from resample_pickle import _atomic_pickle_dump, resample_dataset_dict

_BYTES_PER_GB = 1024 ** 3


def _missing_mat_dir_message(dataset_root, task, in_dir):
    """F2：目錄名對不上時，把上一層實際有什麼列出來，讓 operator 一次就能
    修對 `zuco_paths.TASK_LAYOUT`。上一層本身不存在（`--dataset_root` 打錯）
    時要印出清楚的訊息，不能讓 `os.listdir` 直接炸出 traceback。
    """
    version, _on_disk = zuco_paths.TASK_LAYOUT[task]
    version_dir = os.path.join(dataset_root, version)
    try:
        listing = sorted(os.listdir(version_dir))
    except OSError as exc:
        return (
            "No 'Matlab files' dir at {}.\n"
            "{}/ does not exist either ({}).\n"
            "Check --dataset_root.".format(in_dir, version_dir, exc))
    return (
        "No 'Matlab files' dir at {}.\n"
        "{}/ contains: {}\n"
        "Fix util/zuco_paths.py:TASK_LAYOUT if one of those is the right "
        "directory.".format(in_dir, version_dir, listing))


def _print_summary(task, dataset_dict):
    """F3：寫檔之前印出跨所有 subject 聚合的句數/None 數/rawData 數。

    `--verify` 只印第一個帶 rawData 句子的 shape，抓不到「系統性地沒有任何
    句子帶 rawData」這種全軍覆沒的解析失敗——`_v2.py` 的 `_build_sentence`
    對 rawData 用 `except Exception: pass`，錯誤的 reference layout 會靜
    悄悄產出零 rawData 的 pickle，沒有任何訊息。這個彙總印在寫檔迴圈之前，
    讓 operator 能在幾十 GB 寫出去之前 Ctrl-C。
    """
    subjects = list(dataset_dict)
    print('[{}] Subjects: {}'.format(task, subjects))
    if not subjects:
        print('[WARN] [{}] no subjects parsed'.format(task))
        return

    counts = {s: len(dataset_dict[s]) for s in subjects}
    total_sentences = sum(counts.values())
    total_none = sum(
        1 for s in subjects for sent in dataset_dict[s] if sent is None)
    total_raw = sum(
        1 for s in subjects for sent in dataset_dict[s]
        if sent is not None and 'rawData' in sent)

    print('[{}] Sentences ({}): {}'.format(task, subjects[0], total_sentences))
    print('[{}]   None: {}   rawData: {}'.format(task, total_none, total_raw))

    if len(set(counts.values())) > 1:
        print('[WARN] [{}] subjects disagree on sentence count: {}'.format(
            task, counts))
    if total_raw == 0:
        print('[WARN] [{}] zero sentences carry rawData -- check the '
              "'Matlab files' reference layout".format(task))


def _estimate_output_bytes(mat_paths, fs_out_list):
    """粗估這次轉檔會寫出多少位元組，只用來擋「明顯裝不下」的情況。

    拿輸入 `.mat` 檔案的總大小當 500Hz payload 大小的替代量測——解析之前
    手上就只有這個數字。乘上每個要輸出的採樣率相對 500Hz 的比例（500Hz 本身
    算 1.0）近似輸出總量。pickle 的序列化開銷、非 rawData 欄位都沒算進去，
    所以這只是個粗估上限，不是精確計算。
    """
    total_in = 0
    for p in mat_paths:
        try:
            total_in += os.path.getsize(p)
        except OSError:
            pass
    factor = sum(1.0 if fs is None else fs / 500.0 for fs in fs_out_list)
    return total_in * factor


def _check_free_space(out_dir, estimated_bytes):
    """F8：磁碟空間明顯不夠就在解析任何 `.mat` 之前炸掉。

    `_atomic_pickle_dump` 保證磁碟爆滿時不留半成品檔案，但那是善後，不是
    預防——跑了幾小時才在寫檔時發現空間不夠，代價一樣高。量不到可用空間
    （例如這個檔案系統不支援 `shutil.disk_usage`）就放行，這只是個粗估的
    安全網，不該自己變成新的失敗點。
    """
    try:
        usage = shutil.disk_usage(out_dir)
    except OSError:
        return
    if usage.free < estimated_bytes:
        raise RuntimeError(
            'Only {:.1f} GB free at {}, but this run is estimated to need '
            '~{:.1f} GB (rough estimate: input .mat size x number of '
            'sampling rates). Free up space or pass --output_root pointing '
            'somewhere with more room.'.format(
                usage.free / _BYTES_PER_GB, out_dir,
                estimated_bytes / _BYTES_PER_GB))


def add_common_args(parser):
    """兩支轉檔器共用的四個 CLI flag。

    `-t/--task_name` 不在這裡——v1、v2 的 `choices`（`V1_TASKS` /
    `V2_TASKS`）不一樣，留給呼叫端各自加。
    """
    parser.add_argument('--dataset_root', required=True,
                        help='ZuCo root; the directory containing v1/ and v2/')
    parser.add_argument('--output_root', default=None,
                        help='where the pickle/ dirs go (default: --dataset_root)')
    parser.add_argument('--fs_out', nargs='+', default=['500', '200'],
                        help="sampling rates to write; '500' means the original")
    parser.add_argument('--verify', action='store_true')


def run(dataset_root, output_root, task, fs_out_list, verify, mat_filter,
       build_fn, mat_filter_desc='.mat'):
    """把一個 task 的 `.mat` 轉成 pickle，每個採樣率各一份。

    這是 `construct_dataset_mat_to_pickle_v1.convert()` /
    `_v2.convert()` 共用的骨架；格式專屬的解析邏輯完全在 `build_fn` 裡。

    Args:
        dataset_root: ZuCo 根目錄（底下有 `v1/` `v2/`）。
        output_root: pickle 輸出根目錄。
        task: 正規 task 名，見 `zuco_paths.CANONICAL_TASKS`。
        fs_out_list: 要輸出的採樣率；`None` 代表原始 500Hz。
        verify: 印出重取樣前後的長度對照。
        mat_filter: `(filename: str) -> bool`，決定 `Matlab files/` 目錄下
            哪些檔案要被讀。
        build_fn: `(mat_paths: list) -> dataset_dict`，該版本專屬的解析
            邏輯；呼叫端已經用 closure 綁好 task 專屬的額外參數（例如 v1 的
            `include_answer_eeg`）。
        mat_filter_desc: `mat_filter` 篩不出任何檔案時，錯誤訊息裡怎麼描述
            這個 filter。

    Returns:
        `{採樣率: 寫出的路徑}`，500Hz 的 key 是 `None`。
    """
    in_dir = zuco_paths.mat_dir(dataset_root, task)
    if not os.path.isdir(in_dir):
        raise FileNotFoundError(
            _missing_mat_dir_message(dataset_root, task, in_dir))

    out_dir = zuco_paths.pickle_dir(output_root, task)
    # 先確認寫得了再解析 —— 幾十 GB 跑完才發現唯讀，代價太高。
    zuco_paths.assert_writable(out_dir)

    mat_paths = sorted(os.path.join(in_dir, f) for f in os.listdir(in_dir)
                       if mat_filter(f))
    if not mat_paths:
        raise FileNotFoundError(
            'No {} files under: {}'.format(mat_filter_desc, in_dir))

    print('[{}] {} mat file(s) from {}'.format(task, len(mat_paths), in_dir))
    _check_free_space(out_dir, _estimate_output_bytes(mat_paths, fs_out_list))

    dataset_dict = build_fn(mat_paths)
    _print_summary(task, dataset_dict)

    # verify 用的樣本要在寫檔迴圈之前就先抓起來——迴圈裡可能會在寫完最後一
    # 個採樣率之前就釋放 500Hz dict（見上面峰值記憶體的說明），迴圈跑完才去
    # 翻 `dataset_dict` 會拿到已經被釋放的名字。
    verify_shape, verify_missing = None, False
    if verify:
        subjects = list(dataset_dict)
        sample = None
        if subjects:
            sample = next(
                (s for s in dataset_dict[subjects[0]] if s and 'rawData' in s),
                None)
        if sample is None:
            verify_missing = True
        else:
            verify_shape = sample['rawData'].shape

    written = {}
    last_idx = len(fs_out_list) - 1
    wrote_500hz = None in fs_out_list
    for i, fs_out in enumerate(fs_out_list):
        if fs_out is None:
            payload = dataset_dict
        else:
            payload = resample_dataset_dict(dataset_dict, 500, fs_out,
                                            progress=True)
            if i == last_idx and wrote_500hz:
                # 500Hz 那份這時已經落盤——提前釋放參照，讓接下來的
                # _atomic_pickle_dump 不必再背一份完整的 500Hz dict。
                del dataset_dict
        dst = zuco_paths.pickle_path(output_root, task, fs_out)
        _atomic_pickle_dump(payload, dst)
        written[fs_out] = dst
        label = '500Hz' if fs_out is None else '{}Hz'.format(fs_out)
        print('[{}] {} -> {}'.format(task, label, dst))

    if verify:
        if verify_missing:
            print('[WARN] {}: no sentence carries rawData'.format(task))
        else:
            print('[{}] rawData shape at 500Hz: {}'.format(task, verify_shape))

    return written
