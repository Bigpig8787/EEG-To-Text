"""`convert()`/`main()` 共用的轉檔骨架。

v1、v2 兩支轉檔器除了「怎麼解析一份 `.mat`」（`build_dataset_dict` /
`_build_sentence`——ZuCo 1.0 與 2.0 的檔案格式真的不同，故意不合併）以外，
其餘約 60 行完全相同：目錄檢查、寫檔迴圈、verify 區塊、以及四個共用的 CLI
flag。這支模組把這些抽出來，讓兩支轉檔器各自只留 task 專屬的
`build_dataset_dict` 與 argparse 的 task 專屬部分（`-t/--task_name` 的
`choices=` 不一樣，留給呼叫端自己加）。
"""

import os

import zuco_paths
from resample_pickle import _atomic_pickle_dump, resample_dataset_dict


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
        raise FileNotFoundError('No Matlab files directory at: ' + in_dir)

    out_dir = zuco_paths.pickle_dir(output_root, task)
    # 先確認寫得了再解析 —— 幾十 GB 跑完才發現唯讀，代價太高。
    zuco_paths.assert_writable(out_dir)

    mat_paths = sorted(os.path.join(in_dir, f) for f in os.listdir(in_dir)
                       if mat_filter(f))
    if not mat_paths:
        raise FileNotFoundError(
            'No {} files under: {}'.format(mat_filter_desc, in_dir))

    print('[{}] {} mat file(s) from {}'.format(task, len(mat_paths), in_dir))
    dataset_dict = build_fn(mat_paths)

    written = {}
    for fs_out in fs_out_list:
        if fs_out is None:
            payload = dataset_dict
        else:
            payload = resample_dataset_dict(dataset_dict, 500, fs_out,
                                            progress=True)
        dst = zuco_paths.pickle_path(output_root, task, fs_out)
        _atomic_pickle_dump(payload, dst)
        written[fs_out] = dst
        label = '500Hz' if fs_out is None else '{}Hz'.format(fs_out)
        print('[{}] {} -> {}'.format(task, label, dst))

    if verify:
        subjects = list(dataset_dict)
        if subjects:
            first = dataset_dict[subjects[0]]
            sample = next((s for s in first if s and 'rawData' in s), None)
            if sample is None:
                print('[WARN] {}: no sentence carries rawData'.format(task))
            else:
                print('[{}] rawData shape at 500Hz: {}'.format(
                    task, sample['rawData'].shape))

    return written
