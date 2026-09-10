"""Convert ZuCo v1.0 .mat files to .pickle with rawData extraction.

一次 .mat pass 可同時輸出多個採樣率 —— 讀 .mat 是整個流程最慢的一步，
重取樣相對免費，所以預設同時寫 500Hz 與 200Hz 兩份。

    python util/construct_dataset_mat_to_pickle_v1.py \\
        -t task1-SR \\
        --dataset_root /home/jovyan/datasets/soc-20260831131547 \\
        --output_root  /home/jovyan/datasets/soc-20260831131547
"""

import argparse
import os
import pickle

import numpy as np
import scipy.io as io
from tqdm import tqdm

import zuco_paths
from resample_pickle import _atomic_pickle_dump, resample_dataset_dict

BANDS = ['t1', 't2', 'a1', 'a2', 'b1', 'b2', 'g1', 'g2']
V1_TASKS = ('task1-SR', 'task2-NR', 'task3-TSR')


def build_dataset_dict(mat_paths: list, include_answer_eeg: bool = False) -> dict:
    """解析一組 v1.0 `.mat`，回傳 `{subject: [sent_obj | None, ...]}`。

    `include_answer_eeg` 由呼叫端（`convert()`）決定 —— 只有 task1-SR 的
    `.mat` 真的有 `answer_mean_*` 欄位；這支函式只管解析，不該自己猜 task
    語意。就算 `include_answer_eeg=True`，仍用 `hasattr` 當第二道防線，
    這樣一份真的缺欄位的 task1-SR 檔也不會炸。
    """
    dataset_dict = {}
    for mat_file in tqdm(mat_paths):
        subject = os.path.basename(mat_file).split('_')[0].replace('results', '').strip()
        dataset_dict[subject] = []
        matdata = io.loadmat(mat_file, squeeze_me=True,
                             struct_as_record=False)['sentenceData']

        for sent in matdata:
            word_data = sent.word
            if isinstance(word_data, float):
                dataset_dict[subject].append(None)
                continue

            sent_obj = {'content': sent.content}
            sent_obj['sentence_level_EEG'] = {
                'mean_' + b: getattr(sent, 'mean_' + b) for b in BANDS}

            if hasattr(sent, 'rawData') and not isinstance(sent.rawData, float):
                raw = np.array(sent.rawData, dtype=np.float32)
                if raw.ndim == 2 and raw.shape[0] == 105:
                    sent_obj['rawData'] = raw

            if include_answer_eeg and hasattr(sent, 'answer_mean_t1'):
                sent_obj['answer_EEG'] = {
                    'answer_mean_' + b: getattr(sent, 'answer_mean_' + b)
                    for b in BANDS}

            sent_obj['word'] = []
            word_tokens_has_fixation, word_tokens_with_mask, word_tokens_all = [], [], []
            for word in np.atleast_1d(word_data):
                word_tokens_all.append(word.content)
                word_obj = {'content': word.content, 'nFixations': word.nFixations}
                if word.nFixations > 0:
                    word_obj['word_level_EEG'] = {
                        'FFD': {'FFD_' + b: getattr(word, 'FFD_' + b) for b in BANDS},
                        'TRT': {'TRT_' + b: getattr(word, 'TRT_' + b) for b in BANDS},
                        'GD': {'GD_' + b: getattr(word, 'GD_' + b) for b in BANDS},
                    }
                    sent_obj['word'].append(word_obj)
                    word_tokens_has_fixation.append(word.content)
                    word_tokens_with_mask.append(word.content)
                else:
                    word_tokens_with_mask.append('[MASK]')

            sent_obj['word_tokens_has_fixation'] = word_tokens_has_fixation
            sent_obj['word_tokens_with_mask'] = word_tokens_with_mask
            sent_obj['word_tokens_all'] = word_tokens_all
            dataset_dict[subject].append(sent_obj)

    return dataset_dict


def convert(dataset_root: str, output_root: str, task: str,
            fs_out_list: list, verify: bool = False) -> dict:
    """把一個 task 的 `.mat` 轉成 pickle，每個採樣率各一份。

    Args:
        dataset_root: ZuCo 根目錄（底下有 `v1/` `v2/`）。
        output_root: pickle 輸出根目錄。
        task: 正規 task 名，見 `zuco_paths.CANONICAL_TASKS`。
        fs_out_list: 要輸出的採樣率；`None` 代表原始 500Hz。
        verify: 印出重取樣前後的長度對照。

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
                       if f.endswith('.mat'))
    if not mat_paths:
        raise FileNotFoundError('No .mat files under: ' + in_dir)

    print('[{}] {} mat file(s) from {}'.format(task, len(mat_paths), in_dir))
    dataset_dict = build_dataset_dict(
        mat_paths, include_answer_eeg=(task == 'task1-SR'))

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


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-t', '--task_name', choices=V1_TASKS, required=True)
    parser.add_argument('--dataset_root', required=True,
                        help='ZuCo root; the directory containing v1/ and v2/')
    parser.add_argument('--output_root', default=None,
                        help='where the pickle/ dirs go (default: --dataset_root)')
    parser.add_argument('--fs_out', nargs='+', default=['500', '200'],
                        help="sampling rates to write; '500' means the original")
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()

    fs_out_list = [None if v == '500' else int(v) for v in args.fs_out]
    convert(args.dataset_root, args.output_root or args.dataset_root,
            args.task_name, fs_out_list, verify=args.verify)


if __name__ == '__main__':
    main()
