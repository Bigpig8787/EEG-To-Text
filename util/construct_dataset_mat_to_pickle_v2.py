"""Convert ZuCo v2.0 .mat files to .pickle with rawData extraction.

ZuCo 2.0 的 .mat 是 v7.3（HDF5），要用 h5py 讀，`sentenceData` 底下每一格
都是 object reference。rawData 存成 (T, 105)，需要轉置成 (105, T)。

一次 .mat pass 可同時輸出多個採樣率 —— 讀 .mat 是整個流程最慢的一步，
重取樣相對免費，所以預設同時寫 500Hz 與 200Hz 兩份。

    python util/construct_dataset_mat_to_pickle_v2.py \
        -t task2-NR-2.0 \
        --dataset_root /home/jovyan/datasets/soc-20260831131547 \
        --output_root  /home/jovyan/datasets/soc-20260831131547
"""

import argparse
import os

import h5py
import numpy as np
from tqdm import tqdm

import data_loading_helpers_modified as dh
import zuco_convert

BANDS = ['t1', 't2', 'a1', 'a2', 'b1', 'b2', 'g1', 'g2']
V2_TASKS = ('task2-NR-2.0', 'task2-TSR-2.0')
SKIP_SUBJECTS = ('YMH',)

# 磁碟上的 mat 檔名是 results<SUBJ>_NR.mat / results<SUBJ>_TSR.mat。
# 同一個 Matlab files/ 目錄可能兩種都有，所以要靠後綴過濾。
MAT_SUFFIX = {'task2-NR-2.0': 'NR.mat', 'task2-TSR-2.0': 'TSR.mat'}


def _build_sentence(f, idx, rawData, contentData, wordData, mean_objs):
    """把 `sentenceData[idx]` 組成一個 sent_obj；該句無效時回傳 `None`。"""
    sent_obj = {'content': dh.load_matlab_string(f[contentData[idx][0]])}
    sent_obj['sentence_level_EEG'] = {
        'mean_' + b: np.squeeze(np.array(f[mean_objs['mean_' + b][idx][0]]))
        for b in BANDS}

    # raw EEG：v2 存成 (T, 105)，轉成 (105, T)
    try:
        raw = np.array(f[rawData[idx][0]], dtype=np.float32)
        if raw.ndim == 2:
            if raw.shape[1] == 105:
                raw = raw.T
            if raw.shape[0] == 105:
                sent_obj['rawData'] = raw
    except Exception:
        pass

    sent_obj['word'] = []
    word_data, word_tokens_all, word_tokens_has_fixation, word_tokens_with_mask = \
        dh.extract_word_level_data(f, f[wordData[idx][0]])

    # 這條 early-return 是原本迴圈裡的 `continue`，逐字保留：
    # 沒有 word-level 資料的句子要記成 None，不能整句漏掉，否則句數對不上。
    if word_data == {} or len(word_tokens_all) == 0:
        return None

    for widx in range(len(word_data)):
        d = word_data[widx]
        word_obj = {'content': d['content'], 'nFixations': d['nFix']}
        if 'GD_EEG' in d:
            gd, ffd, trt = d['GD_EEG'], d['FFD_EEG'], d['TRT_EEG']
            word_obj['word_level_EEG'] = {
                'GD': {'GD_' + b: gd[i] for i, b in enumerate(BANDS)},
                'FFD': {'FFD_' + b: ffd[i] for i, b in enumerate(BANDS)},
                'TRT': {'TRT_' + b: trt[i] for i, b in enumerate(BANDS)},
            }
            sent_obj['word'].append(word_obj)

    sent_obj['word_tokens_has_fixation'] = word_tokens_has_fixation
    sent_obj['word_tokens_with_mask'] = word_tokens_with_mask
    sent_obj['word_tokens_all'] = word_tokens_all
    return sent_obj


def build_dataset_dict(mat_paths: list, subject_skip=SKIP_SUBJECTS) -> dict:
    """解析一組 v2.0（HDF5 / v7.3）`.mat`，回傳 `{subject: [sent_obj | None, ...]}`。"""
    dataset_dict = {}
    for p_idx, file_name in enumerate(tqdm(mat_paths, desc='Subjects')):
        subject = os.path.basename(file_name).split('results')[1].split('_')[0]
        if subject in subject_skip:
            continue

        dataset_dict[subject] = []
        with h5py.File(file_name, 'r', rdcc_nbytes=128*1024*1024) as f:
            sd = f['sentenceData']
            rawData = sd['rawData']
            contentData = sd['content']
            wordData = sd['word']
            mean_objs = {'mean_' + b: sd['mean_' + b] for b in BANDS}

            n_sent = len(rawData)
            for idx in tqdm(range(n_sent), desc='{} ({}/{})'.format(subject, p_idx + 1, len(mat_paths)), leave=False):
                dataset_dict[subject].append(
                    _build_sentence(f, idx, rawData, contentData,
                                    wordData, mean_objs))

    return dataset_dict


def convert(dataset_root: str, output_root: str, task: str,
           fs_out_list: list, verify: bool = False) -> dict:
    """把一個 task 的 `.mat` 轉成 pickle，每個採樣率各一份。

    共用的目錄檢查/寫檔/verify 邏輯在 `zuco_convert.run()`；這裡只負責 v2
    專屬的部分：依 task 決定要讀 `_NR.mat` 還是 `_TSR.mat`。

    Args:
        dataset_root: ZuCo 根目錄（底下有 `v1/` `v2/`）。
        output_root: pickle 輸出根目錄。
        task: 正規 task 名，見 `zuco_paths.CANONICAL_TASKS`。
        fs_out_list: 要輸出的採樣率；`None` 代表原始 500Hz。
        verify: 印出 rawData 形狀。

    Returns:
        `{採樣率: 寫出的路徑}`，500Hz 的 key 是 `None`。
    """
    suffix = MAT_SUFFIX[task]
    return zuco_convert.run(
        dataset_root, output_root, task, fs_out_list, verify,
        mat_filter=lambda f: f.endswith(suffix),
        build_fn=build_dataset_dict, mat_filter_desc='*' + suffix)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-t', '--task_name', choices=V2_TASKS, required=True)
    zuco_convert.add_common_args(parser)
    args = parser.parse_args()

    fs_out_list = [None if v == '500' else int(v) for v in args.fs_out]
    convert(args.dataset_root, args.output_root or args.dataset_root,
            args.task_name, fs_out_list, verify=args.verify)


if __name__ == '__main__':
    main()
