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

import zuco_convert

BANDS = ['t1', 't2', 'a1', 'a2', 'b1', 'b2', 'g1', 'g2']
V1_TASKS = ('task1-SR', 'task2-NR', 'task3-TSR')


def build_dataset_dict(mat_paths: list, include_answer_eeg: bool = False) -> dict:
    """解析一組 v1.0 `.mat`，回傳 `{subject: [sent_obj | None, ...]}`。

    `include_answer_eeg` 由呼叫端（`convert()`）決定 —— 只有 task1-SR 的
    `.mat` 真的有 `answer_mean_*` 欄位；這支函式只管解析，不該自己猜 task
    語意。就算 `include_answer_eeg=True`，仍用 `hasattr` 當第二道防線，
    這樣一份真的缺欄位的 task1-SR 檔也不會炸。

    單一檔案解析失敗不會中止整個迴圈——ZuCo 用 archive 分發，macOS/OSF
    解壓縮常留下 AppleDouble 殘檔，跑到最後一個檔案才炸掉的話，前面幾小時
    的解析就整個作廢。失敗的檔案記下來，迴圈跑完後一次印出來；那個 subject
    就是沒有，但其他人的資料還在。
    """
    dataset_dict = {}
    failures = []
    pbar = tqdm(mat_paths)
    for mat_file in pbar:
        pbar.set_description(os.path.basename(mat_file))
        try:
            subject = os.path.basename(mat_file).split('_')[0].replace('results', '').strip()
            matdata = io.loadmat(mat_file, squeeze_me=True,
                                 struct_as_record=False)['sentenceData']

            sentences = []
            for sent in np.atleast_1d(matdata):
                word_data = sent.word
                if isinstance(word_data, float):
                    sentences.append(None)
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
                    if np.size(word.nFixations) > 0 and word.nFixations > 0:
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
                sentences.append(sent_obj)

            dataset_dict[subject] = sentences
        except Exception as exc:
            failures.append((mat_file, exc))

    if failures:
        print('[WARN] {} of {} mat file(s) failed to parse and were skipped:'.format(
            len(failures), len(mat_paths)))
        for path, exc in failures:
            print('  {}: {!r}'.format(path, exc))

    return dataset_dict


def convert(dataset_root: str, output_root: str, task: str,
           fs_out_list: list, verify: bool = False) -> dict:
    """把一個 task 的 `.mat` 轉成 pickle，每個採樣率各一份。

    共用的目錄檢查/寫檔/verify 邏輯在 `zuco_convert.run()`；這裡只負責 v1
    專屬的部分：檔名 filter（`results*.mat`，篩掉 AppleDouble 之類的殘檔）
    與只有 v1 需要猜的 `include_answer_eeg` 旗標。

    Args:
        dataset_root: ZuCo 根目錄（底下有 `v1/` `v2/`）。
        output_root: pickle 輸出根目錄。
        task: 正規 task 名，見 `zuco_paths.CANONICAL_TASKS`。
        fs_out_list: 要輸出的採樣率；`None` 代表原始 500Hz。
        verify: 印出重取樣前後的長度對照。

    Returns:
        `{採樣率: 寫出的路徑}`，500Hz 的 key 是 `None`。
    """
    build_fn = lambda mat_paths: build_dataset_dict(     # noqa: E731
        mat_paths, include_answer_eeg=(task == 'task1-SR'))

    return zuco_convert.run(
        dataset_root, output_root, task, fs_out_list, verify,
        mat_filter=lambda f: f.startswith('results') and f.endswith('.mat'),
        build_fn=build_fn, mat_filter_desc='.mat')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-t', '--task_name', choices=V1_TASKS, required=True)
    zuco_convert.add_common_args(parser)
    args = parser.parse_args()

    fs_out_list = [None if v == '500' else int(v) for v in args.fs_out]
    convert(args.dataset_root, args.output_root or args.dataset_root,
            args.task_name, fs_out_list, verify=args.verify)


if __name__ == '__main__':
    main()
