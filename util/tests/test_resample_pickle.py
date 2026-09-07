"""resample_pickle 的單元測試。用合成的 dataset dict，不需要真的 ZuCo 資料。

    python -m pytest EEG-To-Text/util/tests/ -q
"""

import os
import pickle
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from resample_pickle import process_task, resample_dataset_dict, verify_report      # noqa: E402


def _sent(n_samples=5000, with_raw=True):
    """最小的 sent_obj，欄位名與 construct_dataset_mat_to_pickle_v1.py 一致。"""
    obj = {
        'content': 'the quick brown fox',
        'sentence_level_EEG': {'mean_t1': np.zeros(105, dtype=np.float32)},
        'word': [{'content': 'the', 'nFixations': 1}],
        'word_tokens_all': ['the', 'quick', 'brown', 'fox'],
        'word_tokens_has_fixation': ['the'],
        'word_tokens_with_mask': ['the', '[MASK]', '[MASK]', '[MASK]'],
    }
    if with_raw:
        obj['rawData'] = np.random.randn(105, n_samples).astype(np.float32)
    return obj


# ── rawData 被重取樣 ──────────────────────────────────────────────────
def test_raw_data_is_resampled_to_the_target_length():
    d = {'ZAB': [_sent(5000)]}
    out = resample_dataset_dict(d, 500, 200)
    assert out['ZAB'][0]['rawData'].shape == (105, 2000)


def test_raw_data_stays_float32():
    d = {'ZAB': [_sent(5000)]}
    out = resample_dataset_dict(d, 500, 200)
    assert out['ZAB'][0]['rawData'].dtype == np.float32


# ── 其餘欄位原樣保留 ──────────────────────────────────────────────────
def test_non_raw_fields_are_carried_over_untouched():
    d = {'ZAB': [_sent(5000)]}
    out = resample_dataset_dict(d, 500, 200)
    got = out['ZAB'][0]
    assert got['content'] == 'the quick brown fox'
    assert got['word_tokens_all'] == ['the', 'quick', 'brown', 'fox']
    assert got['word'][0]['nFixations'] == 1


def test_sentence_level_band_features_are_not_resampled():
    """頻帶特徵是 ZuCo 預先算好的統計量，不是時序訊號，長度必須不動。"""
    d = {'ZAB': [_sent(5000)]}
    out = resample_dataset_dict(d, 500, 200)
    assert out['ZAB'][0]['sentence_level_EEG']['mean_t1'].shape == (105,)


# ── 索引對齊（ZuCo_dataset 按 index range 切 train/dev/test）──────────
def test_none_entries_keep_their_index():
    d = {'ZAB': [_sent(5000), None, _sent(5000)]}
    out = resample_dataset_dict(d, 500, 200)
    assert len(out['ZAB']) == 3
    assert out['ZAB'][1] is None


def test_sentences_without_raw_data_are_kept():
    d = {'ZAB': [_sent(5000, with_raw=False)]}
    out = resample_dataset_dict(d, 500, 200)
    assert len(out['ZAB']) == 1
    assert 'rawData' not in out['ZAB'][0]


def test_subject_keys_and_order_are_preserved():
    d = {'ZAB': [_sent(5000)], 'ZDM': [_sent(5000)]}
    out = resample_dataset_dict(d, 500, 200)
    assert list(out.keys()) == ['ZAB', 'ZDM']


def test_input_dict_is_not_mutated():
    d = {'ZAB': [_sent(5000)]}
    resample_dataset_dict(d, 500, 200)
    assert d['ZAB'][0]['rawData'].shape == (105, 5000)


# ── ragged 長度 ───────────────────────────────────────────────────────
def test_short_sentences_resample_without_error():
    d = {'ZAB': [_sent(731)]}
    out = resample_dataset_dict(d, 500, 200)
    assert out['ZAB'][0]['rawData'].shape == (105, 293)


# ── verify 報表 ───────────────────────────────────────────────────────
def test_verify_report_records_both_lengths():
    before = np.random.randn(105, 5000).astype(np.float32)
    after = np.random.randn(105, 2000).astype(np.float32)
    rep = verify_report(before, after, 500, 200)
    assert rep['len_before'] == 5000
    assert rep['len_after'] == 2000


def test_verify_report_shows_the_stopband_is_gone():
    t = np.arange(5000) / 500
    before = np.sin(2 * np.pi * 150 * t).astype(np.float32)[None, :].repeat(105, axis=0)
    from eeg_resample import resample_eeg
    after = resample_eeg(before, 500, 200)
    rep = verify_report(before, after, 500, 200)
    assert rep['band_ratio']['>100'] < 1e-3


# ── 磁碟 I/O 層（process_task）───────────────────────────────────────
# `resample_dataset_dict` 只碰記憶體裡的 dict，上面的測試都不會經過
# `_pickle_paths` / `process_task` 實際讀寫檔案的路徑 —— 檔名樣板改壞、
# 或 src/dst 寫反，都不會被上面任何一個測試抓到。這裡用 tmp_path 建一份
# 假的 `dataset/ZuCo/<task>/pickle/` 目錄，走完整的讀 -> 重取樣 -> 寫流程。

def _write_pickle(path, obj):
    with open(path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _make_task_pickle_dir(tmp_path, task_name, dataset_dict):
    """在 `tmp_path` 下建出 `dataset/ZuCo/<task_name>/pickle/<task_name>-dataset.pickle`。"""
    pickle_dir = tmp_path / 'dataset' / 'ZuCo' / task_name / 'pickle'
    pickle_dir.mkdir(parents=True)
    src_path = pickle_dir / f'{task_name}-dataset.pickle'
    _write_pickle(src_path, dataset_dict)
    return pickle_dir, src_path


def test_process_task_writes_the_exact_contract_filename(tmp_path):
    """輸出檔名是 Task 3 讀取的契約，故意寫死字串比對，不算 f-string。"""
    task_name = 'task1-SR'
    dataset = {'ZAB': [_sent(5000), None, _sent(731), _sent(5000, with_raw=False)]}
    pickle_dir, _ = _make_task_pickle_dir(tmp_path, task_name, dataset)

    process_task(task_name, 500, 200, verify=False, project_root=str(tmp_path))

    expected_dst = pickle_dir / 'task1-SR-dataset-200hz.pickle'
    assert expected_dst.exists()
    # 目錄裡只該多這一個檔案，沒有暫存檔殘留
    assert sorted(p.name for p in pickle_dir.iterdir()) == [
        'task1-SR-dataset-200hz.pickle',
        'task1-SR-dataset.pickle',
    ]


def test_process_task_leaves_the_source_pickle_byte_identical(tmp_path):
    task_name = 'task1-SR'
    dataset = {'ZAB': [_sent(5000)]}
    _, src_path = _make_task_pickle_dir(tmp_path, task_name, dataset)
    bytes_before = src_path.read_bytes()

    process_task(task_name, 500, 200, verify=False, project_root=str(tmp_path))

    assert src_path.read_bytes() == bytes_before


def test_process_task_output_round_trips_to_resampled_shapes(tmp_path):
    task_name = 'task1-SR'
    dataset = {'ZAB': [_sent(5000), None, _sent(731), _sent(5000, with_raw=False)]}
    pickle_dir, _ = _make_task_pickle_dir(tmp_path, task_name, dataset)

    process_task(task_name, 500, 200, verify=False, project_root=str(tmp_path))

    with open(pickle_dir / 'task1-SR-dataset-200hz.pickle', 'rb') as handle:
        out = pickle.load(handle)

    assert out['ZAB'][0]['rawData'].shape == (105, 2000)
    assert out['ZAB'][0]['rawData'].dtype == np.float32
    assert out['ZAB'][1] is None
    assert out['ZAB'][2]['rawData'].shape == (105, 293)
    assert 'rawData' not in out['ZAB'][3]


def test_process_task_overwrite_leaves_no_temp_files_and_says_so(tmp_path, capsys):
    """重跑一次現有輸出：印訊息要講清楚是覆寫，且不留下 `.tmp` 暫存檔。"""
    task_name = 'task1-SR'
    dataset = {'ZAB': [_sent(5000)]}
    pickle_dir, _ = _make_task_pickle_dir(tmp_path, task_name, dataset)

    process_task(task_name, 500, 200, verify=False, project_root=str(tmp_path))
    capsys.readouterr()  # 丟掉第一次跑的輸出

    process_task(task_name, 500, 200, verify=False, project_root=str(tmp_path))
    captured = capsys.readouterr()

    assert 'overwrote' in captured.out
    assert sorted(p.name for p in pickle_dir.iterdir()) == [
        'task1-SR-dataset-200hz.pickle',
        'task1-SR-dataset.pickle',
    ]


def test_process_task_skips_a_missing_source_without_raising(tmp_path, capsys):
    pickle_dir = tmp_path / 'dataset' / 'ZuCo' / 'task1-SR' / 'pickle'
    pickle_dir.mkdir(parents=True)

    process_task('task1-SR', 500, 200, verify=False, project_root=str(tmp_path))
    captured = capsys.readouterr()

    assert 'SKIP' in captured.out
    assert list(pickle_dir.iterdir()) == []
