"""兩支 .mat 轉檔器的端到端測試，用合成的 .mat 檔。

    python -m pytest util/tests/test_mat_converters.py -q

不碰真實 ZuCo 資料：那是幾十 GB，而且這台機器上沒有。
"""

import os
import pickle
import sys

import numpy as np
import pytest
import scipy.io as sio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import construct_dataset_mat_to_pickle_v1 as v1                       # noqa: E402
import zuco_paths                                                     # noqa: E402


def _fake_sentence(n_samples=1000, with_raw=True, with_answer=True, with_word=True):
    """最小的 v1 sentenceData 元素，欄位名與真實 .mat 一致。

    `with_answer=False` 模擬 NR/TSR 的 `.mat`：沒有 `answer_mean_*` 欄位
    （那是只有 task1-SR 才問的閱讀理解題分數）。
    `with_word=False` 模擬完全被跳過的句子：`word` 欄位是 NaN 純量，
    而不是 word-struct 的 object array —— 真實 ZuCo 檔案裡這代表這句沒有
    任何 fixation 記錄。
    """
    bands = ['t1', 't2', 'a1', 'a2', 'b1', 'b2', 'g1', 'g2']
    sent = {'content': 'the quick brown fox'}
    for b in bands:
        sent['mean_' + b] = np.zeros(105, dtype=np.float32)
        if with_answer:
            sent['answer_mean_' + b] = np.zeros(105, dtype=np.float32)
    if with_raw:
        sent['rawData'] = np.random.randn(105, n_samples).astype(np.float32)
    else:
        sent['rawData'] = np.float64(np.nan)      # 真實檔缺 rawData 時是 float
    if with_word:
        word = {'content': 'the', 'nFixations': np.int64(1)}
        for b in bands:
            word['FFD_' + b] = np.zeros(105, dtype=np.float32)
            word['TRT_' + b] = np.zeros(105, dtype=np.float32)
            word['GD_' + b] = np.zeros(105, dtype=np.float32)
        sent['word'] = np.array([word], dtype=object)
    else:
        sent['word'] = np.float64(np.nan)         # 真實檔完全跳過的句子，word 是 NaN 純量
    return sent


def _write_v1_mat(path, sentences):
    sio.savemat(path, {'sentenceData': np.array(sentences, dtype=object)})


def _make_v1_tree(tmp_path, task='task1-SR', n_samples=1000, with_answer=True):
    mat_dir = tmp_path / 'v1' / zuco_paths.TASK_LAYOUT[task][1] / 'Matlab files'
    mat_dir.mkdir(parents=True)
    _write_v1_mat(str(mat_dir / 'resultsZAB_SR.mat'),
                  [_fake_sentence(n_samples, with_answer=with_answer),
                   _fake_sentence(n_samples, with_raw=False, with_answer=with_answer)])
    return mat_dir


def test_convert_writes_both_sampling_rates(tmp_path):
    _make_v1_tree(tmp_path)
    out = tmp_path / 'out'

    written = v1.convert(str(tmp_path), str(out), 'task1-SR', [None, 200])

    assert set(written) == {None, 200}
    assert os.path.isfile(written[None])
    assert os.path.isfile(written[200])
    assert os.path.basename(written[None]) == 'task1-SR-dataset.pickle'
    assert os.path.basename(written[200]) == 'task1-SR-dataset-200hz.pickle'


def test_the_200hz_pickle_has_two_fifths_the_samples(tmp_path):
    """500 -> 200 是 x0.4。1000 點應該變成 400 點。"""
    _make_v1_tree(tmp_path, n_samples=1000)
    out = tmp_path / 'out'
    written = v1.convert(str(tmp_path), str(out), 'task1-SR', [None, 200])

    with open(written[None], 'rb') as handle:
        at_500 = pickle.load(handle)
    with open(written[200], 'rb') as handle:
        at_200 = pickle.load(handle)

    assert at_500['ZAB'][0]['rawData'].shape == (105, 1000)
    assert at_200['ZAB'][0]['rawData'].shape == (105, 400)


def test_a_sentence_without_rawdata_survives_both_rates(tmp_path):
    _make_v1_tree(tmp_path)
    out = tmp_path / 'out'
    written = v1.convert(str(tmp_path), str(out), 'task1-SR', [None, 200])

    with open(written[200], 'rb') as handle:
        data = pickle.load(handle)
    assert len(data['ZAB']) == 2
    assert 'rawData' not in data['ZAB'][1]
    assert data['ZAB'][1]['content'] == 'the quick brown fox'


def test_the_source_mat_is_left_untouched(tmp_path):
    mat_dir = _make_v1_tree(tmp_path)
    mat_file = mat_dir / 'resultsZAB_SR.mat'
    before = mat_file.read_bytes()

    v1.convert(str(tmp_path), str(tmp_path / 'out'), 'task1-SR', [None, 200])

    assert mat_file.read_bytes() == before


def test_output_root_is_checked_before_any_mat_is_parsed(tmp_path, monkeypatch):
    """唯讀輸出要在解析 .mat 之前就失敗，不能跑完幾小時才炸。"""
    _make_v1_tree(tmp_path)
    parsed = []
    monkeypatch.setattr(v1, 'build_dataset_dict',
                        lambda *a, **k: parsed.append(1) or {})
    monkeypatch.setattr(zuco_paths, 'assert_writable',
                        lambda p: (_ for _ in ()).throw(RuntimeError('not writable')))

    with pytest.raises(RuntimeError, match='not writable'):
        v1.convert(str(tmp_path), str(tmp_path / 'out'), 'task1-SR', [None, 200])
    assert parsed == [], 'a .mat was parsed before the writability check'


def test_missing_mat_dir_names_the_path_it_looked_in(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        v1.convert(str(tmp_path), str(tmp_path / 'out'), 'task1-SR', [None])
    assert 'task1- SR' in str(exc.value)


def test_answer_eeg_absent_when_fields_are_absent_for_nr(tmp_path):
    """NR 的 `.mat` 沒有 `answer_mean_*` 欄位；缺欄位分支要安全地不放 answer_EEG。"""
    _make_v1_tree(tmp_path, task='task2-NR', with_answer=False)
    out = tmp_path / 'out'
    written = v1.convert(str(tmp_path), str(out), 'task2-NR', [None])

    with open(written[None], 'rb') as handle:
        data = pickle.load(handle)
    assert all('answer_EEG' not in s for s in data['ZAB'] if s is not None)


def test_answer_eeg_excluded_for_nr_even_if_fields_happen_to_exist(tmp_path):
    """關鍵回歸測試：就算 NR 的 struct 剛好帶了 answer_mean_* 欄位（例如佔位值），
    也不該被收進 answer_EEG —— 這個欄位是 task1-SR 特有的閱讀理解題分數，
    決定要不要收的是 task，不是欄位存不存在。舊版曾經只看
    `hasattr(sent, 'answer_mean_t1')`，會在這裡誤收。
    """
    _make_v1_tree(tmp_path, task='task2-NR', with_answer=True)
    out = tmp_path / 'out'
    written = v1.convert(str(tmp_path), str(out), 'task2-NR', [None])

    with open(written[None], 'rb') as handle:
        data = pickle.load(handle)
    assert all('answer_EEG' not in s for s in data['ZAB'] if s is not None)


def test_answer_eeg_still_present_for_task1_sr(tmp_path):
    _make_v1_tree(tmp_path, task='task1-SR')
    out = tmp_path / 'out'
    written = v1.convert(str(tmp_path), str(out), 'task1-SR', [None])

    with open(written[None], 'rb') as handle:
        data = pickle.load(handle)
    assert 'answer_EEG' in data['ZAB'][0]


def test_a_sentence_with_no_fixations_at_all_becomes_none_at_both_rates(tmp_path):
    """word 欄位是 NaN 純量代表這句完全沒有 fixation 記錄，應該變成 `None`。"""
    task = 'task1-SR'
    mat_dir = tmp_path / 'v1' / zuco_paths.TASK_LAYOUT[task][1] / 'Matlab files'
    mat_dir.mkdir(parents=True)
    _write_v1_mat(str(mat_dir / 'resultsZAB_SR.mat'),
                  [_fake_sentence(1000), _fake_sentence(1000, with_word=False)])
    out = tmp_path / 'out'

    written = v1.convert(str(tmp_path), str(out), task, [None, 200])

    with open(written[None], 'rb') as handle:
        at_500 = pickle.load(handle)
    with open(written[200], 'rb') as handle:
        at_200 = pickle.load(handle)

    assert at_500['ZAB'][1] is None
    assert at_200['ZAB'][1] is None


# ── v2（h5py / v7.3 mat） ─────────────────────────────────────────────
import h5py                                                           # noqa: E402
import construct_dataset_mat_to_pickle_v2 as v2                       # noqa: E402


def _write_v2_mat(path, n_sent=2, n_samples=1000):
    """寫一個最小的 v7.3 風格 mat：sentenceData 底下都是 object reference。"""
    with h5py.File(path, 'w') as f:
        sd = f.create_group('sentenceData')
        refs = {k: [] for k in ('rawData', 'content', 'word')}
        for i in range(n_sent):
            raw = f.create_dataset('raw{}'.format(i),
                                   data=np.random.randn(n_samples, 105))
            content = f.create_dataset(
                'content{}'.format(i),
                data=np.array([[ord(c)] for c in 'hello'], dtype=np.uint16))
            word = f.create_dataset('word{}'.format(i), data=np.zeros(1))
            refs['rawData'].append(raw.ref)
            refs['content'].append(content.ref)
            refs['word'].append(word.ref)
        for key, values in refs.items():
            sd.create_dataset(key, data=np.array(
                [[v] for v in values], dtype=h5py.special_dtype(ref=h5py.Reference)))
        mean_val = f.create_dataset('mean_val', data=np.zeros((105, 1)))
        for b in ['t1', 't2', 'a1', 'a2', 'b1', 'b2', 'g1', 'g2']:
            sd.create_dataset('mean_' + b, data=np.array(
                [[mean_val.ref] for _ in range(n_sent)], dtype=h5py.special_dtype(ref=h5py.Reference)))


def _mock_v2_words(f, word_obj):
    word_data = {0: {'content': 'hello', 'nFix': 1,
                     'GD_EEG': np.zeros((8, 105)),
                     'FFD_EEG': np.zeros((8, 105)),
                     'TRT_EEG': np.zeros((8, 105))}}
    return word_data, ['hello'], ['hello'], ['hello']


def test_v2_transposes_rawdata_to_channels_first(tmp_path, monkeypatch):
    """v2 的 rawData 存成 (T, 105)，必須轉成 (105, T)。"""
    monkeypatch.setattr(v2.dh, 'extract_word_level_data', _mock_v2_words)
    task = 'task2-NR-2.0'
    mat_dir = tmp_path / 'v2' / zuco_paths.TASK_LAYOUT[task][1] / 'Matlab files'
    mat_dir.mkdir(parents=True)
    _write_v2_mat(str(mat_dir / 'resultsYAC_NR.mat'), n_samples=1000)

    written = v2.convert(str(tmp_path), str(tmp_path / 'out'), task, [None, 200])

    with open(written[None], 'rb') as handle:
        at_500 = pickle.load(handle)
    raws = [s['rawData'] for s in at_500['YAC'] if s and 'rawData' in s]
    assert raws, 'no sentence carried rawData'
    assert raws[0].shape == (105, 1000)

    with open(written[200], 'rb') as handle:
        at_200 = pickle.load(handle)
    raws200 = [s['rawData'] for s in at_200['YAC'] if s and 'rawData' in s]
    assert raws200[0].shape == (105, 400)


def test_v2_skips_the_excluded_subject(tmp_path, monkeypatch):
    """YMH 在原本的腳本裡就被跳過，這個行為要保留。"""
    monkeypatch.setattr(v2.dh, 'extract_word_level_data', _mock_v2_words)
    task = 'task2-NR-2.0'
    mat_dir = tmp_path / 'v2' / zuco_paths.TASK_LAYOUT[task][1] / 'Matlab files'
    mat_dir.mkdir(parents=True)
    _write_v2_mat(str(mat_dir / 'resultsYAC_NR.mat'))
    _write_v2_mat(str(mat_dir / 'resultsYMH_NR.mat'))

    written = v2.convert(str(tmp_path), str(tmp_path / 'out'), task, [None])
    with open(written[None], 'rb') as handle:
        data = pickle.load(handle)
    assert 'YAC' in data
    assert 'YMH' not in data


def test_v2_only_reads_mat_files_matching_the_task_suffix(tmp_path, monkeypatch):
    """同一個目錄可能同時放 _NR.mat 與 _TSR.mat，不能混到。"""
    monkeypatch.setattr(v2.dh, 'extract_word_level_data', _mock_v2_words)
    task = 'task2-NR-2.0'
    mat_dir = tmp_path / 'v2' / zuco_paths.TASK_LAYOUT[task][1] / 'Matlab files'
    mat_dir.mkdir(parents=True)
    _write_v2_mat(str(mat_dir / 'resultsYAC_NR.mat'))
    _write_v2_mat(str(mat_dir / 'resultsYAG_TSR.mat'))

    written = v2.convert(str(tmp_path), str(tmp_path / 'out'), task, [None])
    with open(written[None], 'rb') as handle:
        data = pickle.load(handle)
    assert set(data) == {'YAC'}


def test_v2_sentence_without_words_becomes_none_at_both_rates(tmp_path, monkeypatch):
    """沒有 word-level 資料的句子在 v2 轉檔器中必須保留為 None sentinel。"""
    task = 'task2-NR-2.0'
    mat_dir = tmp_path / 'v2' / zuco_paths.TASK_LAYOUT[task][1] / 'Matlab files'
    mat_dir.mkdir(parents=True)
    _write_v2_mat(str(mat_dir / 'resultsYAC_NR.mat'), n_sent=2)
    monkeypatch.setattr(v2.dh, 'extract_word_level_data',
                        lambda f, w: ({}, [], [], []))

    written = v2.convert(str(tmp_path), str(tmp_path / 'out'), task, [None, 200])

    with open(written[None], 'rb') as handle:
        at_500 = pickle.load(handle)
    with open(written[200], 'rb') as handle:
        at_200 = pickle.load(handle)
    assert at_500['YAC'] == [None, None]
    assert at_200['YAC'] == [None, None]

