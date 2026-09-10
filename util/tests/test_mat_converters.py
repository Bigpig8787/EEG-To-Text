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


def _fake_sentence(n_samples=1000, with_raw=True):
    """最小的 v1 sentenceData 元素，欄位名與真實 .mat 一致。"""
    bands = ['t1', 't2', 'a1', 'a2', 'b1', 'b2', 'g1', 'g2']
    sent = {'content': 'the quick brown fox'}
    for b in bands:
        sent['mean_' + b] = np.zeros(105, dtype=np.float32)
        sent['answer_mean_' + b] = np.zeros(105, dtype=np.float32)
    if with_raw:
        sent['rawData'] = np.random.randn(105, n_samples).astype(np.float32)
    else:
        sent['rawData'] = np.float64(np.nan)      # 真實檔缺 rawData 時是 float
    word = {'content': 'the', 'nFixations': np.int64(1)}
    for b in bands:
        word['FFD_' + b] = np.zeros(105, dtype=np.float32)
        word['TRT_' + b] = np.zeros(105, dtype=np.float32)
        word['GD_' + b] = np.zeros(105, dtype=np.float32)
    sent['word'] = np.array([word], dtype=object)
    return sent


def _write_v1_mat(path, sentences):
    sio.savemat(path, {'sentenceData': np.array(sentences, dtype=object)})


def _make_v1_tree(tmp_path, task='task1-SR', n_samples=1000):
    mat_dir = tmp_path / 'v1' / zuco_paths.TASK_LAYOUT[task][1] / 'Matlab files'
    mat_dir.mkdir(parents=True)
    _write_v1_mat(str(mat_dir / 'resultsZAB_SR.mat'),
                  [_fake_sentence(n_samples), _fake_sentence(n_samples, with_raw=False)])
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
