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
import zuco_convert                                                   # noqa: E402
import zuco_paths                                                     # noqa: E402

# h5py 是 v2 的可選依賴（v1 完全不需要）。用 importorskip 而不是裸的
# `import h5py`，這樣沒裝 h5py 的環境至少會拿到一個清楚的 skip 訊息，
# 而不是一個從 mid-file import 冒出來、看起來像是隨機位置壞掉的
# collection error。
h5py = pytest.importorskip('h5py')                                    # noqa: E402
import construct_dataset_mat_to_pickle_v2 as v2                       # noqa: E402


def _fake_sentence(n_samples=1000, with_raw=True, with_answer=True, with_word=True,
                   empty_nfixations=False):
    """最小的 v1 sentenceData 元素，欄位名與真實 .mat 一致。

    `with_answer=False` 模擬 NR/TSR 的 `.mat`：沒有 `answer_mean_*` 欄位
    （那是只有 task1-SR 才問的閱讀理解題分數）。
    `with_word=False` 模擬完全被跳過的句子：`word` 欄位是 NaN 純量，
    而不是 word-struct 的 object array —— 真實 ZuCo 檔案裡這代表這句沒有
    任何 fixation 記錄。
    `empty_nfixations=True` 讓唯一的 word 的 `nFixations` 是空陣列而不是
    純量 0——真實 ZuCo 檔案偶爾會這樣存。`np.size(word.nFixations) > 0`
    那道守門（d236ecc）就是為了擋這個案例，之前沒有測試覆蓋到。
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
        nfix = np.array([]) if empty_nfixations else np.int64(1)
        word = {'content': 'the', 'nFixations': nfix}
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


def test_apple_double_stray_files_are_ignored(tmp_path):
    """F4：v1 之前吃任何 `*.mat`；macOS/OSF 解壓縮常留下 AppleDouble 殘檔
    （`._resultsZAB_SR.mat`），會被解析成 subject='._ZAB' 並在
    `io.loadmat` 裡炸掉。過濾成 `results*.mat` 才會跳過它。"""
    mat_dir = _make_v1_tree(tmp_path)
    (mat_dir / '._resultsZAB_SR.mat').write_bytes(b'not a real mat file')
    out = tmp_path / 'out'

    written = v1.convert(str(tmp_path), str(out), 'task1-SR', [None])

    with open(written[None], 'rb') as handle:
        data = pickle.load(handle)
    assert set(data) == {'ZAB'}


def test_a_single_corrupt_mat_file_does_not_abort_the_whole_run(tmp_path, capsys):
    """F4：解析失敗的檔案要記下來、跳過，而不是讓整個 run 連其他 subject
    的資料都拿不到——失去一個 subject 好過失去整個 run。"""
    mat_dir = _make_v1_tree(tmp_path)
    (mat_dir / 'resultsBAD_SR.mat').write_bytes(b'garbage, not a real mat file')
    out = tmp_path / 'out'

    written = v1.convert(str(tmp_path), str(out), 'task1-SR', [None])
    captured = capsys.readouterr()

    with open(written[None], 'rb') as handle:
        data = pickle.load(handle)
    assert 'ZAB' in data
    assert 'BAD' not in data
    assert 'failed to parse' in captured.out
    assert 'resultsBAD_SR.mat' in captured.out


def test_disk_space_precheck_raises_before_any_mat_is_parsed(tmp_path, monkeypatch):
    """F8：磁碟空間明顯不夠要在解析任何 .mat 之前就擋下來，不是等寫檔才發現。
    `_atomic_pickle_dump` 不留半成品檔案是善後，不是預防。"""
    _make_v1_tree(tmp_path)
    parsed = []
    monkeypatch.setattr(v1, 'build_dataset_dict',
                        lambda *a, **k: parsed.append(1) or {})

    class _FakeUsage:
        free = 0

    monkeypatch.setattr(zuco_convert.shutil, 'disk_usage', lambda path: _FakeUsage())

    with pytest.raises(RuntimeError, match='free'):
        v1.convert(str(tmp_path), str(tmp_path / 'out'), 'task1-SR', [None, 200])
    assert parsed == [], 'a .mat was parsed before the disk-space check'


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


def test_missing_mat_dir_lists_what_is_actually_there(tmp_path):
    """F2：目錄名對不上時，把上一層實際有什麼列出來，讓 operator 一次就能
    修對 zuco_paths.TASK_LAYOUT，而不是只看到一句 FileNotFoundError。"""
    (tmp_path / 'v1' / 'task1 - SR (typo)').mkdir(parents=True)

    with pytest.raises(FileNotFoundError) as exc:
        v1.convert(str(tmp_path), str(tmp_path / 'out'), 'task1-SR', [None])

    msg = str(exc.value)
    assert 'task1 - SR (typo)' in msg
    assert 'TASK_LAYOUT' in msg


def test_summary_prints_aggregated_counts_before_the_write_loop(tmp_path, capsys):
    """F3：--verify 只印第一個帶 rawData 句子的 shape，抓不到「系統性地沒有
    任何句子帶 rawData」這種全軍覆沒的解析失敗。這個彙總要在寫檔迴圈之前
    印出來，讓 operator 能在幾十 GB 寫出去之前 Ctrl-C，而且要跨所有
    subject 聚合，不能只看 subjects[0]。"""
    mat_dir = tmp_path / 'v1' / zuco_paths.TASK_LAYOUT['task1-SR'][1] / 'Matlab files'
    mat_dir.mkdir(parents=True)
    _write_v1_mat(str(mat_dir / 'resultsZAB_SR.mat'),
                  [_fake_sentence(1000), _fake_sentence(1000, with_word=False)])
    out = tmp_path / 'out'

    v1.convert(str(tmp_path), str(out), 'task1-SR', [None])
    captured = capsys.readouterr()

    summary_idx = captured.out.index('Sentences (ZAB): 2')
    write_idx = captured.out.index('500Hz ->')
    assert summary_idx < write_idx, 'summary must print before the write loop'
    assert 'None: 1   rawData: 1' in captured.out


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


def test_word_with_empty_nfixations_array_is_masked_without_raising(tmp_path):
    """F8：`np.size(word.nFixations) > 0` 的守門（d236ecc）在空陣列（不是
    純量 0）上不會出錯，且該字正確被當成沒有 fixation 處理——之前沒有
    測試覆蓋這個分支。"""
    task = 'task1-SR'
    mat_dir = tmp_path / 'v1' / zuco_paths.TASK_LAYOUT[task][1] / 'Matlab files'
    mat_dir.mkdir(parents=True)
    _write_v1_mat(str(mat_dir / 'resultsZAB_SR.mat'),
                  [_fake_sentence(1000, empty_nfixations=True)])
    out = tmp_path / 'out'

    written = v1.convert(str(tmp_path), str(out), task, [None])

    with open(written[None], 'rb') as handle:
        data = pickle.load(handle)
    sent = data['ZAB'][0]
    assert sent['word'] == []
    assert sent['word_tokens_with_mask'] == ['[MASK]']
    assert sent['word_tokens_has_fixation'] == []


# ── v2（h5py / v7.3 mat） ─────────────────────────────────────────────

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
    """替身，不是懶惰：`_write_v2_mat` 的 `word` 欄位故意寫成一個
    `float64` 純量 dataset（`np.zeros(1)`），不是真正的 word-struct
    group——真正的 `dh.extract_word_level_data` 在這個 fixture 上
    `available_objects[0]` 不是 `str`，會直接走 else 分支回傳
    `({}, [], [], [])`（見 `test_extract_word_level_data_contract_is_a_four_tuple`）。
    要測到 rawData 轉置、YMH 跳過、task 後綴過濾這些跟 word-level 資料無關
    的行為，就得把 word 解析短路掉，所以才 monkeypatch 這個函式，換成一份
    看起來像真的有 fixation 的資料。建一份忠實的 word-struct HDF5 fixture
    來取代這個 mock，划不來：多了近百行只為了保護這個分支完全不會碰到的
    程式碼，而且一份「看起來忠實但其實不是」的 fixture 比一個誠實的 mock
    更危險。
    """
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


def test_v2_sentence_without_words_becomes_none_at_both_rates(tmp_path):
    """沒有 word-level 資料的句子在 v2 轉檔器中必須保留為 None sentinel。

    這裡刻意不 monkeypatch `dh.extract_word_level_data`——`_write_v2_mat`
    的 `word` 欄位本來就是一個 float64 純量 dataset，真正的
    `extract_word_level_data` 在它上面就會走 else 分支回傳
    `({}, [], [], [])`（`available_objects[0]` 不是 `str`）。用真正的函式
    跑，這個測試才是貨真價實的端到端測試，蓋到 `_build_sentence` 裡
    `word_data == {}` 那個 early-return，而不是「mock 驗證 mock」。
    """
    task = 'task2-NR-2.0'
    mat_dir = tmp_path / 'v2' / zuco_paths.TASK_LAYOUT[task][1] / 'Matlab files'
    mat_dir.mkdir(parents=True)
    _write_v2_mat(str(mat_dir / 'resultsYAC_NR.mat'), n_sent=2)

    written = v2.convert(str(tmp_path), str(tmp_path / 'out'), task, [None, 200])

    with open(written[None], 'rb') as handle:
        at_500 = pickle.load(handle)
    with open(written[200], 'rb') as handle:
        at_200 = pickle.load(handle)
    assert at_500['YAC'] == [None, None]
    assert at_200['YAC'] == [None, None]


def test_extract_word_level_data_contract_is_a_four_tuple(tmp_path):
    """簽章回歸測試：真正的 `dh.extract_word_level_data` 必須一直回傳
    `(word_level_data, word_tokens_all, word_tokens_has_fixation,
    word_tokens_with_mask)` 這個 4-tuple。上面所有測試都靠
    monkeypatch（`_mock_v2_words`）或依賴這份 fixture 的特定形狀
    （`word` 是純量 dataset）來繞開真正的 word-struct 解析，簽章本身
    沒有被單獨釘住——一旦真正的函式簽章漂移，不會有任何測試發現。"""
    path = tmp_path / 'contract.mat'
    _write_v2_mat(str(path), n_sent=1)

    with h5py.File(str(path), 'r') as f:
        sd = f['sentenceData']
        result = v2.dh.extract_word_level_data(f, f[sd['word'][0][0]])

    assert isinstance(result, tuple)
    assert len(result) == 4

