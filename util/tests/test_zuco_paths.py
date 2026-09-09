"""zuco_paths 的單元測試。純路徑邏輯，不需要 numpy/scipy/h5py。

    python -m pytest util/tests/test_zuco_paths.py -q
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zuco_paths import (                                              # noqa: E402
    CANONICAL_TASKS, TASK_LAYOUT, assert_writable, mat_dir, pickle_dir,
    pickle_name, pickle_path)


def test_the_five_canonical_names_match_what_the_snn_side_expects():
    """這五個字串同時出現在 cfg_runner/pickle_naming.py 與 resample_pickle.py。
    改動它們會連鎖到 SNN 端所有 config，所以在這裡釘死。"""
    assert CANONICAL_TASKS == (
        'task1-SR', 'task2-NR', 'task3-TSR', 'task2-NR-2.0', 'task2-TSR-2.0')


def test_every_canonical_task_has_a_layout_entry():
    assert set(TASK_LAYOUT) == set(CANONICAL_TASKS)


@pytest.mark.parametrize('task,version,on_disk', [
    ('task1-SR',      'v1', 'task1- SR'),
    ('task2-NR',      'v1', 'task2 - NR'),
    ('task3-TSR',     'v1', 'task3 - TSR'),
    ('task2-NR-2.0',  'v2', 'task2 - NR-2.0'),
    ('task2-TSR-2.0', 'v2', 'task2 - TSR'),
])
def test_on_disk_names_carry_the_exact_spacing(task, version, on_disk):
    """磁碟上的名字帶空格，且 task2-TSR-2.0 的磁碟名沒有 -2.0。
    這些是實地看過的字串，不是推測出來的。"""
    assert TASK_LAYOUT[task] == (version, on_disk)


def test_mat_dir_joins_version_on_disk_name_and_matlab_files():
    got = mat_dir(os.path.join('root'), 'task2-TSR-2.0')
    assert got == os.path.join('root', 'v2', 'task2 - TSR', 'Matlab files')


def test_pickle_dir_uses_the_canonical_name_and_no_version_layer():
    """SNN 端的 pickle_naming 假設 task 平鋪在 dataset_root 底下。
    輸出若帶 v1/ v2/ 層，train 就找不到檔案。"""
    got = pickle_dir(os.path.join('out'), 'task2-NR-2.0')
    assert got == os.path.join('out', 'task2-NR-2.0', 'pickle')


def test_pickle_name_without_fs_is_the_500hz_original():
    assert pickle_name('task1-SR') == 'task1-SR-dataset.pickle'
    assert pickle_name('task2-TSR-2.0') == 'task2-TSR-2.0-dataset.pickle'


def test_pickle_name_with_fs_matches_the_pickle_suffix_contract():
    """'-dataset-200hz.pickle' 是 data.pickle_suffix 的契約值，寫死比對。"""
    assert pickle_name('task1-SR', 200) == 'task1-SR-dataset-200hz.pickle'
    assert pickle_name('task2-NR-2.0', 200) == 'task2-NR-2.0-dataset-200hz.pickle'


def test_pickle_path_composes_dir_and_name():
    got = pickle_path(os.path.join('out'), 'task3-TSR', 200)
    assert got == os.path.join(
        'out', 'task3-TSR', 'pickle', 'task3-TSR-dataset-200hz.pickle')


def test_unknown_task_raises_with_the_valid_names_listed():
    with pytest.raises(KeyError) as exc:
        mat_dir('root', 'task9-XX')
    assert 'task1-SR' in str(exc.value)


def test_assert_writable_passes_on_a_writable_dir(tmp_path):
    assert_writable(str(tmp_path))          # 不應拋出


def test_assert_writable_creates_missing_parents(tmp_path):
    target = tmp_path / 'a' / 'b' / 'c'
    assert_writable(str(target))
    assert target.is_dir()


def test_assert_writable_reports_the_path_when_it_cannot_write(tmp_path, monkeypatch):
    """掛載進來的 dataset 唯讀是常態。要在解析 .mat 之前就失敗，
    不是跑幾小時之後才在寫檔時炸。"""
    def _boom(*a, **k):
        raise PermissionError('read-only file system')
    monkeypatch.setattr('zuco_paths.tempfile.NamedTemporaryFile', _boom)

    with pytest.raises(RuntimeError, match='not writable'):
        assert_writable(str(tmp_path))
