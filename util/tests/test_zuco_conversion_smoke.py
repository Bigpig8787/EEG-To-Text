"""轉檔器輸出的檔名 vs SNN 端讀回的檔名，跨 repo 契約測試。

    python -m pytest util/tests/test_zuco_conversion_smoke.py -q

兩邊各自算路徑：轉檔器用 util/zuco_paths.py，訓練入口用
Spiking-EEG2TEXT/EEGSNN/cfg_runner/pickle_naming.py。不一致的後果是訓練
啟動時 FileNotFoundError —— 要在這裡就抓到，不要等跑訓練。
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import zuco_paths                                                     # noqa: E402

# util/tests/ -> util/ -> EEG-To-Text/ -> snn_muti-view_transformer/
# 四層 dirname，不是三層：三層只會停在 EEG-To-Text，接出來的路徑永遠不存在，
# 於是 fixture 會無條件 skip，而這個 task 就白做了。
_REPO_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
_SNN_CFG = os.path.join(_REPO_PARENT, 'Spiking-EEG2TEXT', 'EEGSNN', 'cfg_runner')


@pytest.fixture(scope='module')
def pickle_naming():
    if not os.path.isdir(_SNN_CFG):
        pytest.skip('Spiking-EEG2TEXT checkout not found at ' + _SNN_CFG)
    sys.path.insert(0, _SNN_CFG)
    import pickle_naming
    return pickle_naming


def test_every_canonical_task_is_reachable_from_the_snn_side(pickle_naming):
    """SNN 端的 TASK_DIRS value 必須正好是這邊的正規名。"""
    assert set(pickle_naming.TASK_DIRS.values()) == set(zuco_paths.CANONICAL_TASKS)


@pytest.mark.parametrize('fs_out,suffix', [
    (None, '-dataset.pickle'),
    (200, '-dataset-200hz.pickle'),
])
def test_written_filename_matches_what_the_trainer_looks_for(
        pickle_naming, fs_out, suffix):
    for key, canonical in pickle_naming.TASK_DIRS.items():
        written = os.path.basename(zuco_paths.pickle_name(canonical, fs_out))
        looked_for = pickle_naming.pickle_file_name(key, suffix)
        assert written == looked_for, (canonical, written, looked_for)
