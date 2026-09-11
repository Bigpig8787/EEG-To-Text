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
def test_written_path_matches_what_the_trainer_looks_for(
        pickle_naming, fs_out, suffix, tmp_path):
    """比對整條路徑，不只檔名——只比檔名擋不住「轉檔器輸出多套一層 v1/v2/
    子目錄」這種回歸：`zuco_paths.pickle_name()` 照樣回傳一模一樣的檔名，
    只有 `pickle_dir()`（這裡走的是 `pickle_path()`）多了一層才會露餡。

    trainer 端（`cfg_runner/train_multiview_snn_cfg.py:78`，
    `pretrain_snn_cfg.py:70`、`eval_multiview_snn_cfg.py:78` 三份逐字相同）
    組路徑的方式是
    `dataset_root / TASK_DIRS[key] / 'pickle' / pickle_file_name(key, suffix)`；
    這裡刻意用同樣的積木（`pickle_naming.TASK_DIRS`、
    `pickle_naming.pickle_file_name`）逐字重組一次，而不是拿一個寫死的路徑
    字串比對，這樣「flat layout、無 v1/v2 層」這條契約才是被組出來驗證，
    不是被抄一次而已。
    """
    dataset_root = str(tmp_path)
    for key, canonical in pickle_naming.TASK_DIRS.items():
        written = zuco_paths.pickle_path(dataset_root, canonical, fs_out)
        looked_for = os.path.join(
            dataset_root, canonical, 'pickle',
            pickle_naming.pickle_file_name(key, suffix))
        assert written == looked_for, (canonical, written, looked_for)
