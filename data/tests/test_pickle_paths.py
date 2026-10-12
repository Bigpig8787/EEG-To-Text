"""`data/pickle_paths.py`: the trainers' pickle lookup. Standard library only.

    python -m pytest data/tests/test_pickle_paths.py -q
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from data.pickle_paths import (DEFAULT_PICKLE_SUFFIX, TASK_KEYS,  # noqa: E402
                               default_dataset_root, existing_pickle_paths,
                               pickle_path, task_pickle_paths)

ALL_TASKS = 'task1_task2_task3_taskNRv2_taskTSRv2'


def _touch(root, task_dir, suffix):
    p = pickle_path(str(root), task_dir, suffix)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, 'wb').close()
    return p


def test_default_root_and_suffix_reproduce_the_old_hard_coded_path():
    root = default_dataset_root('/repo')
    assert root == os.path.join('/repo', 'dataset', 'ZuCo')
    assert pickle_path(root, 'task1-SR') == os.path.join(
        '/repo', 'dataset', 'ZuCo', 'task1-SR', 'pickle', 'task1-SR-dataset.pickle')


def test_200hz_suffix_matches_the_snn_side_contract():
    # EEGSNN/cfg_runner/pickle_naming.py: 'task1' -> 'task1-SR-dataset-200hz.pickle'
    assert pickle_path('r', 'task1-SR', '-dataset-200hz.pickle').endswith(
        os.path.join('task1-SR', 'pickle', 'task1-SR-dataset-200hz.pickle'))


def test_task_pickle_paths_selects_all_five_in_the_historical_order(tmp_path):
    suffix = '-dataset-200hz.pickle'
    expected = [_touch(tmp_path, d, suffix) for _, d in TASK_KEYS]
    assert task_pickle_paths(ALL_TASKS, str(tmp_path), suffix) == expected


def test_task_pickle_paths_substring_match(tmp_path):
    p1 = _touch(tmp_path, 'task1-SR', DEFAULT_PICKLE_SUFFIX)
    _touch(tmp_path, 'task2-NR', DEFAULT_PICKLE_SUFFIX)
    assert task_pickle_paths('task1', str(tmp_path)) == [p1]


def test_task_pickle_paths_names_the_missing_file(tmp_path):
    _touch(tmp_path, 'task1-SR', DEFAULT_PICKLE_SUFFIX)  # 500 Hz only
    with pytest.raises(FileNotFoundError, match='task1-SR-dataset-200hz.pickle'):
        task_pickle_paths('task1', str(tmp_path), '-dataset-200hz.pickle')


def test_task_pickle_paths_rejects_a_name_that_selects_nothing(tmp_path):
    with pytest.raises(ValueError):
        task_pickle_paths('taskX', str(tmp_path))


def test_existing_pickle_paths_skips_missing_tasks_and_other_rates(tmp_path):
    a = _touch(tmp_path, 'task1-SR', '-dataset-200hz.pickle')
    _touch(tmp_path, 'task2-NR', DEFAULT_PICKLE_SUFFIX)
    b = _touch(tmp_path, 'task2-TSR-2.0', '-dataset-200hz.pickle')
    assert existing_pickle_paths(str(tmp_path), '-dataset-200hz.pickle') == [a, b]
