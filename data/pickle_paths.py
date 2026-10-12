"""Where the trainers find their ZuCo pickles.

`train_pretrain.py`, `train_multiview.py` and `eval_multiview.py` each used to
hard-code `<repo>/dataset/ZuCo/<task>/pickle/<task>-dataset.pickle`, which only
reaches the 500 Hz files in this checkout. The training box keeps its data
elsewhere and also has 200 Hz files (`<task>-dataset-200hz.pickle`, written by
`util/resample_pickle.py`). This module lets all three take the root and the
filename suffix as arguments, with defaults that reproduce the old paths.

The layout and the suffix convention match the SNN side's
`EEGSNN/cfg_runner/pickle_naming.py`, so one `dataset_root` + `pickle_suffix`
pair points both pipelines at the same files.

Standard library only, so it can be tested without torch.
"""

import os

DEFAULT_PICKLE_SUFFIX = '-dataset.pickle'

# task_name substring -> canonical task directory, in the order the trainers
# have always loaded them (the order of the concatenated dataset depends on it).
TASK_KEYS = (
    ('task1', 'task1-SR'),
    ('task2', 'task2-NR'),
    ('task3', 'task3-TSR'),
    ('taskNRv2', 'task2-NR-2.0'),
    ('taskTSRv2', 'task2-TSR-2.0'),
)


def default_dataset_root(project_root: str) -> str:
    return os.path.join(project_root, 'dataset', 'ZuCo')


def pickle_path(dataset_root: str, task_dir: str,
                suffix: str = DEFAULT_PICKLE_SUFFIX) -> str:
    """`<root>/<task>/pickle/<task><suffix>`."""
    return os.path.join(dataset_root, task_dir, 'pickle', task_dir + suffix)


def task_pickle_paths(task_name: str, dataset_root: str,
                      suffix: str = DEFAULT_PICKLE_SUFFIX) -> list:
    """Pickles selected by `task_name`, which is matched by substring.

    Raises:
        FileNotFoundError: a selected task has no pickle. Failing here names the
            missing file; the old code failed inside `open()` with the same
            information but after the tokenizer download.
        ValueError: `task_name` selects nothing.
    """
    paths = []
    for key, task_dir in TASK_KEYS:
        if key in task_name:
            p = pickle_path(dataset_root, task_dir, suffix)
            if not os.path.isfile(p):
                raise FileNotFoundError(f'Missing pickle for {key}: {p}')
            paths.append(p)
    if not paths:
        raise ValueError(f'No task matched task_name={task_name!r}')
    return paths


def existing_pickle_paths(dataset_root: str,
                          suffix: str = DEFAULT_PICKLE_SUFFIX) -> list:
    """Every task pickle that exists, for MAE pre-training (uses all tasks)."""
    paths = [pickle_path(dataset_root, task_dir, suffix) for _, task_dir in TASK_KEYS]
    return [p for p in paths if os.path.isfile(p)]
