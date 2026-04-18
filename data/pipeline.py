import os
import pickle
from typing import List, Dict, Optional

import torch
from torch.utils.data import DataLoader
from transformers import BartTokenizer

from data.dataset import ZuCo_dataset

PICKLE_PATHS = {
    'task1':    './dataset/ZuCo/task1-SR/pickle/task1-SR-dataset.pickle',
    'task2':    './dataset/ZuCo/task2-NR/pickle/task2-NR-dataset.pickle',
    'task3':    './dataset/ZuCo/task3-TSR/pickle/task3-TSR-dataset.pickle',
    'taskNRv2': './dataset/ZuCo/task2-NR-2.0/pickle/task2-NR-2.0-dataset.pickle',
}

TASK_KEYS = {
    'task1':                ['task1'],
    'task1_task2':          ['task1', 'task2'],
    'task1_task2_task3':    ['task1', 'task2', 'task3'],
    'task1_task2_taskNRv2': ['task1', 'task2', 'taskNRv2'],
}

DEFAULT_BANDS = ['_t1', '_t2', '_a1', '_a2', '_b1', '_b2', '_g1', '_g2']


def _collate(batch):
    (ie, sl, am, ami, ti, tm, sent_lbl, se, rv) = zip(*batch)
    rv_batched = {}
    if rv[0]:
        for region in rv[0]:
            rv_batched[region] = torch.stack([r[region] for r in rv])
    return (torch.stack(ie), list(sl), torch.stack(am), torch.stack(ami),
            torch.stack(ti), torch.stack(tm), list(sent_lbl), torch.stack(se), rv_batched)


class EEGDataPipeline:
    """Load ZuCo EEG data and build DataLoaders. Supports disk caching."""

    def __init__(self, task_name='task1_task2_taskNRv2', subject='ALL',
                 eeg_type='GD', bands=None, batch_size=4, num_workers=0,
                 tokenizer=None, custom_pickle_paths=None):
        self.task_name = task_name
        self.subject = subject
        self.eeg_type = eeg_type
        self.bands = bands or DEFAULT_BANDS
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = tokenizer or BartTokenizer.from_pretrained('facebook/bart-large')
        self._pickle_map = custom_pickle_paths or PICKLE_PATHS
        self._raw_dicts = None  # loaded pickle data, cached in memory

    # ── public ─────────────────────────────────────────────────────────────

    def build(self, phases=('train', 'dev', 'test')) -> Dict[str, DataLoader]:
        """Build and return DataLoaders for requested phases."""
        raw = self._get_raw_dicts()
        loaders = {}
        for phase in phases:
            ds = ZuCo_dataset(raw, phase, self.tokenizer,
                              subject=self.subject, eeg_type=self.eeg_type,
                              bands=self.bands, setting='unique_sent')
            shuffle = (phase == 'train')
            loaders[phase] = DataLoader(ds, batch_size=self.batch_size,
                                        shuffle=shuffle, num_workers=self.num_workers,
                                        collate_fn=_collate, drop_last=shuffle)
            print(f'[DataPipeline] {phase}: {len(ds)} samples, {len(loaders[phase])} batches')
        return loaders

    def save_cache(self, path: str):
        """Save loaded pickle dicts to disk for faster future loading."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'task_name': self.task_name, 'subject': self.subject,
                         'eeg_type': self.eeg_type, 'bands': self.bands,
                         'raw_dicts': self._get_raw_dicts()}, f)
        print(f'[DataPipeline] cache saved -> {path}')

    @classmethod
    def load_cache(cls, path: str, batch_size=4, num_workers=0, tokenizer=None):
        """Restore pipeline from a saved cache file."""
        with open(path, 'rb') as f:
            c = pickle.load(f)
        pipeline = cls(task_name=c['task_name'], subject=c['subject'],
                       eeg_type=c['eeg_type'], bands=c['bands'],
                       batch_size=batch_size, num_workers=num_workers, tokenizer=tokenizer)
        pipeline._raw_dicts = c['raw_dicts']
        return pipeline

    @classmethod
    def from_config(cls, cfg: dict, **kw):
        """Construct from a training config dict (as saved by train_multiview.py)."""
        return cls(task_name=cfg.get('task_name', 'task1_task2_taskNRv2'),
                   subject=cfg.get('subjects', 'ALL'),
                   eeg_type=cfg.get('eeg_type', 'GD'),
                   bands=cfg.get('eeg_bands', None), **kw)

    # ── private ────────────────────────────────────────────────────────────

    def _get_raw_dicts(self):
        if self._raw_dicts is None:
            keys = TASK_KEYS[self.task_name]
            self._raw_dicts = []
            for k in keys:
                path = self._pickle_map[k]
                with open(path, 'rb') as f:
                    self._raw_dicts.append(pickle.load(f))
                print(f'[DataPipeline] loaded {k} <- {path}')
        return self._raw_dicts
