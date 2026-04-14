"""Dataset for EEG pre-training. Only loads sentence-level raw EEG."""

import numpy as np
import torch
from torch.utils.data import Dataset
import pickle

RAW_EEG_MAX_LEN = 5000


class EEGPretrainDataset(Dataset):
    def __init__(self, pickle_paths, split='all'):
        self.samples = []
        for path in pickle_paths:
            print(f'[PretrainDataset] loading {path}')
            with open(path, 'rb') as f:
                data = pickle.load(f)

            subjects = list(data.keys())
            total = len(data[subjects[0]])

            if split == 'all':
                idx_range = range(total)
            elif split == 'train':
                idx_range = range(int(0.8 * total))
            elif split == 'dev':
                t = int(0.8 * total)
                idx_range = range(t, t + int(0.1 * total))
            elif split == 'test':
                d = int(0.8 * total) + int(0.1 * total)
                idx_range = range(d, total)

            for subj in subjects:
                for i in idx_range:
                    sent = data[subj][i]
                    if sent is None or 'rawData' not in sent:
                        continue
                    raw = sent['rawData']
                    if np.isnan(raw).any() or raw.shape[1] < 50:
                        continue
                    self.samples.append(raw)

        print(f'[PretrainDataset] total: {len(self.samples)} (split={split})')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw = self.samples[idx]
        T = raw.shape[1]

        if T < RAW_EEG_MAX_LEN:
            padded = np.zeros((105, RAW_EEG_MAX_LEN), dtype=np.float32)
            padded[:, :T] = raw
            raw = padded
            actual_len = T
        elif T > RAW_EEG_MAX_LEN:
            raw = raw[:, :RAW_EEG_MAX_LEN].copy()
            actual_len = RAW_EEG_MAX_LEN
        else:
            raw = raw.copy()
            actual_len = T

        mean = raw.mean(axis=1, keepdims=True)
        std = raw.std(axis=1, keepdims=True)
        std[std == 0] = 1.0
        raw = (raw - mean) / std

        return torch.from_numpy(raw).float(), actual_len
