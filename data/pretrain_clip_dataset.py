"""Dataset for CLIP-style EEG↔Text contrastive pre-training.

Yields (raw_eeg, actual_len, sentence_text). Falls back to empty string when
sentence content is missing in the pickle.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import pickle

RAW_EEG_MAX_LEN = 5000


class EEGPretrainCLIPDataset(Dataset):
    def __init__(self, pickle_paths, split='all'):
        self.samples = []  # list of (raw_eeg ndarray, sentence_str)
        for path in pickle_paths:
            print(f'[CLIPPretrainDataset] loading {path}')
            with open(path, 'rb') as f:
                data = pickle.load(f)

            subjects = list(data.keys())
            total = len(data[subjects[0]])

            if split == 'all':
                idx_range = range(total)
            else:
                cut = int(total * 0.8)
                idx_range = range(cut) if split == 'train' else range(cut, total)

            for subj in subjects:
                for i in idx_range:
                    sent = data[subj][i]
                    if sent is None or 'rawData' not in sent:
                        continue
                    raw = sent['rawData']
                    if raw.shape[1] < 50:
                        continue
                    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
                    text = sent.get('content', '') if isinstance(sent, dict) else ''
                    if not isinstance(text, str):
                        text = str(text)
                    self.samples.append((raw, text))

        print(f'[CLIPPretrainDataset] total: {len(self.samples)} (split={split})')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw, text = self.samples[idx]
        T = raw.shape[1]

        if T < RAW_EEG_MAX_LEN:
            padded = np.zeros((105, RAW_EEG_MAX_LEN), dtype=np.float32)
            padded[:, :T] = raw
            raw = padded
            actual_len = T
        elif T > RAW_EEG_MAX_LEN:
            raw = raw[:, :RAW_EEG_MAX_LEN]
            actual_len = RAW_EEG_MAX_LEN
        else:
            actual_len = T

        return torch.from_numpy(raw).float(), actual_len, text
