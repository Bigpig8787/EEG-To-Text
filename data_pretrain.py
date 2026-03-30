"""
Dataset for EEG pre-training.
Only loads sentence-level raw EEG (rawData), no word-level features needed.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle


RAW_EEG_MAX_LEN = 5000  # pad/truncate to this length


class EEGPretrainDataset(Dataset):
    """
    Dataset for EEG pre-training.
    Each sample is a sentence-level raw EEG signal: (105, T) → (105, RAW_EEG_MAX_LEN).
    """
    def __init__(self, pickle_paths, split='all'):
        """
        Args:
            pickle_paths: list of paths to pickle files
            split: 'all' uses all data (recommended for pre-training),
                   'train'/'dev'/'test' splits 80/10/10
        """
        self.samples = []
        
        for pickle_path in pickle_paths:
            print(f'[PretrainDataset] loading {pickle_path}')
            with open(pickle_path, 'rb') as f:
                dataset_dict = pickle.load(f)
            
            subjects = list(dataset_dict.keys())
            total_sents = len(dataset_dict[subjects[0]])
            
            # determine index range
            if split == 'all':
                idx_range = range(total_sents)
            elif split == 'train':
                idx_range = range(int(0.8 * total_sents))
            elif split == 'dev':
                train_end = int(0.8 * total_sents)
                dev_end = train_end + int(0.1 * total_sents)
                idx_range = range(train_end, dev_end)
            elif split == 'test':
                dev_end = int(0.8 * total_sents) + int(0.1 * total_sents)
                idx_range = range(dev_end, total_sents)
            
            for subj in subjects:
                for i in idx_range:
                    sent = dataset_dict[subj][i]
                    if sent is None:
                        continue
                    if 'rawData' not in sent:
                        continue
                    
                    raw = sent['rawData']  # (105, T)
                    
                    # skip if NaN
                    if np.isnan(raw).any():
                        continue
                    
                    # skip if too short (less than 1 pool_stride worth of data)
                    if raw.shape[1] < 50:
                        continue
                    
                    self.samples.append(raw)
        
        print(f'[PretrainDataset] total samples: {len(self.samples)} (split={split})')
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        raw = self.samples[idx]  # (105, T), numpy float32
        T = raw.shape[1]
        
        # pad or truncate to RAW_EEG_MAX_LEN
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
        
        # z-score normalize per channel
        mean = raw.mean(axis=1, keepdims=True)
        std = raw.std(axis=1, keepdims=True)
        std[std == 0] = 1.0
        raw = (raw - mean) / std
        
        return torch.from_numpy(raw).float(), actual_len
