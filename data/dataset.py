"""
ZuCo Dataset for EEG-to-Text decoding.
Supports both word-level features (for baseline) and sentence-level raw EEG (for multi-view).
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from data.channel_mapping import split_raw_eeg_by_region

RAW_EEG_MAX_LEN = 5000  # pad/truncate raw EEG to this length


def normalize_1d(input_tensor):
    mean = torch.mean(input_tensor)
    std = torch.std(input_tensor)
    if std == 0:
        return input_tensor - mean
    return (input_tensor - mean) / std


def get_input_sample(sent_obj, tokenizer, eeg_type='GD',
                     bands=('_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'),
                     max_len=56, add_CLS_token=False):

    def get_word_embedding_eeg_tensor(word_obj, eeg_type, bands):
        frequency_features = []
        for band in bands:
            frequency_features.append(word_obj['word_level_EEG'][eeg_type][eeg_type+band])
        word_eeg_embedding = np.concatenate(frequency_features)
        if len(word_eeg_embedding) != 105 * len(bands):
            return None
        return normalize_1d(torch.from_numpy(word_eeg_embedding))

    def get_sent_eeg(sent_obj, bands):
        sent_eeg_features = []
        for band in bands:
            sent_eeg_features.append(sent_obj['sentence_level_EEG']['mean' + band])
        sent_eeg_embedding = np.concatenate(sent_eeg_features)
        assert len(sent_eeg_embedding) == 105 * len(bands)
        return normalize_1d(torch.from_numpy(sent_eeg_embedding))

    if sent_obj is None:
        return None

    input_sample = {}

    # target text
    target_string = sent_obj['content']
    target_tokenized = tokenizer(target_string, padding='max_length', max_length=max_len,
                                 truncation=True, return_tensors='pt', return_attention_mask=True)
    input_sample['target_ids'] = target_tokenized['input_ids'][0]

    # sentence-level EEG features (frequency bands)
    sent_level_eeg_tensor = get_sent_eeg(sent_obj, bands)
    if torch.isnan(sent_level_eeg_tensor).any():
        return None
    input_sample['sent_level_EEG'] = sent_level_eeg_tensor

    # sentiment label (dummy)
    input_sample['sentiment_label'] = torch.tensor(-100)

    # word-level embeddings
    word_embeddings = []
    if add_CLS_token:
        word_embeddings.append(torch.ones(105 * len(bands)))

    for word in sent_obj['word']:
        wt = get_word_embedding_eeg_tensor(word, eeg_type, bands=bands)
        if wt is None:
            return None
        if torch.isnan(wt).any():
            return None
        word_embeddings.append(wt)

    while len(word_embeddings) < max_len:
        word_embeddings.append(torch.zeros(105 * len(bands)))

    input_sample['input_embeddings'] = torch.stack(word_embeddings)

    # attention masks
    n_words = len(sent_obj['word']) + (1 if add_CLS_token else 0)
    input_sample['input_attn_mask'] = torch.zeros(max_len)
    input_sample['input_attn_mask'][:n_words] = 1.0
    input_sample['input_attn_mask_invert'] = torch.ones(max_len)
    input_sample['input_attn_mask_invert'][:n_words] = 0.0

    input_sample['target_mask'] = target_tokenized['attention_mask'][0]
    input_sample['seq_len'] = len(sent_obj['word'])

    if input_sample['seq_len'] == 0:
        return None

    # ---- raw EEG for multi-view model ----
    if 'rawData' in sent_obj:
        raw_eeg = sent_obj['rawData']  # (105, T)
        if np.isnan(raw_eeg).any():
            return None

        T = raw_eeg.shape[1]
        if T < RAW_EEG_MAX_LEN:
            padded = np.zeros((105, RAW_EEG_MAX_LEN), dtype=np.float32)
            padded[:, :T] = raw_eeg
            raw_eeg = padded
        elif T > RAW_EEG_MAX_LEN:
            raw_eeg = raw_eeg[:, :RAW_EEG_MAX_LEN]

        # z-score per channel
        raw_eeg = raw_eeg.astype(np.float32)
        mean = raw_eeg.mean(axis=1, keepdims=True)
        std = raw_eeg.std(axis=1, keepdims=True)
        std[std == 0] = 1.0
        raw_eeg = (raw_eeg - mean) / std

        raw_views = split_raw_eeg_by_region(raw_eeg)
        input_sample['raw_eeg_views'] = {
            region: torch.from_numpy(arr.copy()).float()
            for region, arr in raw_views.items()
        }

    return input_sample


class ZuCo_dataset(Dataset):
    def __init__(self, input_dataset_dicts, phase, tokenizer, subject='ALL',
                 eeg_type='GD', bands=('_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'),
                 setting='unique_sent', is_add_CLS_token=False):
        self.inputs = []
        self.tokenizer = tokenizer

        if not isinstance(input_dataset_dicts, list):
            input_dataset_dicts = [input_dataset_dicts]

        print(f'[INFO] loading {len(input_dataset_dicts)} task datasets')
        for input_dataset_dict in input_dataset_dicts:
            if subject == 'ALL':
                subjects = list(input_dataset_dict.keys())
                print('[INFO] subjects:', subjects)
            else:
                subjects = [subject]

            total = len(input_dataset_dict[subjects[0]])
            train_div = int(0.8 * total)
            dev_div = train_div + int(0.1 * total)

            if setting == 'unique_sent':
                if phase == 'train':
                    idx_range = range(train_div)
                elif phase == 'dev':
                    idx_range = range(train_div, dev_div)
                elif phase == 'test':
                    idx_range = range(dev_div, total)
            elif setting == 'unique_subj':
                idx_range = range(total)
                if phase == 'train':
                    subjects = ['ZAB','ZDM','ZGW','ZJM','ZJN','ZJS','ZKB','ZKH','ZKW']
                elif phase == 'dev':
                    subjects = ['ZMG']
                elif phase == 'test':
                    subjects = ['ZPH']

            print(f'[INFO] initializing {phase} set...')
            for key in subjects:
                for i in idx_range:
                    sample = get_input_sample(input_dataset_dict[key][i], self.tokenizer,
                                              eeg_type, bands=bands, add_CLS_token=is_add_CLS_token)
                    if sample is not None:
                        self.inputs.append(sample)

            print(f'  ++ now have {len(self.inputs)} samples')

        if len(self.inputs) > 0:
            print(f'[INFO] input tensor size: {self.inputs[0]["input_embeddings"].size()}')
            if 'raw_eeg_views' in self.inputs[0]:
                views = self.inputs[0]['raw_eeg_views']
                print(f'[INFO] raw_eeg_views available: {list(views.keys())}')

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        s = self.inputs[idx]
        return (
            s['input_embeddings'],
            s['seq_len'],
            s['input_attn_mask'],
            s['input_attn_mask_invert'],
            s['target_ids'],
            s['target_mask'],
            s['sentiment_label'],
            s['sent_level_EEG'],
            s.get('raw_eeg_views', {}),
        )
