"""
ZuCo Dataset for EEG-to-Text decoding.
Supports both word-level features (for baseline) and sentence-level raw EEG (for multi-view).
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from data.channel_mapping import split_raw_eeg_by_region, BRAIN_REGION_CHANNEL_COUNT

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
        expected = 105 * len(bands)
        frequency_features = []
        for band in bands:
            try:
                feat = word_obj['word_level_EEG'][eeg_type][eeg_type+band]
            except (KeyError, TypeError):
                feat = np.zeros(105, dtype=np.float32)
            frequency_features.append(feat)
        word_eeg_embedding = np.concatenate(frequency_features)
        if len(word_eeg_embedding) != expected:
            word_eeg_embedding = np.zeros(expected, dtype=np.float32)
        word_eeg_embedding = np.nan_to_num(word_eeg_embedding, nan=0.0, posinf=0.0, neginf=0.0)
        return normalize_1d(torch.from_numpy(word_eeg_embedding))

    def get_sent_eeg(sent_obj, bands):
        expected = 105 * len(bands)
        sent_eeg_features = []
        for band in bands:
            try:
                feat = sent_obj['sentence_level_EEG']['mean' + band]
            except (KeyError, TypeError):
                feat = np.zeros(105, dtype=np.float32)
            sent_eeg_features.append(feat)
        sent_eeg_embedding = np.concatenate(sent_eeg_features)
        if len(sent_eeg_embedding) != expected:
            sent_eeg_embedding = np.zeros(expected, dtype=np.float32)
        sent_eeg_embedding = np.nan_to_num(sent_eeg_embedding, nan=0.0, posinf=0.0, neginf=0.0)
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
    sent_level_eeg_tensor = torch.nan_to_num(sent_level_eeg_tensor, nan=0.0, posinf=0.0, neginf=0.0)
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
            wt = torch.zeros(105 * len(bands))
        wt = torch.nan_to_num(wt, nan=0.0, posinf=0.0, neginf=0.0)
        word_embeddings.append(wt)

    while len(word_embeddings) < max_len:
        word_embeddings.append(torch.zeros(105 * len(bands)))
    # Truncate oversize sentences so torch.stack produces uniform (max_len, D).
    word_embeddings = word_embeddings[:max_len]

    input_sample['input_embeddings'] = torch.stack(word_embeddings)

    # attention masks
    n_words = min(len(sent_obj['word']) + (1 if add_CLS_token else 0), max_len)
    input_sample['input_attn_mask'] = torch.zeros(max_len)
    input_sample['input_attn_mask'][:n_words] = 1.0
    input_sample['input_attn_mask_invert'] = torch.ones(max_len)
    input_sample['input_attn_mask_invert'][:n_words] = 0.0

    input_sample['target_mask'] = target_tokenized['attention_mask'][0]
    input_sample['seq_len'] = max(min(len(sent_obj['word']), max_len), 1)

    # ---- raw EEG for multi-view model ----
    # Always emit the full 10-region dict so DataLoader.collate can batch;
    # zero-fill when rawData is missing so this sample is kept (consistent
    # with the zero-imputation policy used elsewhere).
    raw = sent_obj.get('rawData', None)
    if raw is not None:
        raw_eeg = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        T = raw_eeg.shape[1]
        actual_T = min(T, RAW_EEG_MAX_LEN)
        if T < RAW_EEG_MAX_LEN:
            padded = np.zeros((105, RAW_EEG_MAX_LEN), dtype=np.float32)
            padded[:, :T] = raw_eeg
            raw_eeg = padded
        elif T > RAW_EEG_MAX_LEN:
            raw_eeg = raw_eeg[:, :RAW_EEG_MAX_LEN]

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
        input_sample['raw_eeg_len'] = actual_T
    else:
        input_sample['raw_eeg_views'] = {
            region: torch.zeros(ch_count, RAW_EEG_MAX_LEN, dtype=torch.float32)
            for region, ch_count in BRAIN_REGION_CHANNEL_COUNT.items()
        }
        input_sample['raw_eeg_len'] = 0

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
            dev_div   = int(0.9 * total)

            if setting == 'unique_sent':
                if phase == 'train':
                    idx_range = range(0, train_div)
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
            s.get('raw_eeg_len', 0),
        )
