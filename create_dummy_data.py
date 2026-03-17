import os
import pickle
import numpy as np

# 腦區分組（各區 channel 數）
BRAIN_REGION_CHANNEL_COUNT = {
    'prefrontal': 26, 'premotor': 16, 'brocas': 4,
    'auditory_assoc': 9, 'primary_motor': 9, 'primary_sensory': 11,
    'somatic_sensory': 9, 'auditory': 4, 'wernickes': 6, 'visual': 11,
}
T = 512  # 固定時間長度

num_sentences  = 100
num_channels   = 105
num_bands      = 8
eeg_dim        = num_channels * num_bands  # 840，給原版 baseline 用
dummy_subjects = ['ZAB', 'ZDM', 'ZGW']
dummy_sentences_text = [
    "The cat sat on the mat .",
    "A dog ran across the field .",
    "She read the book carefully .",
    "He won a Nobel Prize in Chemistry .",
    "The movie was surprisingly good .",
    "Scientists discovered a new species .",
    "The weather today is quite pleasant .",
    "Children played in the park happily .",
    "The train arrived at the station .",
    "She wrote a letter to her friend .",
] * 10

bands    = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2']
eeg_type = 'GD'

dataset_dict = {}
for subj in dummy_subjects:
    sentences = []
    for i in range(num_sentences):
        text  = dummy_sentences_text[i]
        words = text.split()

        # word-level EEG（給原版 baseline 用）
        word_list = []
        for w in words:
            word_eeg = {eeg_type: {}}
            for band in bands:
                word_eeg[eeg_type][eeg_type + band] = np.random.randn(num_channels).astype(np.float32)
            word_list.append({'content': w, 'word_level_EEG': word_eeg})

        # sentence-level EEG（給 MultiView 用）
        sent_level_eeg = {}
        for band in bands:
            sent_level_eeg['mean' + band] = np.random.randn(num_channels).astype(np.float32)

        # raw EEG per brain region（給 MultiView EEGNet 用）
        raw_eeg_views = {
            region: np.random.randn(ch_count, T).astype(np.float32)
            for region, ch_count in BRAIN_REGION_CHANNEL_COUNT.items()
        }

        sentences.append({
            'content':            text,
            'word':               word_list,
            'sentence_level_EEG': sent_level_eeg,
            'raw_eeg_views':      raw_eeg_views,   # 新增欄位
        })
    dataset_dict[subj] = sentences

home      = os.path.expanduser("~")
save_dir  = os.path.join(home, 'datasets', 'ZuCo', 'task1-SR', 'pickle')
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'task1-SR-dataset.pickle')

with open(save_path, 'wb') as f:
    pickle.dump(dataset_dict, f)

print(f"✅ 假資料建立完成：{save_path}")