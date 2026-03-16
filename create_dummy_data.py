import os
import pickle
import numpy as np

num_sentences = 100
num_words     = 8
num_channels  = 105
bands = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2']
eeg_type = 'GD'
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

dataset_dict = {}
for subj in dummy_subjects:
    sentences = []
    for i in range(num_sentences):
        text = dummy_sentences_text[i]
        words = text.split()
        word_list = []
        for w in words:
            word_eeg = {eeg_type: {}}
            for band in bands:
                word_eeg[eeg_type][eeg_type + band] = np.random.randn(num_channels).astype(np.float32)
            word_list.append({'content': w, 'word_level_EEG': word_eeg})

        sent_level_eeg = {}
        for band in bands:
            sent_level_eeg['mean' + band] = np.random.randn(num_channels).astype(np.float32)

        sentences.append({
            'content': text,
            'word': word_list,
            'sentence_level_EEG': sent_level_eeg
        })
    dataset_dict[subj] = sentences

# ── 路徑以 train_decoding.py 為準 ──────────────────────
home = os.path.expanduser("~")
save_dir = os.path.join(home, 'datasets', 'ZuCo', 'task1-SR', 'pickle')
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, 'task1-SR-dataset.pickle')

with open(save_path, 'wb') as f:
    pickle.dump(dataset_dict, f)

print(f"✅ 假資料建立完成：{save_path}")
