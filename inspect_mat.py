# inspect_mat.py
import scipy.io as io
import h5py
import numpy as np
import os

home = r"D:\EEG-BCI\EEG-To-Text"

# === 檢查 v1 ===
print("=== ZuCo v1 (task1-SR) ===")
v1_dir = os.path.join(home, "dataset", "ZuCo", "task1- SR", "Matlab files")
v1_files = sorted(os.listdir(v1_dir))
if v1_files:
    f1 = io.loadmat(os.path.join(v1_dir, v1_files[0]), squeeze_me=True, struct_as_record=False)
    sent0 = f1['sentenceData'][0]
    print("sentence fields:", dir(sent0))
    print("content:", sent0.content)
    
    # 檢查有沒有 rawData
    if hasattr(sent0, 'rawData'):
        rd = sent0.rawData
        print("rawData type:", type(rd))
        print("rawData shape:", np.array(rd).shape if hasattr(rd, '__len__') else "scalar")
    else:
        print("NO rawData field!")
    
    # 檢查 word level 有沒有 raw
    word0 = sent0.word[0] if not isinstance(sent0.word, float) else None
    if word0:
        print("\nword fields:", dir(word0))
        if hasattr(word0, 'rawEEG'):
            print("word rawEEG shape:", np.array(word0.rawEEG).shape)
        if hasattr(word0, 'rawET'):
            print("word rawET shape:", np.array(word0.rawET).shape)

# === 檢查 v2 ===
print("\n=== ZuCo v2 (task2-NR-2.0) ===")
v2_dir = os.path.join(home, "dataset", "ZuCo", "task2 - NR-2.0", "Matlab files")
v2_files = [f for f in sorted(os.listdir(v2_dir)) if f.endswith("NR.mat")]
if v2_files:
    f2 = h5py.File(os.path.join(v2_dir, v2_files[0]), 'r')
    sd = f2['sentenceData']
    print("sentenceData keys:", list(sd.keys()))
    
    # 檢查 rawData
    if 'rawData' in sd:
        raw_ref = sd['rawData'][0][0]
        raw_arr = np.array(f2[raw_ref])
        print("rawData shape (sent 0):", raw_arr.shape)
    
    # 檢查 word level
    if 'word' in sd:
        word_ref = sd['word'][0][0]
        word_group = f2[word_ref]
        print("word keys:", list(word_group.keys()))
        if 'rawEEG' in word_group:
            w_raw_ref = word_group['rawEEG'][0][0]
            w_raw = np.array(f2[w_raw_ref])
            print("word rawEEG shape:", w_raw.shape)
    f2.close()