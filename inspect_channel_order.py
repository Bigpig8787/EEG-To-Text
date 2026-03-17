# inspect_channel_order.py
import scipy.io as io
import numpy as np
import os

home = r"D:\EEG-BCI\EEG-To-Text"
v1_dir = os.path.join(home, "dataset", "ZuCo", "task1- SR", "Matlab files")
v1_files = sorted(os.listdir(v1_dir))
f1 = io.loadmat(os.path.join(v1_dir, v1_files[0]), squeeze_me=True, struct_as_record=False)

# 檢查有沒有 channel labels
print("Top-level keys:", [k for k in f1.keys() if not k.startswith('__')])

sent0 = f1['sentenceData'][0]

# 檢查有沒有 channel info
for attr in ['chanlocs', 'channelLocations', 'channels', 'labels', 'chanlabels', 'nbchan']:
    if hasattr(sent0, attr):
        val = getattr(sent0, attr)
        print(f"Found: {attr} = {val}")

# 也看看 mat file 頂層有沒有
for key in f1.keys():
    if 'chan' in key.lower() or 'label' in key.lower() or 'elect' in key.lower():
        print(f"Top-level: {key} = {f1[key]}")

print("\nrawData shape:", sent0.rawData.shape)
print("rawData dtype:", sent0.rawData.dtype)