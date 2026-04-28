import pickle, os, numpy as np

root = r"D:\EEG-BCI\EEG-To-Text\dataset\ZuCo"
path = os.path.join(root, "task1-SR", "pickle", "task1-SR-dataset.pickle")

with open(path, 'rb') as f:
    data = pickle.load(f)

subj = list(data.keys())[0]
sent = None
for s in data[subj]:
    if s is not None and 'rawData' in s:
        sent = s
        break

raw = sent['rawData']  # (105, T)
print(f"shape: {raw.shape}")
print(f"dtype: {raw.dtype}")
print(f"min: {raw.min():.4f}, max: {raw.max():.4f}")
print(f"mean: {raw.mean():.4f}, std: {raw.std():.4f}")
print(f"content: {sent['content'][:80]}")

# 看看數值範圍，判斷是否是 microvolt 級別的原始 EEG
# 原始 EEG 通常在 ±100 μV 範圍
# 如果是頻帶功率或其他特徵，數值範圍會不同
print(f"\nChannel 0 前 20 個 time points:")
print(raw[0, :20])

print(f"\nChannel 0 統計:")
print(f"  mean: {raw[0].mean():.4f}")
print(f"  std:  {raw[0].std():.4f}")
print(f"  min:  {raw[0].min():.4f}")
print(f"  max:  {raw[0].max():.4f}")