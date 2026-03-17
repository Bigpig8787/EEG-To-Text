# inspect_channel_mapping.py
# EGI HydroCel 128 去掉的 23 個外圈電極
# 參考 ZuCo paper (Hollenstein et al., 2018)
removed = [1, 8, 14, 17, 21, 25, 32, 48, 49, 56, 63, 68, 73, 81, 88, 94, 99, 107, 113, 119, 125, 126, 127]
# 加上 reference electrode Cz (= 128 在 EGI 系統中通常是 Cz/REF)

kept = [i for i in range(1, 129) if i not in removed]
print(f"Number of kept channels: {len(kept)}")
print(f"Kept channels: {kept}")

# 建立 electrode name -> 105-index mapping
egi_to_idx = {}
for idx_105, egi_num in enumerate(kept):
    egi_to_idx[f'E{egi_num}'] = idx_105

# 特殊: CZ 在 EGI 128 系統中
# 有些定義 CZ=Ref (不在 128 裡), 有些定義為 E129
# 先印出來看
print(f"\nE128 in kept? {128 in [e for e in kept]}")
print(f"\nFirst 10 mappings:")
for k in list(egi_to_idx.keys())[:10]:
    print(f"  {k} -> index {egi_to_idx[k]}")