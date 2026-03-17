"""
Channel-to-brain-region mapping for ZuCo 105-channel EGI HydroCel system.
Based on EEG2TEXT (Liu et al., 2024) Table 1.

ZuCo uses 105 of the 128 EGI channels. The assumed ordering in rawData is:
  E2, E3, E4, ..., E124, CZ  (sorted by electrode number, skipping removed channels)

Removed from original 128: E1, E8, E14, E17, E21, E25, E32, E48, E49,
  E56, E63, E68, E73, E81, E88, E94, E99, E107, E113, E119
"""

# 105 channels in order (index 0–104)
CHANNEL_ORDER = [
    'E2',  'E3',  'E4',  'E5',  'E6',  'E7',  'E9',  'E10', 'E11', 'E12',   # 0-9
    'E13', 'E15', 'E16', 'E18', 'E19', 'E20', 'E22', 'E23', 'E24', 'E26',   # 10-19
    'E27', 'E28', 'E29', 'E30', 'E31', 'E33', 'E34', 'E35', 'E36', 'E37',   # 20-29
    'E38', 'E39', 'E40', 'E41', 'E42', 'E43', 'E44', 'E45', 'E46', 'E47',   # 30-39
    'E50', 'E51', 'E52', 'E53', 'E54', 'E55', 'E57', 'E58', 'E59', 'E60',   # 40-49
    'E61', 'E62', 'E64', 'E65', 'E66', 'E67', 'E69', 'E70', 'E71', 'E72',   # 50-59
    'E74', 'E75', 'E76', 'E77', 'E78', 'E79', 'E80', 'E82', 'E83', 'E84',   # 60-69
    'E85', 'E86', 'E87', 'E89', 'E90', 'E91', 'E92', 'E93', 'E95', 'E96',   # 70-79
    'E97', 'E98', 'E100','E101','E102','E103','E104','E105','E106','E108',    # 80-89
    'E109','E110','E111','E112','E114','E115','E116','E117','E118','E120',    # 90-99
    'E121','E122','E123','E124','CZ',                                         # 100-104
]

_name2idx = {name: idx for idx, name in enumerate(CHANNEL_ORDER)}

# Electrode names per brain region (from EEG2TEXT Table 1)
BRAIN_REGION_ELECTRODES = {
    'prefrontal': [
        'E6','E12','E5','E11','E16','E15','E20','E118',
        'E24','E124','E26','E2','E27','E123','E3','E4',
        'E23','E19','E22','E9','E10','E18','E28','E33',
        'E117','E122',
    ],
    'premotor': [
        'CZ','E7','E106','E105','E104','E115','E114','E120',
        'E110','E116','E121','E111','E112','E109','E13','E30',
    ],
    'brocas': ['E29','E36','E35','E34'],
    'auditory_assoc': [
        'E40','E38','E39','E43','E44','E46','E57','E58','E64',
    ],
    'primary_motor': [
        'E31','E80','E55','E37','E87','E93','E103','E102','E108',
    ],
    'primary_sensory': [
        'E54','E79','E61','E78','E62','E53','E86','E92',
        'E98','E100','E101',
    ],
    'somatic_sensory': [
        'E67','E77','E71','E72','E76','E66','E84','E60','E85',
    ],
    'auditory': ['E59','E91','E97','E51'],
    'wernickes': ['E41','E42','E52','E47','E45','E50'],
    'visual': [
        'E65','E69','E70','E74','E75','E82','E83','E89',
        'E90','E95','E96',
    ],
}

# Convert electrode names → channel indices (0-based)
BRAIN_REGION_INDICES = {
    region: sorted([_name2idx[e] for e in electrodes])
    for region, electrodes in BRAIN_REGION_ELECTRODES.items()
}

# Sanity check
assert sum(len(v) for v in BRAIN_REGION_INDICES.values()) == 105
assert sorted(sum(BRAIN_REGION_INDICES.values(), [])) == list(range(105))


if __name__ == '__main__':
    for region, indices in BRAIN_REGION_INDICES.items():
        print(f'{region:20s}  {len(indices):3d} channels  indices: {indices}')