"""
Channel-to-brain-region mapping for EGI HydroCel 128 system (105 channels).

Based on EEG2TEXT paper (Liu et al., 2024) Table 1 and ZuCo dataset
(Hollenstein et al., 2018) using EGI HydroCel 128 montage with 23
outer-ring electrodes removed.

Each value is a list of indices into the 105-channel array.
All 105 channels are covered with no overlap.
"""

# 105-channel index for each brain region
BRAIN_REGION_CHANNELS = {
    'prefrontal':      [4, 9, 3, 8, 12, 11, 15, 98, 18, 103, 19, 0, 20, 102, 1, 2, 17, 14, 16, 6, 7, 13, 21, 25, 97, 101],
    'premotor':        [104, 5, 88, 87, 86, 95, 94, 99, 91, 96, 100, 92, 93, 90, 10, 23],
    'brocas':          [22, 28, 27, 26],
    'auditory_assoc':  [32, 30, 31, 35, 36, 38, 46, 47, 52],
    'primary_motor':   [24, 66, 45, 29, 72, 77, 85, 84, 89],
    'primary_sensory': [44, 65, 50, 64, 51, 43, 71, 76, 81, 82, 83],
    'somatic_sensory': [55, 63, 58, 59, 62, 54, 69, 49, 70],
    'auditory':        [48, 75, 80, 41],
    'wernickes':       [33, 34, 42, 39, 37, 40],
    'visual':          [53, 56, 57, 60, 61, 67, 68, 73, 74, 78, 79],
}

# Number of channels per region (for model init)
BRAIN_REGION_CHANNEL_COUNT = {
    region: len(indices) for region, indices in BRAIN_REGION_CHANNELS.items()
}


def split_raw_eeg_by_region(raw_eeg):
    """
    Split a (105, T) raw EEG array into a dict of (C_region, T) arrays.
    
    Args:
        raw_eeg: numpy array or torch tensor of shape (105, T)
    
    Returns:
        dict: {region_name: array of shape (C_region, T)}
    """
    views = {}
    for region, indices in BRAIN_REGION_CHANNELS.items():
        views[region] = raw_eeg[indices]
    return views