"""Convert ZuCo v2.0 .mat files to .pickle with rawData extraction.

Supports both v2.0 reading tasks:
  -t task2-NR-2.0   (Normal Reading v2.0,  mat suffix _NR.mat)
  -t task2-TSR-2.0  (Task-Specific Reading v2.0, mat suffix _TSR.mat)
"""

import os
import argparse
import numpy as np
import h5py
import data_loading_helpers_modified as dh
from tqdm import tqdm
import pickle

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--task_name', required=True,
                    choices=['task2-NR-2.0', 'task2-TSR-2.0'],
                    help='choose from {task2-NR-2.0, task2-TSR-2.0}')
args = vars(parser.parse_args())
task_name = args['task_name']

# v2 mat filename suffix: results<SUBJ>_NR.mat / results<SUBJ>_TSR.mat
mat_suffix = task_name.split('-')[1] + '.mat'  # 'NR.mat' or 'TSR.mat'

rootdir = os.path.join(PROJECT_ROOT, "dataset", "ZuCo", task_name, "Matlab files")

print(f'Processing ZuCo {task_name}...')
print(f'Input: {rootdir}')
print(f'Mat suffix filter: *{mat_suffix}')

dataset_dict = {}
for file in tqdm(sorted(os.listdir(rootdir))):
    if file.endswith(mat_suffix):
        file_name = os.path.join(rootdir, file)
        subject = file_name.split("results")[1].split("_")[0]
        if subject == 'YMH':
            continue

        dataset_dict[subject] = []
        f = h5py.File(file_name, 'r')
        sd = f['sentenceData']

        mean_objs = {k: sd[k] for k in ['mean_t1','mean_t2','mean_a1','mean_a2','mean_b1','mean_b2','mean_g1','mean_g2']}
        rawData = sd['rawData']
        contentData = sd['content']
        wordData = sd['word']

        for idx in range(len(rawData)):
            sent_string = dh.load_matlab_string(f[contentData[idx][0]])
            sent_obj = {'content': sent_string}
            sent_obj['sentence_level_EEG'] = {
                k: np.squeeze(f[mean_objs[k][idx][0]][()]) for k in mean_objs
            }

            # raw EEG
            try:
                raw = np.array(f[rawData[idx][0]], dtype=np.float32)
                if raw.ndim == 2:
                    if raw.shape[1] == 105:
                        raw = raw.T
                    if raw.shape[0] == 105:
                        sent_obj['rawData'] = raw
            except Exception as e:
                pass

            sent_obj['word'] = []
            word_data, word_tokens_all, word_tokens_has_fixation, word_tokens_with_mask = \
                dh.extract_word_level_data(f, f[wordData[idx][0]])

            if word_data == {} or len(word_tokens_all) == 0:
                dataset_dict[subject].append(None)
                continue

            for widx in range(len(word_data)):
                d = word_data[widx]
                word_obj = {'content': d['content'], 'nFixations': d['nFix']}
                if 'GD_EEG' in d:
                    gd, ffd, trt = d["GD_EEG"], d["FFD_EEG"], d["TRT_EEG"]
                    bands = ['t1','t2','a1','a2','b1','b2','g1','g2']
                    word_obj['word_level_EEG'] = {
                        'GD':  {f'GD_{b}':  gd[i]  for i, b in enumerate(bands)},
                        'FFD': {f'FFD_{b}': ffd[i] for i, b in enumerate(bands)},
                        'TRT': {f'TRT_{b}': trt[i] for i, b in enumerate(bands)},
                    }
                    sent_obj['word'].append(word_obj)

            sent_obj['word_tokens_has_fixation'] = word_tokens_has_fixation
            sent_obj['word_tokens_with_mask'] = word_tokens_with_mask
            sent_obj['word_tokens_all'] = word_tokens_all
            dataset_dict[subject].append(sent_obj)

if not dataset_dict:
    print('No mat files found'); exit()

output_dir = os.path.join(PROJECT_ROOT, "dataset", "ZuCo", task_name, "pickle")
os.makedirs(output_dir, exist_ok=True)
output_name = f'{task_name}-dataset.pickle'
with open(os.path.join(output_dir, output_name), 'wb') as f:
    pickle.dump(dataset_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Saved: {os.path.join(output_dir, output_name)}')

first = list(dataset_dict.keys())[0]
print(f'Subjects: {list(dataset_dict.keys())}')
print(f'Sentences ({first}): {len(dataset_dict[first])}')
for i, s in enumerate(dataset_dict[first]):
    if s and 'rawData' in s:
        print(f'rawData check: sent[{i}] shape={s["rawData"].shape}')
        break
