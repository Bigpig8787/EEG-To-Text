"""Convert ZuCo v1.0 .mat files to .pickle with rawData extraction."""

import scipy.io as io
import os
from tqdm import tqdm
import numpy as np
import pickle
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--task_name', required=True,
                    help='choose from {task1-SR, task2-NR, task3-TSR}')
args = vars(parser.parse_args())

task_name = args['task_name']
input_dir = os.path.join(PROJECT_ROOT, "dataset", "ZuCo", task_name, "Matlab files")
output_dir = os.path.join(PROJECT_ROOT, "dataset", "ZuCo", task_name, "pickle")
os.makedirs(output_dir, exist_ok=True)

print(f'Processing ZuCo {task_name}...')
print(f'Input: {input_dir}')

mat_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)])
if not mat_files:
    print(f'No mat files found'); exit()

dataset_dict = {}
for mat_file in tqdm(mat_files):
    subject = os.path.basename(mat_file).split('_')[0].replace('results','').strip()
    dataset_dict[subject] = []
    matdata = io.loadmat(mat_file, squeeze_me=True, struct_as_record=False)['sentenceData']

    for sent in matdata:
        word_data = sent.word
        if not isinstance(word_data, float):
            sent_obj = {'content': sent.content}
            sent_obj['sentence_level_EEG'] = {
                'mean_t1':sent.mean_t1, 'mean_t2':sent.mean_t2,
                'mean_a1':sent.mean_a1, 'mean_a2':sent.mean_a2,
                'mean_b1':sent.mean_b1, 'mean_b2':sent.mean_b2,
                'mean_g1':sent.mean_g1, 'mean_g2':sent.mean_g2,
            }

            # raw EEG
            if hasattr(sent, 'rawData') and not isinstance(sent.rawData, float):
                raw = np.array(sent.rawData, dtype=np.float32)
                if raw.ndim == 2 and raw.shape[0] == 105:
                    sent_obj['rawData'] = raw

            if task_name == 'task1-SR':
                sent_obj['answer_EEG'] = {
                    'answer_mean_t1':sent.answer_mean_t1, 'answer_mean_t2':sent.answer_mean_t2,
                    'answer_mean_a1':sent.answer_mean_a1, 'answer_mean_a2':sent.answer_mean_a2,
                    'answer_mean_b1':sent.answer_mean_b1, 'answer_mean_b2':sent.answer_mean_b2,
                    'answer_mean_g1':sent.answer_mean_g1, 'answer_mean_g2':sent.answer_mean_g2,
                }

            sent_obj['word'] = []
            word_tokens_has_fixation, word_tokens_with_mask, word_tokens_all = [], [], []

            for word in word_data:
                word_tokens_all.append(word.content)
                word_obj = {'content': word.content, 'nFixations': word.nFixations}
                if word.nFixations > 0:
                    word_obj['word_level_EEG'] = {
                        'FFD': {f'FFD_{b}': getattr(word, f'FFD_{b}') for b in ['t1','t2','a1','a2','b1','b2','g1','g2']},
                        'TRT': {f'TRT_{b}': getattr(word, f'TRT_{b}') for b in ['t1','t2','a1','a2','b1','b2','g1','g2']},
                        'GD':  {f'GD_{b}':  getattr(word, f'GD_{b}')  for b in ['t1','t2','a1','a2','b1','b2','g1','g2']},
                    }
                    sent_obj['word'].append(word_obj)
                    word_tokens_has_fixation.append(word.content)
                    word_tokens_with_mask.append(word.content)
                else:
                    word_tokens_with_mask.append('[MASK]')

            sent_obj['word_tokens_has_fixation'] = word_tokens_has_fixation
            sent_obj['word_tokens_with_mask'] = word_tokens_with_mask
            sent_obj['word_tokens_all'] = word_tokens_all
            dataset_dict[subject].append(sent_obj)
        else:
            dataset_dict[subject].append(None)

output_name = f'{task_name}-dataset.pickle'
with open(os.path.join(output_dir, output_name), 'wb') as f:
    pickle.dump(dataset_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Saved: {os.path.join(output_dir, output_name)}')

# sanity check
first = list(dataset_dict.keys())[0]
print(f'Subjects: {list(dataset_dict.keys())}')
print(f'Sentences ({first}): {len(dataset_dict[first])}')
for i, s in enumerate(dataset_dict[first]):
    if s and 'rawData' in s:
        print(f'rawData check: sent[{i}] shape={s["rawData"].shape}')
        break
