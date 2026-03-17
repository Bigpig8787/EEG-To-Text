import scipy.io as io
import h5py
import os
import json
from glob import glob
from tqdm import tqdm
import numpy as np
import pickle
import argparse

# ---- 路徑設定：指向專案根目錄（util/ 的上一層）----
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

parser = argparse.ArgumentParser(description='Specify task name for converting ZuCo v1.0 Mat file to Pickle')
parser.add_argument('-t', '--task_name', help='name of the task in /dataset/ZuCo, choose from {task1-SR,task2-NR,task3-TSR}', required=True)
args = vars(parser.parse_args())


"""config"""
task_name = args['task_name']

print('##############################')
print(f'start processing ZuCo {task_name}...')

# ---- 使用專案內的 dataset 路徑，注意 "Matlab files" 有空格 ----
input_mat_files_dir = os.path.join(PROJECT_ROOT, "dataset", "ZuCo", task_name, "Matlab files")
output_dir = os.path.join(PROJECT_ROOT, "dataset", "ZuCo", task_name, "pickle")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

"""load files"""
print(f'input dir: {input_mat_files_dir}')

mat_files = os.listdir(input_mat_files_dir)
mat_files = [os.path.join(input_mat_files_dir, mat_file) for mat_file in mat_files]
mat_files = sorted(mat_files)

if len(mat_files) == 0:
    print(f'No mat files found for {task_name}')
    quit()

dataset_dict = {}
for mat_file in tqdm(mat_files):
    subject_name = os.path.basename(mat_file).split('_')[0].replace('results','').strip()
    dataset_dict[subject_name] = []
    
    matdata = io.loadmat(mat_file, squeeze_me=True, struct_as_record=False)['sentenceData']

    for sent in matdata:
        word_data = sent.word
        if not isinstance(word_data, float):
            # sentence level:
            sent_obj = {'content':sent.content}
            sent_obj['sentence_level_EEG'] = {'mean_t1':sent.mean_t1, 'mean_t2':sent.mean_t2, 'mean_a1':sent.mean_a1, 'mean_a2':sent.mean_a2, 'mean_b1':sent.mean_b1, 'mean_b2':sent.mean_b2, 'mean_g1':sent.mean_g1, 'mean_g2':sent.mean_g2}

            # ---- NEW: store sentence-level raw EEG signal ----
            if hasattr(sent, 'rawData') and not isinstance(sent.rawData, float):
                raw = np.array(sent.rawData, dtype=np.float32)
                # v1 shape: (105, T)
                if raw.ndim == 2 and raw.shape[0] == 105:
                    sent_obj['rawData'] = raw
                else:
                    print(f'  unexpected rawData shape {raw.shape}, skipping rawData for this sentence')
            # ---- END NEW ----

            if task_name == 'task1-SR':
                sent_obj['answer_EEG'] = {'answer_mean_t1':sent.answer_mean_t1, 'answer_mean_t2':sent.answer_mean_t2, 'answer_mean_a1':sent.answer_mean_a1, 'answer_mean_a2':sent.answer_mean_a2, 'answer_mean_b1':sent.answer_mean_b1, 'answer_mean_b2':sent.answer_mean_b2, 'answer_mean_g1':sent.answer_mean_g1, 'answer_mean_g2':sent.answer_mean_g2}
            
            # word level:
            sent_obj['word'] = []
            
            word_tokens_has_fixation = [] 
            word_tokens_with_mask = []
            word_tokens_all = []

            for word in word_data:
                word_obj = {'content':word.content}
                word_tokens_all.append(word.content)
                word_obj['nFixations'] = word.nFixations
                if word.nFixations > 0:    
                    word_obj['word_level_EEG'] = {'FFD':{'FFD_t1':word.FFD_t1, 'FFD_t2':word.FFD_t2, 'FFD_a1':word.FFD_a1, 'FFD_a2':word.FFD_a2, 'FFD_b1':word.FFD_b1, 'FFD_b2':word.FFD_b2, 'FFD_g1':word.FFD_g1, 'FFD_g2':word.FFD_g2}}
                    word_obj['word_level_EEG']['TRT'] = {'TRT_t1':word.TRT_t1, 'TRT_t2':word.TRT_t2, 'TRT_a1':word.TRT_a1, 'TRT_a2':word.TRT_a2, 'TRT_b1':word.TRT_b1, 'TRT_b2':word.TRT_b2, 'TRT_g1':word.TRT_g1, 'TRT_g2':word.TRT_g2}
                    word_obj['word_level_EEG']['GD'] = {'GD_t1':word.GD_t1, 'GD_t2':word.GD_t2, 'GD_a1':word.GD_a1, 'GD_a2':word.GD_a2, 'GD_b1':word.GD_b1, 'GD_b2':word.GD_b2, 'GD_g1':word.GD_g1, 'GD_g2':word.GD_g2}
                    sent_obj['word'].append(word_obj)
                    word_tokens_has_fixation.append(word.content)
                    word_tokens_with_mask.append(word.content)
                else:
                    word_tokens_with_mask.append('[MASK]')
                    continue
            
            sent_obj['word_tokens_has_fixation'] = word_tokens_has_fixation
            sent_obj['word_tokens_with_mask'] = word_tokens_with_mask
            sent_obj['word_tokens_all'] = word_tokens_all
            
            dataset_dict[subject_name].append(sent_obj)

        else:
            print(f'missing sent: subj:{subject_name} content:{sent.content}, return None')
            dataset_dict[subject_name].append(None)
            continue

"""output"""
output_name = f'{task_name}-dataset.pickle'

with open(os.path.join(output_dir, output_name), 'wb') as handle:
    pickle.dump(dataset_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('write to:', os.path.join(output_dir, output_name))


"""sanity check"""
with open(os.path.join(output_dir, output_name), 'rb') as handle:
    whole_dataset = pickle.load(handle)
print('subjects:', whole_dataset.keys())

first_subj = list(whole_dataset.keys())[0]
print('num of sent:', len(whole_dataset[first_subj]))

# verify rawData was saved
for i, s in enumerate(whole_dataset[first_subj]):
    if s is not None and 'rawData' in s:
        print(f'rawData sanity check - subj:{first_subj} sent[{i}] rawData shape: {s["rawData"].shape}')
        break
else:
    print('WARNING: no rawData found in any sentence!')