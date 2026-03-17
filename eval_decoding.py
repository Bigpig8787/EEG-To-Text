import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import pickle
import json
import matplotlib.pyplot as plt
from glob import glob
import time
import copy
from tqdm import tqdm
import re
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig, BertTokenizer
from data import ZuCo_dataset
from model_decoding import BrainTranslator, BrainTranslatorNaive, MultiViewBrainTranslator
from metrics import compute_metrics
from config import get_config

# ---- 路徑設定 ----
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


def eval_model(dataloaders, device, tokenizer, criterion, model, model_name,
               output_all_results_path='./results/temp.txt'):
    # modified from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

    model.eval()  # Set model to evaluate mode
    running_loss = 0.0

    target_tokens_list = []
    target_string_list = []
    pred_tokens_list = []
    pred_string_list = []
    with open(output_all_results_path, 'w') as f:
        for (input_embeddings, seq_len, input_masks, input_mask_invert, 
             target_ids, target_mask, sentiment_labels, sent_level_EEG, raw_eeg_views) in \
        dataloaders['test']:

            # load in batch
            input_embeddings_batch = input_embeddings.to(device).float()
            input_masks_batch = input_masks.to(device)
            target_ids_batch = target_ids.to(device)
            input_mask_invert_batch = input_mask_invert.to(device)

            if intput_noise:
                input_embeddings_batch = torch.rand_like(input_embeddings_batch)

            target_string = tokenizer.batch_decode(target_ids_batch, skip_special_tokens=True)
            target_string_list.extend(target_string)

            """replace padding ids in target_ids with -100"""
            target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100

            # ---- forward: dispatch by model type ----
            if model_name == 'MultiViewBrainTranslator':
                view_inputs = {k: v.to(device).float() for k, v in raw_eeg_views.items()}
                if not teacher_forcing:
                    predictions = model.generate(
                        view_inputs, input_masks_batch, input_mask_invert_batch,
                        target_ids_batch,
                        max_length=100, num_beams=5, do_sample=False, 
                        repetition_penalty=5.0,
                    )
                else:
                    seq2seqLMoutput = model(view_inputs, input_masks_batch,
                                            input_mask_invert_batch, target_ids_batch)
                    logits = seq2seqLMoutput.logits
                    probs = logits.softmax(dim=-1)
                    values, predictions = probs.topk(1)
                    predictions = torch.squeeze(predictions, dim=-1)
            else:
                if not teacher_forcing:
                    predictions = model.generate(
                        input_embeddings_batch, input_masks_batch, input_mask_invert_batch,
                        target_ids_batch,
                        max_length=100, num_beams=5, do_sample=False,
                        repetition_penalty=5.0,
                    )
                else:
                    seq2seqLMoutput = model(input_embeddings_batch, input_masks_batch,
                                            input_mask_invert_batch, target_ids_batch)
                    logits = seq2seqLMoutput.logits
                    probs = logits.softmax(dim=-1)
                    values, predictions = probs.topk(1)
                    predictions = torch.squeeze(predictions, dim=-1)

            predicted_string = tokenizer.batch_decode(predictions, skip_special_tokens=True)

            for str_id in range(len(target_string)):
                f.write(f'start################################################\n')
                f.write(f'Predicted: {predicted_string[str_id]}\n')
                f.write(f'True: {target_string[str_id]}\n')
                f.write(f'end################################################\n\n\n')

            pred_string_list.extend(predicted_string)

    metrics_results = compute_metrics(pred_string_list, target_string_list)
    print(f'teacher_forcing{teacher_forcing} intput_noise{intput_noise}')
    print(metrics_results)
    print(output_all_results_path)
    print(output_all_metrics_results_path)
    with open(output_all_metrics_results_path, "w") as json_file:
        json.dump(metrics_results, json_file, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    ''' get args'''
    args = get_config('eval_decoding')

    ''' load training config'''
    training_config = json.load(open(args['config_path']))

    batch_size = 1

    subject_choice = training_config['subjects']
    print(f'[INFO]subjects: {subject_choice}')
    eeg_type_choice = training_config['eeg_type']
    print(f'[INFO]eeg type: {eeg_type_choice}')
    bands_choice = training_config['eeg_bands']
    print(f'[INFO]using bands: {bands_choice}')

    dataset_setting = 'unique_sent'

    task_name = training_config['task_name']

    model_name = training_config['model_name']

    teacher_forcing = eval(args['tf'])
    intput_noise = eval(args['noise'])
    print(f'teacher_forcing{teacher_forcing} intput_noise{intput_noise}')

    os.makedirs('./results', exist_ok=True)
    output_all_results_path = (f'./results/{task_name}-{model_name}{"-teacher_forcing" if teacher_forcing else ""}{"-intput_noise" if intput_noise else ""}-all_decoding_results.txt')
    output_all_metrics_results_path = output_all_results_path.replace('txt', 'json')

    ''' set random seeds '''
    seed_val = 312
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    ''' set up device '''
    if torch.cuda.is_available():
        dev = args['cuda']
    else:
        dev = "cpu"
    device = torch.device(dev)
    print(f'[INFO]using device {dev}')

    ''' set up dataloader '''
    whole_dataset_dicts = []
    if 'task1' in task_name:
        dataset_path_task1 = os.path.join(PROJECT_ROOT, 'dataset', 'ZuCo', 'task1-SR', 'pickle', 'task1-SR-dataset.pickle')
        with open(dataset_path_task1, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task2' in task_name:
        dataset_path_task2 = os.path.join(PROJECT_ROOT, 'dataset', 'ZuCo', 'task2-NR', 'pickle', 'task2-NR-dataset.pickle')
        with open(dataset_path_task2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task3' in task_name:
        dataset_path_task3 = os.path.join(PROJECT_ROOT, 'dataset', 'ZuCo', 'task3-TSR', 'pickle', 'task3-TSR-dataset.pickle')
        with open(dataset_path_task3, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'taskNRv2' in task_name:
        dataset_path_taskNRv2 = os.path.join(PROJECT_ROOT, 'dataset', 'ZuCo', 'task2-NR-2.0', 'pickle', 'task2-NR-2.0-dataset.pickle')
        with open(dataset_path_taskNRv2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    print()

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    # test dataset
    test_set = ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer, subject=subject_choice,
                            eeg_type=eeg_type_choice, bands=bands_choice, setting=dataset_setting)

    dataset_sizes = {"test_set": len(test_set)}
    print('[INFO]test_set size: ', len(test_set))

    # dataloaders
    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
    dataloaders = {'test': test_dataloader}

    ''' set up model '''
    checkpoint_path = args['checkpoint_path']
    pretrained_bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

    if model_name == 'BrainTranslator':
        model = BrainTranslator(pretrained_bart, in_feature=105*len(bands_choice),
                                decoder_embedding_size=1024, additional_encoder_nhead=8,
                                additional_encoder_dim_feedforward=2048)
    elif model_name == 'BrainTranslatorNaive':
        model = BrainTranslatorNaive(pretrained_bart, in_feature=105*len(bands_choice),
                                     decoder_embedding_size=1024, additional_encoder_nhead=8,
                                     additional_encoder_dim_feedforward=2048)
    elif model_name == 'MultiViewBrainTranslator':
        model = MultiViewBrainTranslator(
            pretrained_bart,
            d_model=512, nhead=8, num_layers=6,
            decoder_embedding_size=1024
        )

    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    ''' eval '''
    eval_model(dataloaders, device, tokenizer, criterion, model, model_name,
               output_all_results_path=output_all_results_path)