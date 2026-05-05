"""Evaluate Multi-View Conformer Translator."""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import pickle
import json
from tqdm import tqdm
from transformers import BartTokenizer, BartForConditionalGeneration

from data.dataset import ZuCo_dataset
from models.multiview import MultiViewConformerTranslator
from metrics import compute_metrics
from config import get_config

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


def eval_model(dataloaders, device, tokenizer, model, teacher_forcing, input_noise,
               output_results_path, output_metrics_path):
    model.eval()
    target_list, pred_list = [], []

    with open(output_results_path, 'w', encoding='utf-8') as f:
        for (_, _, masks, mask_inv, target_ids, _, _, _, raw_views) in tqdm(dataloaders['test'], desc='Eval'):
            if not raw_views:
                continue

            target_ids_batch = target_ids.to(device)
            masks_batch = masks.to(device)
            mask_inv_batch = mask_inv.to(device)
            view_inputs = {k: v.to(device).float() for k, v in raw_views.items()}

            if input_noise:
                view_inputs = {k: torch.rand_like(v) for k, v in view_inputs.items()}

            target_string = tokenizer.batch_decode(target_ids_batch, skip_special_tokens=True)
            target_list.extend(target_string)
            target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100

            with torch.no_grad():
                if teacher_forcing:
                    out = model(view_inputs, masks_batch, mask_inv_batch, target_ids_batch)
                    predictions = out.logits.softmax(dim=-1).topk(1).indices.squeeze(-1)
                else:
                    predictions = model.generate(view_inputs, masks_batch, mask_inv_batch,
                                                 target_ids_batch, max_new_tokens=50, num_beams=5,
                                                 do_sample=False, repetition_penalty=1.5,
                                                 no_repeat_ngram_size=3,
                                                 forced_bos_token_id=tokenizer.bos_token_id,
                                                 early_stopping=True)

            pred_string = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            pred_list.extend(pred_string)

            for i in range(len(target_string)):
                f.write(f'Predicted: {pred_string[i]}\nTrue: {target_string[i]}\n{"="*50}\n\n')

    results = compute_metrics(pred_list, target_list)
    print(f'\ntf={teacher_forcing}, noise={input_noise}')
    for k, v in results.items():
        print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')

    with open(output_metrics_path, 'w') as f:
        json.dump(results, f, indent=4)
    return results


if __name__ == '__main__':
    args = get_config('eval_decoding')
    config = json.load(open(args['config_path']))

    task_name = config['task_name']
    teacher_forcing = eval(args['tf'])
    input_noise = eval(args['noise'])

    np.random.seed(312)
    torch.manual_seed(312)
    device = torch.device(args['cuda'] if torch.cuda.is_available() else 'cpu')

    # data
    whole_dataset_dicts = []
    dd = os.path.join(PROJECT_ROOT, 'dataset', 'ZuCo')
    for key, (task, fname) in {'task1':     ('task1-SR',      'task1-SR-dataset.pickle'),
                                'task2':     ('task2-NR',      'task2-NR-dataset.pickle'),
                                'task3':     ('task3-TSR',     'task3-TSR-dataset.pickle'),
                                'taskNRv2':  ('task2-NR-2.0',  'task2-NR-2.0-dataset.pickle'),
                                'taskTSRv2': ('task2-TSR-2.0', 'task2-TSR-2.0-dataset.pickle')}.items():
        if key in task_name:
            with open(os.path.join(dd, task, 'pickle', fname), 'rb') as f:
                whole_dataset_dicts.append(pickle.load(f))

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    test_set = ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer,
                            subject=config['subjects'], eeg_type=config['eeg_type'],
                            bands=config['eeg_bands'], setting='unique_sent')
    dataloaders = {'test': DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)}

    # model
    bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    model = MultiViewConformerTranslator(bart, d_model=512, n_filters=40, temporal_kernel=200,
                                         pool_stride=100, tokens_per_view=32, n_heads=8,
                                         n_encoder_layers=4, n_global_layers=3, decoder_embedding_size=1024)
    model.load_state_dict(torch.load(args['checkpoint_path'], map_location=device))
    model.to(device)

    os.makedirs('./results', exist_ok=True)
    tag = f'{task_name}-multiview{"_tf" if teacher_forcing else ""}{"_noise" if input_noise else ""}'
    eval_model(dataloaders, device, tokenizer, model, teacher_forcing, input_noise,
               f'./results/{tag}_results.txt', f'./results/{tag}_metrics.json')
