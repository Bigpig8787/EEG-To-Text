"""Sentence-ranking evaluation for MultiViewConformerTranslator.

Neither teacher forcing nor free generation: for each test sample, score the
true sentence and K-1 other test sentences by NLL under the sample's EEG and
report where the true one ranks (top-1 / top-5 / MRR), plus the true sentence's
corpus NLL / perplexity. Run on real EEG and on two controls (rand_like noise,
EEG from another sentence). See `eval/ranking_core.py`.

    python eval_ranking.py -checkpoint <..._merged.pt> -conf config/decoding/<save_name>.json \
        -cuda cuda:1 [--k 50] [--modes real rand shuffle] [--n_samples 0]

Writes `results/<task>-multiview<save_suffix>_ranking.json`.
"""

import argparse
import json
import os
import pickle

import numpy as np
import torch
from transformers import BartTokenizer, BartForConditionalGeneration

from data.dataset import ZuCo_dataset
from data.pickle_paths import default_dataset_root, task_pickle_paths
from eval.ranking import format_table, run_ranking
from eval.ranking_core import MODES
from models.multiview import MultiViewConformerTranslator

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


def get_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('-checkpoint', '--checkpoint_path', required=True)
    p.add_argument('-conf', '--config_path', required=True)
    p.add_argument('-cuda', '--cuda', default='cuda:0')
    p.add_argument('--k', type=int, default=50, help='candidates per sample (true + K-1)')
    p.add_argument('--modes', nargs='+', default=list(MODES), choices=MODES)
    p.add_argument('--seed', type=int, default=312)
    p.add_argument('--n_samples', type=int, default=0, help='0 = whole test set')
    p.add_argument('--chunk', type=int, default=25, help='candidates per decoder pass')
    return vars(p.parse_args())


def build_model(config, checkpoint_path, device, raw_eeg_max_len):
    """Same construction as `eval_multiview.py` (geometry from the training config)."""
    bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    model = MultiViewConformerTranslator(
        bart,
        d_model=config.get('d_model', 512),
        n_filters=config.get('n_filters', 40),
        n_spatial_filters=config.get('n_spatial_filters'),
        temporal_kernel=config.get('temporal_kernel', 200),
        pool_stride=config.get('pool_stride', 50),
        tokens_per_view=config.get('tokens_per_view', 64),
        n_cls_per_view=config.get('n_cls_per_view', 8),
        n_heads=config.get('n_heads', 8),
        n_encoder_layers=config.get('n_encoder_layers', 4),
        n_global_layers=config.get('n_global_layers', 3),
        decoder_embedding_size=1024,
        raw_eeg_max_len=raw_eeg_max_len)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model.to(device).eval()


def main():
    args = get_args()
    config = json.load(open(args['config_path']))
    np.random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    device = torch.device(args['cuda'] if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    raw_eeg_max_len = config.get('raw_eeg_max_len', 5000)
    dd = config.get('dataset_root') or default_dataset_root(PROJECT_ROOT)
    dicts = []
    for pkl in task_pickle_paths(config['task_name'], dd,
                                 config.get('pickle_suffix', '-dataset.pickle')):
        with open(pkl, 'rb') as f:
            dicts.append(pickle.load(f))
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    test_set = ZuCo_dataset(dicts, 'test', tokenizer, subject=config['subjects'],
                            eeg_type=config['eeg_type'], bands=config['eeg_bands'],
                            setting='unique_sent', raw_eeg_max_len=raw_eeg_max_len)

    model = build_model(config, args['checkpoint_path'], device, raw_eeg_max_len)
    encode_fn = lambda views, lens: model.encode(views, raw_eeg_lens=lens)

    results = run_ranking(test_set, encode_fn, model.pretrained, tokenizer, device,
                          modes=args['modes'], k=args['k'], seed=args['seed'],
                          n_samples=args['n_samples'], chunk=args['chunk'])
    print(format_table(results))

    os.makedirs('./results', exist_ok=True)
    suffix = config.get('save_suffix', '') or ''
    out = f'./results/{config["task_name"]}-multiview{suffix}_ranking.json'
    with open(out, 'w') as f:
        json.dump({'checkpoint': args['checkpoint_path'], 'config': args['config_path'],
                   'seed': args['seed'], 'results': results}, f, indent=1)
    print(f'saved {out}')


if __name__ == '__main__':
    main()
